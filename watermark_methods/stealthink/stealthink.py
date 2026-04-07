"""
StealthInk implementation adapted from 

https://github.com/yajiang4215/StealthInk_A-Multi-bit-and-Stealthy-Watermark-for-Large-Language-Models

"""


from __future__ import annotations  
import numpy as np
import math
from scipy.stats import norm
import numpy as np 
import torch 
import random
from watermark_methods.stealthink.hash_scheme import prf_lookup, seeding_scheme_lookup
from transformers import LogitsProcessor

class WatermarkBase:
    def __init__(
        self,
        vocab = None,
        seeding_scheme = "simple_3",  # length of seed is 3
        hash_key = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
    ):

        # watermarking parameters
        self.vocab = vocab
        self.rng = None
        self.vocab_size = len(vocab)
        self.seeding_scheme = seeding_scheme
        self.hash_key = hash_key

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        # print("seeding scheme:", self.seeding_scheme)
        # self.rng = torch.Generator(device=input_ids.device)
        self.rng = torch.Generator(device='cpu') # can also do it on device=input_ids.device (gpu), but needs to guarantee the generation and detection are on the same gpu device
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme
        if seeding_scheme == "simple_1": # length of seed is 1
            assert input_ids.shape[-1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(seeding_scheme)
            prf_key = prf_lookup[self.prf_type](input_ids[-1:], salt_key=self.hash_key)
            # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
            self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long
        elif seeding_scheme == "simple_3": # length of seed is 3
            assert input_ids.shape[-1] >= 3, f"seeding_scheme={seeding_scheme} requires at least a 3 token prefix sequence to seed rng"
            self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(seeding_scheme)
            prf_key = prf_lookup[self.prf_type](input_ids[:, -3:], salt_key=self.hash_key)
            self.rng.manual_seed(prf_key % (2**64 - 1)) 
        return


class ReweightProcessor(WatermarkBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reweight(self, seed, original_token_probs, pos_embedded_message, base): 
        """
        implementing reweight function as fig. 2 in paper
        """
        self._seed_rng(seed)
        vocab_perm = torch.randperm(self.vocab_size, device='cpu', generator=self.rng).detach().cpu().tolist()
        colorlist = torch.chunk(torch.tensor(vocab_perm), base)
        original_probs_tensor = torch.tensor([original_token_probs[tok] for tok in vocab_perm], dtype=torch.float64)

        red_tokens_alpha = 0
        red_tokens_beta = 0
        for i in range(base):
            if i < pos_embedded_message:
                red_tokens_alpha += len(colorlist[i])
            if i == pos_embedded_message:
                red_tokens_beta = red_tokens_alpha + len(colorlist[i])

        if red_tokens_alpha == 0:
            alpha = torch.tensor(0.0, dtype=torch.float64)
        else:
            alpha = original_probs_tensor.cumsum(dim=0)[red_tokens_alpha - 1]
        beta = original_probs_tensor.cumsum(dim=0)[red_tokens_beta - 1]

        acc = torch.zeros_like(original_probs_tensor, dtype=torch.float64)
        acc += original_probs_tensor.cumsum(dim=0)
        acc = torch.cat((torch.tensor([0.0], dtype=torch.float64), acc))

        if alpha >= 0.5 or beta <= 0.5:
            if alpha >= 0.5:  # 2p p 0
                a, b, c, d = 1 - beta, 1 - alpha, alpha, beta
                mapped = torch.where(
                    acc <= a, acc - d,
                    torch.where(
                        acc <= b, 2 * acc - 1,
                        torch.where(
                            acc <= c, acc - c,
                            torch.where(acc <= d, torch.zeros(1, dtype=torch.float64), acc - d),
                        ),
                    ),
                )
            else:  # beta <= 0.5, 0 p 2p
                a, b, c, d = alpha, beta, 1 - beta, 1 - alpha
                mapped = torch.where(
                    acc <= a, acc - a,
                    torch.where(
                        acc <= b,
                        torch.zeros(1, dtype=torch.float64),
                        torch.where(
                            acc <= c,
                            acc - b,
                            torch.where(acc <= d, 2 * acc - 1, acc - a),
                        ),
                    ),
                )
        else:
            if alpha <= 1 - beta <= beta <= 1 - alpha:  # alpha+beta<1 -> 0 p 2p
                a, b, c, d = alpha, 1 - beta, beta, 1 - alpha
                mapped = torch.where(
                    acc <= a, acc - a,
                    torch.where(
                        acc <= b,
                        torch.zeros(1, dtype=torch.float64),
                        torch.where(
                            acc <= c,
                            acc - b,
                            torch.where(acc <= d, 2 * acc - 1, acc - a),
                        ),
                    ),
                )
            else:  # alpha+beta>1 -> 2p p 0
                a, b, c, d = 1 - beta, alpha, 1 - alpha, beta
                mapped = torch.where(
                    acc <= a, acc - d,
                    torch.where(
                        acc <= b,
                        2 * acc - 1,
                        torch.where(
                            acc <= c,
                            acc - c,
                            torch.where(acc <= d, torch.zeros(1, dtype=torch.float64), acc - d),
                        ),
                    ),
                )

        reweighted_probs = mapped[1:] - mapped[:-1]
        combined = {k: v for k, v in zip(vocab_perm, reweighted_probs)}
        sorted_vals = torch.tensor([combined[k] for k in sorted(combined.keys())], dtype=torch.float64)
        v_non_zero = torch.where(sorted_vals > 0, sorted_vals, torch.tensor(1e-50, dtype=torch.float64))
        logits = torch.log(v_non_zero).to(dtype=torch.float32)
        return logits


class SteathInkProcessor(LogitsProcessor):
    def __init__(self, vocab, chunk_capacity, embedded_message, args, cache_max=50000):
        super().__init__()
        self.reweight_processor = ReweightProcessor(vocab=vocab)
        self.n_gram_len = 3
        self.chunk_capacity = chunk_capacity
        self.num_value = 2 ** self.chunk_capacity

        self.base = int(1 / (1.0 / self.num_value))
        self.args = args
        
        self.converted_msg_length = int(self.args.bits / self.chunk_capacity)
        self.embedded_message = embedded_message
        self.output_logits = None
        self.topk_records = {'top1': [], 'top5': [], 'top10': []}
        self.seen_seeds = set()
        self.is_r = False

        # cache: seed_tuple -> (vocab_perm (cpu tensor), colorlist_indices)
        self._perm_cache = {}
        self._cache_max = cache_max

    def _get_perm_and_chunks(self, seed, base, vocab_size):
        seed_tuple = tuple(seed.view(-1).tolist())
        hit = self._perm_cache.get(seed_tuple)
        if hit is not None:
            return hit

        self.reweight_processor._seed_rng(seed)
        vocab_perm = torch.randperm(vocab_size, device='cpu', generator=self.reweight_processor.rng)
        colorlist = torch.chunk(vocab_perm, base)

        if len(self._perm_cache) >= self._cache_max:
            self._perm_cache.clear()
        self._perm_cache[seed_tuple] = (vocab_perm, colorlist)
        return self._perm_cache[seed_tuple]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top1_ids = torch.topk(scores[0], k=1).indices.tolist()
        top5_ids = torch.topk(scores[0], k=5).indices.tolist()
        top10_ids = torch.topk(scores[0], k=10).indices.tolist()


        self.topk_records['top1'].append(top1_ids)
        self.topk_records['top5'].append(top5_ids)
        self.topk_records['top10'].append(top10_ids)
        
        device = scores.device
        logits = scores
        seed = input_ids[:, -self.n_gram_len:]  # [1, H]
        seed_tuple = tuple(seed.view(-1).tolist())

        # skip repeated seed (purely in-memory; no disk I/O)
        if seed_tuple in self.seen_seeds:
            # print("repeated!")
            self.is_r = True
            self.output_logits = logits
            return logits
        self.seen_seeds.add(seed_tuple)
        self.is_r = False

        # bit position from CPU RNG (deterministic)
        self.reweight_processor._seed_rng(seed)
        bit_pos = torch.randint(low=0, high=self.converted_msg_length, size=(1,), generator=self.reweight_processor.rng).item()
        # print("no r, bit_pos in generation:", bit_pos)
        pos_embedded_message = self.embedded_message[bit_pos]

        # probs on the same device
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # [V] on device

        # get vocab perm on CPU, then index on device with a moved view
        vocab_size = probs.shape[-1]
        vocab_perm_cpu, colorlist = self._get_perm_and_chunks(seed, self.base, vocab_size)
        vocab_perm = vocab_perm_cpu.to(device)

        # reorder probs by permutation (vectorized)
        original_probs_tensor = probs.index_select(dim=0, index=vocab_perm).to(torch.float64)

        # compute alpha/beta via cumsum (vectorized)
        cdf = original_probs_tensor.cumsum(dim=0)
        chunk_sizes = [len(t) for t in colorlist]
        red_alpha = sum(chunk_sizes[:pos_embedded_message])
        red_beta  = red_alpha + chunk_sizes[pos_embedded_message]

        alpha = cdf[red_alpha - 1] if red_alpha > 0 else torch.tensor(0.0, dtype=torch.float64, device=device)
        beta  = cdf[red_beta - 1]

        # build acc = [0, cdf]
        acc = torch.cat([torch.zeros(1, dtype=torch.float64, device=device), cdf], dim=0)

        # piecewise mapping (same logic, fully tensorized on device)
        if alpha >= 0.5 or beta <= 0.5:
            if alpha >= 0.5:  # 2p p 0
                a, b, c, d = 1 - beta, 1 - alpha, alpha, beta
                z = torch.where(
                    acc <= a, acc - d,
                    torch.where(acc <= b, 2 * acc - 1,
                    torch.where(acc <= c, acc - c,
                    torch.where(acc <= d, torch.zeros_like(acc), acc - d))))
            else:  # beta <= 0.5 (0 p 2p)
                a, b, c, d = alpha, beta, 1 - beta, 1 - alpha
                z = torch.where(
                    acc <= a, acc - a,
                    torch.where(acc <= b, torch.zeros_like(acc),
                    torch.where(acc <= c, acc - b,
                    torch.where(acc <= d, 2 * acc - 1, acc - a))))
        else:
            if alpha <= 1 - beta <= beta <= 1 - alpha:  # alpha+beta<1 -> 0 p 2p
                a, b, c, d = alpha, 1 - beta, beta, 1 - alpha
                z = torch.where(
                    acc <= a, acc - a,
                    torch.where(acc <= b, torch.zeros_like(acc),
                    torch.where(acc <= c, acc - b,
                    torch.where(acc <= d, 2 * acc - 1, acc - a))))
            else:  # alpha+beta>1 -> 2p p 0
                a, b, c, d = 1 - beta, alpha, 1 - alpha, beta
                z = torch.where(
                    acc <= a, acc - d,
                    torch.where(acc <= b, 2 * acc - 1,
                    torch.where(acc <= c, acc - c,
                    torch.where(acc <= d, torch.zeros_like(acc), acc - d))))

        reweighted_probs = (z[1:] - z[:-1]).clamp_min(1e-50)  # avoid log(0)
        # map back to original vocab order
        logits_out = torch.full_like(probs, fill_value=-1e9, dtype=torch.float32)
        logits_out.index_copy_(0, vocab_perm, reweighted_probs.log().to(torch.float32))
        self.output_logits = logits_out

        return logits_out.unsqueeze(0)  # [1, V]
    
    
class SteathInkDetector(WatermarkBase):
    def __init__(self, vocab, device=None, tokenizer=None, chunk_capacity=None, args=None):

        self.device = device
        self.tokenizer = tokenizer
        
        self.processor = ReweightProcessor(vocab=vocab)
        self.args = args
        
        self.chunk_capacity = chunk_capacity
        self.num_value = 2 ** self.chunk_capacity
        max_len_bits = len(bin(self.num_value - 1)[2:])
        self.binary_mapping = {gamma: bin(gamma)[2:].zfill(max_len_bits) for gamma in range(self.num_value)}
        
        self.converted_msg_length = int(self.args.bits / self.chunk_capacity)
        self.length_candi = np.arange(50, args.num_token_detection+50, 50)
        self.R = 1.0 / self.num_value
        
        self.seeding_scheme = "simple_3"
        
        self.vocab_size = len(vocab)
        
    def _compute_norm_p_val(self, cl_total, R):
        T_total = 0
        t_total = 0
        min_p_value = 10.0
        msg = []

        for _, value in cl_total.items():
            T = sum(value)
            if T:
                t = min(value)
                cur_msg = [i for i, v in enumerate(value) if v == t]
                msg.append(cur_msg)
                z = (t - R * T) / (math.sqrt(R * (1 - R) * T))
                cur_p_value = 1 - pow((1 - norm.cdf(z)), len(value))
                if cur_p_value < min_p_value:
                    min_p_value = cur_p_value
                T_total += T
                t_total += t
            else:
                cur_msg = [int(random.choice(np.arange(len(value))))]
                msg.append(cur_msg)

        p_value = norm.cdf((t_total - R * T_total) / (math.sqrt(R * (1 - R) * T_total))) if T_total > 0 else 0.5
        return p_value, msg
        
    def detect(self, text: list[str]):
        
        bp_res = {}
        cl_total = {i: [0 for _ in range(self.num_value)] for i in range(self.converted_msg_length)}
        detector_cache = {}  # seed_tuple -> colorlist
        
        for each_text in text:
            input_ids = self.tokenizer(each_text, return_tensors="pt").input_ids.to(self.device)
            
            for i in range(3, input_ids[0].shape[0]):
                l = i - 3
                r = i
                cur_seed = input_ids[:, l:r]
                seed_tuple = tuple(cur_seed[0].tolist())
                self.processor._seed_rng(cur_seed)
                bit_position = torch.randint(low=0, high=self.converted_msg_length, size=(1,), generator=self.processor.rng).item()
                
                hit = detector_cache.get(seed_tuple)
                if hit is None:
                    self._seed_rng(cur_seed)
                    vocab_perm = torch.randperm(self.vocab_size, device='cpu', generator=self.rng)
                    colorlist = torch.chunk(vocab_perm, self.num_value)
                    detector_cache[seed_tuple] = colorlist
                else:
                    colorlist = hit
                    
                new_token = input_ids[:, i].item()
                for guessed_info in range(self.num_value):
                    if new_token in colorlist[guessed_info]:
                        cl_total[bit_position][guessed_info] += 1

                cur_len = i + 1
                if cur_len in self.length_candi:
                    bp_res[cur_len] = {k: v[:] for k, v in cl_total.items()}
                    
        target_res = bp_res[max(bp_res.keys())] if bp_res else cl_total
        _, msg = self._compute_norm_p_val(target_res, self.R)
        symbols = [min(cands) if cands else 0 for cands in msg]
        
        
        return ''.join(self.binary_mapping[s] for s in symbols)
        
        