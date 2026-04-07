"""
This code is modified from the original code in

https://github.com/kushr11/llm-identify

"""


from __future__ import annotations

import numpy as np
import torch
import copy
import math
import itertools

from tokenizers import Tokenizer
from transformers import LogitsProcessor
from .mpac.normalizers import normalization_strategy_lookup


class WatermarkBase:
    def __init__(
            self,
            vocab: list[int] = None,
            gamma: float = 0.5,
            decrease_delta: bool = True,
            delta: float = 2.0,
            wm_mode = "combination",
            seeding_scheme: str = "simple_1",  # mostly unused/always default
            hash_key: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
            select_green_tokens: bool = True,
            userid="10000100",
            args=None,
    ):

        # watermarking parameters
        self.wm_mode=wm_mode
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.decrease_delta = decrease_delta
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens
        self.idx_t = 0
        self.userid = userid
        self.hit = 0
        self.args=args
        self.args.gen_mode = "depth_d"
        self.args.depth = 3
        self.topk_records = {'top1': [], 'top5': [], 'top10': []}

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[
                       -1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            # print("prev token: ",prev_token)
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return
    def _seed_depth_rng(self) -> None:
        self.rng.manual_seed(self.hash_key * int(self.userid,2))
        return


    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        # seed the rng using the previous tokens/prefix
        # according to the seeding_scheme
        self._seed_rng(input_ids)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)

        greenlist_ids = vocab_permutation[:greenlist_size]  # new
        redlist_ids = vocab_permutation[greenlist_size:]

        self._seed_depth_rng()
        depth_permutation=torch.randperm(len(greenlist_ids), device=input_ids.device, generator=self.rng)
        depth_green_ids=greenlist_ids[depth_permutation]
        if len(redlist_ids)!=len(greenlist_ids):
            depth_red_ids=redlist_ids[:-1][depth_permutation]
            #append to tail
            depth_red_ids=torch.cat((depth_red_ids,torch.tensor([redlist_ids[-1]]).to(depth_red_ids.device)))
        else:
            depth_red_ids=redlist_ids[depth_permutation]

        if self.args.gen_mode=="depth_d":
            green_d_masks=[]
            red_d_masks=[]
            discrete_depth=self.args.depth
            g_discrete_length=greenlist_size//discrete_depth
            r_discrete_length=greenlist_size//discrete_depth
            for i in range(discrete_depth):
                if i == discrete_depth-1:
                    green_d_masks.append(depth_green_ids[i*g_discrete_length:])
                    red_d_masks.append(depth_red_ids[i*r_discrete_length:])
                else:
                    green_d_masks.append(depth_green_ids[i*g_discrete_length:(i+1)*g_discrete_length])
                    red_d_masks.append(depth_red_ids[i*r_discrete_length:(i+1)*r_discrete_length])
            return greenlist_ids,redlist_ids,green_d_masks,red_d_masks
        return greenlist_ids, redlist_ids,[],[]
        # return greenlist_ids
        

class DepthWProcessor(WatermarkBase, LogitsProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _bias_depth_d_logits(self,scores: torch.FloatTensor,greenlist_token_ids,d_masks,delta):
        for i in range(len(d_masks)):
            delta=delta*0.5**i
            for j in range(len(greenlist_token_ids)):
                scores[j][d_masks[i]]=scores[j][d_masks[i]]+delta
        return scores
    
    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float, decrease_delta: bool) -> torch.Tensor:
        if decrease_delta:
            greenlist_bias=4.84*(math.e)**(-1*0.001*self.idx_t)
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        # print(greenlist_bias,self.idx_t)
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        
        top1_ids = torch.topk(scores[0], k=1).indices.tolist()
        top5_ids = torch.topk(scores[0], k=5).indices.tolist()
        top10_ids = torch.topk(scores[0], k=10).indices.tolist()


        self.topk_records['top1'].append(top1_ids)
        self.topk_records['top5'].append(top5_ids)
        self.topk_records['top10'].append(top10_ids)
        
        n = len(self.userid)

        # preferance = self.userid[(self.idx_t - n * (self.idx_t // n)) % n]  # 1->green ; 0-> red
        if self.wm_mode =='previous1':
            preferance = self.userid[input_ids[-1][-1] % n]  # 1->green ; 0-> red
        else:
            preferance = self.userid[(input_ids[-1][-1]*input_ids[-1][-2]) % n]  # 1->green ; 0-> red
        self.idx_t += 1
        # this is lazy to allow us to colocate on the watermarked model's device
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]
        if self.args.gen_mode=='depth_d':
            batched_d_masks = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            # d_masks only availiable in "depth_d"
            if preferance == '1':
                greenlist_ids, _ ,d_masks,_= self._get_greenlist_ids(input_ids[b_idx])
            else:
                _, greenlist_ids,_,d_masks = self._get_greenlist_ids(input_ids[b_idx])
            
            batched_greenlist_ids[b_idx] = greenlist_ids
            if self.args.gen_mode=='depth_d':
                batched_d_masks[b_idx] = d_masks

        if self.args.gen_mode !='depth_d':
            green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)
            scores_withnomask=copy.deepcopy(scores)
            scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta,decrease_delta=self.decrease_delta)
        else:
            scores_withnomask=copy.deepcopy(scores)
            scores=self._bias_depth_d_logits(scores=scores,greenlist_token_ids=batched_greenlist_ids,d_masks=d_masks,delta=self.delta)
            
        return scores


class DepthWDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],
        ignore_repeated_bigrams: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.device = device
        self.tokenizer = tokenizer
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1" and self.wm_mode == 'previous1':
            self.min_prefix_len = 1
        elif self.wm_mode == 'combination':
            self.min_prefix_len = 2
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        self.normalizers = [normalization_strategy_lookup(n) for n in normalizers]
        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1"

    def _score_sequence(self, input_ids: torch.Tensor) -> float:
        green_token_count = 0
        depth_hit = torch.zeros(self.args.depth, device=self.device)

        for idx in range(self.min_prefix_len, len(input_ids)):
            curr_token = input_ids[idx]
            n = len(self.userid)

            if self.wm_mode == 'previous1':
                preferance = self.userid[input_ids[idx - 1] % n]
            else:
                preferance = self.userid[(input_ids[idx - 1] * input_ids[idx - 2]) % n]

            if preferance == '1':
                greenlist_ids, _, d_masks, _ = self._get_greenlist_ids(input_ids[:idx])
            else:
                _, greenlist_ids, _, d_masks = self._get_greenlist_ids(input_ids[:idx])

            if curr_token in greenlist_ids:
                green_token_count += 1
                if self.args.gen_mode == "depth_d":
                    for j in range(len(d_masks)):
                        if curr_token in d_masks[j]:
                            depth_hit[j] += 1

        if green_token_count == 0:
            return 0.0

        depth_pd = depth_hit / green_token_count
        sim_score = green_token_count / (len(input_ids) - self.min_prefix_len)

        standard_depth_distribution = torch.tensor(
            [0.6834, 0.1800, 0.1366], device=self.device
        )
        standard_gr = 0.7959

        loss_func = torch.nn.CrossEntropyLoss()
        total_loss = -loss_func(depth_pd * sim_score, standard_depth_distribution * standard_gr)

        return float(total_loss)

    def detect(
        self,
        text=None,
        tokenized_text=None,
        **kwargs,
    ) -> str:
        all_depth_scores = []
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either raw or tokenized input"

        all_possible_codes = [''.join(bits) for bits in itertools.product('01', repeat=self.args.bits)]

        for j, code in enumerate(all_possible_codes):
            if j % 25 == 0:
                print(f"Evaluating Code {j + 1}/{len(all_possible_codes)}: {code}")
            self.userid = code
            sample_scores = []

            if text is not None:
                texts = text if isinstance(text, list) else [text]
                for t in texts:
                    for normalizer in self.normalizers:
                        t = normalizer(t)
                    tokenized_input = self.tokenizer(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
                    if tokenized_input[0] == self.tokenizer.bos_token_id:
                        tokenized_input = tokenized_input[1:]
                    score = self._score_sequence(tokenized_input)
                    sample_scores.append(score)
            else:
                tokenized_inputs = tokenized_text if isinstance(tokenized_text[0], list) else [tokenized_text]
                for input_ids in tokenized_inputs:
                    if self.tokenizer and input_ids[0] == self.tokenizer.bos_token_id:
                        input_ids = input_ids[1:]
                    input_tensor = torch.tensor(input_ids).to(self.device)
                    score = self._score_sequence(input_tensor)
                    sample_scores.append(score)

            avg_score = float(np.mean(sample_scores))
            all_depth_scores.append(avg_score)

        best_code_idx = np.argmax(all_depth_scores)
        return all_possible_codes[best_code_idx]
