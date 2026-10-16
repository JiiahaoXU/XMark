from transformers import LogitsProcessor
import torch
import numpy as np
from sklearn.cluster import KMeans

class MajorMarkWatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        vocab: list,
        delta: float = 4.0,
        hash_key: int = 15485863,
        bits: int = 8,
        current_binary_code: str = None
    ):
        self.vocab_size = len(vocab)
        self.delta = delta
        self.hash_key = hash_key
        self.num_shards = bits
        self.rng = None
        self.topk_records = {'top1': [], 'top5': [], 'top10': []}
        if current_binary_code is None:
            raise ValueError("current_binary_code must be provided.")
        if len(current_binary_code) != bits:
            raise ValueError(f"Length of current_binary_code ({len(current_binary_code)}) must match bits ({bits}).")

        self.current_binary_code = current_binary_code

        count_1 = current_binary_code.count('1')
        count_0 = current_binary_code.count('0')
        self.marjor_bit = '1' if count_1 >= count_0 else '0'
        print('Major bit for current message:', self.marjor_bit)

    def _seed_rng(self, input_ids: torch.LongTensor):
        last_token = input_ids[0, -1].item() if input_ids.ndim == 2 else input_ids[-1].item()
        last_last_token = input_ids[0, -2].item() if input_ids.ndim == 2 else input_ids[-2].item()

        seed = (self.hash_key * (last_token * last_last_token) +
                int(self.marjor_bit) * 31) % (2**64)
        self.rng = torch.Generator(device=input_ids.device).manual_seed(seed)


    def _get_shard_ids(self, input_ids: torch.LongTensor) -> list[torch.Tensor]:
        self._seed_rng(input_ids)
        vocab_perm = torch.randperm(self.vocab_size, generator=self.rng, device=input_ids.device)
        shard_size = self.vocab_size // self.num_shards
        shards = []

        for i in range(self.num_shards):
            start = i * shard_size
            end = (i + 1) * shard_size if i < self.num_shards - 1 else self.vocab_size
            shards.append(vocab_perm[start:end])

        return shards

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        top1_ids = torch.topk(scores[0], k=1).indices.tolist()
        top5_ids = torch.topk(scores[0], k=5).indices.tolist()
        top10_ids = torch.topk(scores[0], k=10).indices.tolist()


        self.topk_records['top1'].append(top1_ids)
        self.topk_records['top5'].append(top5_ids)
        self.topk_records['top10'].append(top10_ids)


        shards = self._get_shard_ids(input_ids)

        active_shard_ids = [i for i, bit in enumerate(self.current_binary_code) if bit == self.marjor_bit]
        if not active_shard_ids:
            return scores

        boost_ids = torch.cat([shards[i] for i in active_shard_ids])
        boost_mask = torch.zeros_like(scores, dtype=torch.bool)
        boost_mask[..., boost_ids] = True

        scores = scores + self.delta * boost_mask.float()
        return scores




class MajorMarkWatermarkDetector:
    def __init__(self, vocab_size, device=None, tokenizer=None,
                 num_shards=None, hash_key=15485863, userid=None, args=None):

        self.vocab_size = vocab_size
        self.device = device
        self.tokenizer = tokenizer
        self.num_shards = num_shards
        self.hash_key = hash_key
        self.userid = userid
        self.args = args

    def _seed_rng(self, prev_token_id: int, prev_prev_token: int, flag_bit: int):

        seed = (self.hash_key * (prev_token_id * prev_prev_token) +
                flag_bit * 31) % (2**64)
        return torch.Generator(device=self.device).manual_seed(seed)

    def _get_shards(self, rng: torch.Generator):
        vocab_perm = torch.randperm(self.vocab_size, generator=rng, device=self.device)
        shard_size = self.vocab_size // self.num_shards
        shards = []
        for i in range(self.num_shards):
            start = i * shard_size
            end = (i + 1) * shard_size if i < self.num_shards - 1 else self.vocab_size
            shards.append(set(vocab_perm[start:end].tolist()))
        return shards

    def detect(self, generated_text_list: list[str]):
        total_counts_flag0 = [0] * self.num_shards
        total_counts_flag1 = [0] * self.num_shards

        for input_text in generated_text_list:
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)[0]
            for i in range(2, input_ids.shape[0]):
                prev_token = input_ids[i - 1].item()
                prev_prev_token = input_ids[i - 2].item()
                curr_token = input_ids[i].item()

                for flag in [0, 1]:
                    rng = self._seed_rng(prev_token, prev_prev_token, flag_bit=flag)
                    shards = self._get_shards(rng)
                    for j, shard in enumerate(shards):
                        if curr_token in shard:
                            if flag == 0:
                                total_counts_flag0[j] += 1
                            else:
                                total_counts_flag1[j] += 1
                            break

        std_0 = np.std(total_counts_flag0)
        std_1 = np.std(total_counts_flag1)
        chosen_counts = total_counts_flag1 if std_1 > std_0 else total_counts_flag0
        predicted_bit = '1' if std_1 > std_0 else '0'

        counts_np = np.array(chosen_counts).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, n_init="auto", random_state=0).fit(counts_np)
        labels = kmeans.labels_
        high_cluster = np.argmax(kmeans.cluster_centers_)

        decoded_binary_code = ''.join(
            predicted_bit if labels[i] == high_cluster else ('0' if predicted_bit == '1' else '1')
            for i in range(self.num_shards)
        )

        print("\n[Flag = 0] Shard token counts: \t \t [Flag = 1] Shard token counts:")
        for i, count in enumerate(total_counts_flag0):
            print(f"Shard {i}: {count} hits \t \t \t Shard {i}: {total_counts_flag1[i]} hits")


        print(f"\nPredicted encoding strategy: using major bit = {predicted_bit}")
        print(f"[STD] Flag=0: {std_0:.2f}, Flag=1: {std_1:.2f}")
        print("Decoded binary code:", decoded_binary_code)

        return decoded_binary_code
