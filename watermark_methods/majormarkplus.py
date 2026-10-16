from transformers import LogitsProcessor
import torch
import numpy as np


class MajorMarkPlusWatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab, bits=32, block_size=8, delta=4.0, current_binary_code=None, hash_key=15485863, args=None):
        if bits < 2 or block_size < 2 or bits % block_size != 0:
            raise ValueError("MajorMark+ requires block_size >= 2 and bits divisible by block_size.")
        self.bits = bits
        self.block_size = block_size
        self.num_blocks = bits // block_size
        self.delta = delta
        self.vocab_size = len(vocab)
        self.hash_key = hash_key
        self.args = args

        if current_binary_code is None or len(current_binary_code) != bits or set(current_binary_code) - {'0', '1'}:
            raise ValueError("Invalid binary code")

        self.blocks = []
        self.marjor_bits = []

        self.topk_records = {'top1': [], 'top5': [], 'top10': []}

        for i in range(self.num_blocks):
            block = current_binary_code[i * block_size : (i + 1) * block_size]
            if len(set(block)) < 2:
                raise ValueError("MajorMark+ requires both 0 and 1 in every message block.")
            count_1 = block.count('1')
            count_0 = block.count('0')
            marjor_bit = '1' if count_1 >= count_0 else '0'
            self.blocks.append(block)
            self.marjor_bits.append(marjor_bit)

    def _get_block_index(self, input_ids):
        last_token = input_ids[0, -1].item() if input_ids.ndim == 2 else input_ids[-1].item()
        last_last_token = input_ids[0, -2].item() if input_ids.ndim == 2 else input_ids[-2].item()
        return (last_token + last_last_token) % self.num_blocks

    def _seed_rng(self, input_ids, flag_bit: int, block_index: int):
        last_token = input_ids[0, -1].item() if input_ids.ndim == 2 else input_ids[-1].item()
        last_last_token = input_ids[0, -2].item() if input_ids.ndim == 2 else input_ids[-2].item()

        block_bits = self.blocks[block_index]
        count_flag = block_bits.count(str(flag_bit))

        seed = (self.hash_key * last_token * last_last_token +
                flag_bit * 31 +
                count_flag * 97) % (2**64)

        return torch.Generator().manual_seed(seed), seed

    def _get_shards(self, rng: torch.Generator, input_ids, block_bits, marjor_bit):
        vocab_perm = torch.randperm(self.vocab_size, generator=rng)
        shard_size = self.vocab_size // self.block_size
        shards = []
        for i in range(self.block_size):
            start = i * shard_size
            end = (i + 1) * shard_size if i < self.block_size - 1 else self.vocab_size
            shards.append(vocab_perm[start:end])
        active_shard_ids = [i for i, b in enumerate(block_bits) if b == marjor_bit]

        return torch.cat([shards[i] for i in active_shard_ids])


    def __call__(self, input_ids, scores):

        top1_ids = torch.topk(scores[0], k=1).indices.tolist()
        top5_ids = torch.topk(scores[0], k=5).indices.tolist()
        top10_ids = torch.topk(scores[0], k=10).indices.tolist()


        self.topk_records['top1'].append(top1_ids)
        self.topk_records['top5'].append(top5_ids)
        self.topk_records['top10'].append(top10_ids)

        boosted_scores = scores.clone()
        block_index = self._get_block_index(input_ids)
        block_bits = self.blocks[block_index]
        marjor_bit = self.marjor_bits[block_index]

        for flag_bit in [0, 1]:
            if int(marjor_bit) != flag_bit:
                continue
            rng, seed = self._seed_rng(input_ids, flag_bit, block_index)
            boost_ids = self._get_shards(rng, input_ids, block_bits, marjor_bit)
            boost_mask = torch.zeros_like(scores, dtype=torch.bool)
            boost_mask[..., boost_ids] = True
            boosted_scores += self.delta * boost_mask.float()

        return boosted_scores


class MajorMarkPlusWatermarkDetector:
    def __init__(self, vocab_size, device=None, tokenizer=None,
                 bits=32, block_size=8, hash_key=15485863):
        if bits < 2 or block_size < 2 or bits % block_size != 0:
            raise ValueError("MajorMark+ requires block_size >= 2 and bits divisible by block_size.")
        self.vocab_size = vocab_size
        self.device = device
        self.tokenizer = tokenizer
        self.bits = bits
        self.block_size = block_size
        self.num_blocks = bits // block_size
        self.hash_key = hash_key


    def _seed_rng(self, prev_token_id: int, flag_bit: int, prev_prev_token: int, count_flag: int):
        seed = (self.hash_key * prev_token_id * prev_prev_token +
                flag_bit * 31 + count_flag * 97) % (2**64)

        return torch.Generator().manual_seed(seed), seed

    def _get_block_index(self, i: int, j:int):
        return (i + j) % self.num_blocks

    def _get_shards(self, rng: torch.Generator):
        vocab_perm = torch.randperm(self.vocab_size, generator=rng)
        shard_size = self.vocab_size // self.block_size
        shards = []
        for i in range(self.block_size):
            start = i * shard_size
            end = (i + 1) * shard_size if i < self.block_size - 1 else self.vocab_size
            shards.append(set(vocab_perm[start:end].tolist()))
        return shards

    def _valid_count_flags(self, flag: int) -> list[int]:

        half = self.block_size // 2
        if flag == 1:
            return list(range((self.block_size + 1) // 2, self.block_size))
        else:
            return list(range(half + 1, self.block_size))

    def detect(self, generated_text_list: list[str]):
        block_stats = {}

        block_0_count = 0
        block_1_count = 0

        print("=== Detection Pass: Collecting Token Statistics ===")
        for text_id, text in enumerate(generated_text_list):
        #     print(text_id, text)
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)[0]
            for i in range(2, input_ids.shape[0]):
                prev_token = input_ids[i - 1].item()
                prev_prev_token = input_ids[i - 2].item()
                curr_token = input_ids[i].item()


                for flag in [0, 1]:
                    for count_flag in self._valid_count_flags(flag):

                        rng, seed = self._seed_rng(prev_token, flag, prev_prev_token, count_flag)
                        shards = self._get_shards(rng)

                        block_idx = self._get_block_index(prev_token, prev_prev_token)

                        if block_idx == 0:
                            block_0_count +=1
                        elif block_idx == 1:
                            block_1_count += 1

                        key = (block_idx, flag, count_flag)
                        if key not in block_stats:
                            block_stats[key] = [0] * self.block_size

                        for shard_id, shard in enumerate(shards):
                            if curr_token in shard:
                                block_stats[key][shard_id] += 1
                                break

        print("\n=== Decoding Binary Code ===")
        decoded_bits = ''
        for block_idx in range(self.num_blocks):
            best_std = -1
            predicted_major_bit = None
            best_count_flag = None
            best_hits = None

            print(f"\n[Block {block_idx}]")
            for (b_idx, flag, count_flag), shard_hits in block_stats.items():
                if b_idx != block_idx:
                    continue
                std = np.std(shard_hits)
                print(f"  Major bit={flag}, Major bit occ={count_flag}, Hits={shard_hits}, Std={std:.4f}")
                if std > best_std:
                    best_std = std
                    predicted_major_bit = flag
                    best_count_flag = count_flag
                    best_hits = shard_hits

            if best_hits is None:
                decoded_bits += '0' * self.block_size
                print("  No token evidence; decoding this block as zeros.")
                continue

            shard_hits_np = np.array(best_hits)
            top_k_indices = np.argsort(shard_hits_np)[-best_count_flag:]

            block_bits = ''.join(
                str(predicted_major_bit) if i in top_k_indices else str(1 - predicted_major_bit)
                for i in range(self.block_size)
            )
            decoded_bits += block_bits

            print(f"  → Chosen: Major bit={predicted_major_bit}, Major bit occ={best_count_flag}, TopK={top_k_indices.tolist()}, Decoded = {block_bits}")

        print(f"\n[Final Decoded Binary Code] {decoded_bits}")
        return decoded_bits
