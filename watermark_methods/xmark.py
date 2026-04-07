from transformers import LogitsProcessor
import torch
import pickle

class XMARKProcessor(LogitsProcessor):
    def __init__(self, vocab, bits=32, delta=4.0, current_binary_code=None, hash_key=15485863, args=None):
        self.bits = bits
        self.num_blocks = bits // 2
        self.delta = delta
        self.vocab_size = len(vocab)
        self.hash_key = hash_key
        self.hash_key_2 = 12345
        self.hash_key_3 = 54321
        self.args = args
        self.current_binary_code = current_binary_code
        self.pos_bits = self.get_pos_bits_dict(self.current_binary_code)
        self.topk_records = {'top1': [], 'top5': [], 'top10': []}
          
    def get_pos_bits_dict(self, binary_code: str):
        pos_bits = {}
        for i in range(0, len(binary_code), 2):
            pos = i // 2  
            bits = binary_code[i:i+2]
            pos_bits[pos] = int(bits, 2)
        return pos_bits    
        
        
    def _get_block_index(self, input_ids):
        last_token = input_ids[0, -1].item() if input_ids.ndim == 2 else input_ids[-1].item()
        last_last_token = input_ids[0, -2].item() if input_ids.ndim == 2 else input_ids[-2].item()
        return (last_token + last_last_token) % self.num_blocks
        
    def _seed_rng(self, input_ids, hash_key: int):
        last_token = input_ids[0, -1].item() if input_ids.ndim == 2 else input_ids[-1].item()
        last_last_token = input_ids[0, -2].item() if input_ids.ndim == 2 else input_ids[-2].item()

        seed = (hash_key * last_token * last_last_token) % (2**64)

        return torch.Generator().manual_seed(seed), seed

    
    def _get_all_boost_ids(self, n_parts, block_num, rng):
        
        vocab_perm = torch.randperm(self.vocab_size, generator=rng)

        V = self.vocab_size
        chunk = V // n_parts

        start = block_num * chunk
        end = V if block_num == n_parts - 1 else (block_num + 1) * chunk
        boost_ids = torch.cat([vocab_perm[:start], vocab_perm[end:]], dim=0)
        
        return boost_ids
    

    def __call__(self, input_ids, scores):
        
        top1_ids = torch.topk(scores[0], k=1).indices.tolist()
        top5_ids = torch.topk(scores[0], k=5).indices.tolist()
        top10_ids = torch.topk(scores[0], k=10).indices.tolist()


        self.topk_records['top1'].append(top1_ids)
        self.topk_records['top5'].append(top5_ids)
        self.topk_records['top10'].append(top10_ids)

        boosted_scores = scores.clone()
        
        block_index = self._get_block_index(input_ids)
        block_num = self.pos_bits[block_index] # int number

        n_parts = 4
        
        rng_1, seed_1 = self._seed_rng(input_ids, self.hash_key)
        boost_ids_1 = self._get_all_boost_ids(n_parts, block_num, rng_1)
        
        rng_2, seed_2 = self._seed_rng(input_ids, self.hash_key_2)
        boost_ids_2 = self._get_all_boost_ids(n_parts, block_num, rng_2)
        
        dark_ids = torch.tensor(list(set(boost_ids_1.tolist()).intersection(set(boost_ids_2.tolist()))), device=scores.device, dtype=torch.long)
        boost_mask = torch.zeros_like(scores, dtype=torch.bool)
        boost_mask[..., dark_ids] = True
        
        boosted_scores += self.delta * boost_mask.float()
        
        
        return boosted_scores
    

class XMARKDetector:
    def __init__(self, vocab_size, device=None, tokenizer=None,
                 bits=32, hash_key=15485863):
        self.vocab_size = vocab_size
        self.device = device
        self.tokenizer = tokenizer
        self.bits = bits
        self.hash_key = hash_key
        self.hash_key_2 = 12345
        self.hash_key_3 = 54321
        self.num_blocks = bits // 2
    def _seed_rng(self, prev_token_id: int, prev_prev_token: int, hash_key: int):
        seed = (hash_key * prev_token_id * prev_prev_token) % (2**64)
        
        return torch.Generator().manual_seed(seed), seed

    def _get_block_index(self, i: int, j:int):
        return (i + j) % self.num_blocks

    def detect(self, generated_text_list: list[str]):
        counts = torch.zeros((self.num_blocks, 4), dtype=torch.long)

        print("=== Detection Pass: Collecting Token Statistics ===")
        for text_id, text in enumerate(generated_text_list):
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)[0]
            for i in range(2, input_ids.shape[0]):
                prev_prev = int(input_ids[i-2].item())
                prev      = int(input_ids[i-1].item())
                curr      = int(input_ids[i].item())

                bidx = self._get_block_index(prev, prev_prev)  

                rng_1, _ = self._seed_rng(prev, prev_prev, self.hash_key)
                vocab_perm_1 = torch.randperm(self.vocab_size, generator=rng_1)
                parts_1 = torch.chunk(vocab_perm_1, chunks=4, dim=0)
                
                rng_2, _ = self._seed_rng(prev, prev_prev, self.hash_key_2)
                vocab_perm_2 = torch.randperm(self.vocab_size, generator=rng_2)
                parts_2 = torch.chunk(vocab_perm_2, chunks=4, dim=0)

                for q, part in enumerate(parts_1):
                    if curr in part:
                        counts[bidx, q] += 1
                        break
                
                for q, part in enumerate(parts_2):
                    if curr in part:
                        # cTMM
                        if curr not in parts_1[q]:  
                            counts[bidx, q] += 1
                            break
                        

        for block_idx, block_count in enumerate(counts):
            print(f"Block {block_idx}: {block_count.tolist()}, total={block_count.sum().item()}")
            
        total_tokens = counts.sum().item()
        print(f"total tokens analyzed: {total_tokens}")
        
        print("\n=== Decoding Binary Code ===")
        block_indices = torch.argmin(counts, dim=1).tolist()  # 0..3

        bits_per_block = [format(idx, "02b") for idx in block_indices]
        bitstring = "".join(bits_per_block)

        print(f"\n[Final Decoded Binary Code] {bitstring}")
        return bitstring
      