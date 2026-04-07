"""
This code is modified from the original code in

https://zenodo.org/records/14729410

"""



from transformers import LogitsProcessor
import torch
from .generalizedReedSolomon.generalizedreedsolo import Generalized_Reed_Solomon
from numpy._core.multiarray import array as array
import torch
import pickle


class RSBHProcessor(LogitsProcessor):
    def __init__(self, vocab, delta, payload, gamma, segments_num, gf_segments_num, segment_bit, hash_key=15485863):
        
        self.payload = payload
        self.gamma = gamma
        self.is_slash_n = False
        # self.eos_id = model.config.eos_token_id
        
        self.segments_num = segments_num
        # total number of segments after RS, or n in RS code
        self.gf_segments_num = gf_segments_num
        # the # of bits within one segment, m in RS code (GF(q^m))
        self.segment_bit = segment_bit
        
        # bidwidth = segment_bit * segments_num
        self.bitwidth = self.segments_num * self.segment_bit
        
        with open('./watermark_methods/rsbh/token_freq_llama.pkl', 'rb') as f:
            self.mapping = pickle.load(f)

        # 1. divide original message into segments
        mask = 2 ** self.segment_bit - 1
        self.segments = [
            (self.payload >> (self.segment_bit * i)) & mask for i in range(self.segments_num)
        ]
        
        print(self.segments)

        # 2. encode segments with RS
        self.rs = Generalized_Reed_Solomon(
            field_size=2,
            message_length=self.gf_segments_num,
            payload_length=self.segments_num,
            symbol_size=self.segment_bit,
            p_factor=1
        )

        self.gf_segments = [int(i) for i in self.rs.encode(self.segments)]      
        
        self.delta = delta
        self.vocab_size = len(vocab)
        self.salt_key = hash_key
        
        
        
        self.seed = 1
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

        self.topk_records = {'top1': [], 'top5': [], 'top10': []}
        
        self.token_count = 0
        
        
    def get_seed_rng(self, input_ids, E_p):
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        seed = self.seed
        # for i in input_ids:
        seed = (seed * self.salt_key * E_p + input_ids.item()) % (2 ** 64 - 1)
        
        return seed
    
    def _get_segment_index(self):
        return self.token_count % len(self.gf_segments)
    

    def __call__(self, input_ids, scores):
        
        top1_ids = torch.topk(scores[0], k=1).indices.tolist()
        top5_ids = torch.topk(scores[0], k=5).indices.tolist()
        top10_ids = torch.topk(scores[0], k=10).indices.tolist()


        self.topk_records['top1'].append(top1_ids)
        self.topk_records['top5'].append(top5_ids)
        self.topk_records['top10'].append(top10_ids)

        scores = scores.clone()
        current_segment_idx_p = self.mapping[input_ids[0, -1].item() % 32000] % len(self.gf_segments)            
        seed = self.get_seed_rng(input_ids[0, -1], self.gf_segments[current_segment_idx_p])
        
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * payload_max, index of greenlist token
        bias = torch.zeros_like(scores).to(scores.device) # payload_max
        bias[..., greenlist] += self.delta

        scores += bias

        return scores
    
    
class WmDetector():
    def __init__(self, 
            tokenizer, 
            ngram,
            seeding,
        ):
        # model config
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer)
        # watermark config
        self.ngram = ngram
        self.salt_key = 15485863
        self.seed = 1
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)



class RSBHDetector(WmDetector):
    def __init__(
            self,
            gamma,
            segments_num,
            gf_segments_num,
            segment_bit,
            device,
            **kwargs):
        super().__init__(**kwargs)
        # total number of segments, or k in RS code
        self.segments_num = segments_num
        # total number of segments after RS, or n in RS code
        self.gf_segments_num = gf_segments_num
        # the # of bits within one segment, or m in RS code (GF(q^m))
        self.segment_bit = segment_bit
        # bitwidth
        self.bitwidth = self.segments_num * self.segment_bit
        
        self.gamma = gamma # gamma is the ratio of payloads in the greenlist, default 0.5
        self.device = device

        self.rs = Generalized_Reed_Solomon(
            field_size=2,
            message_length=self.gf_segments_num,   # n
            payload_length=self.segments_num,      # k
            symbol_size=self.segment_bit,          # m
            p_factor=1
        )
        
        with open('./watermark_methods/rsbh/token_freq_llama.pkl', 'rb') as f:
            self.mapping = pickle.load(f)
        
    def get_seed_rng(self, input_ids, E_p):
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        seed = self.seed
        # for i in input_ids:
        seed = (seed * self.salt_key * E_p + input_ids) % (2 ** 64 - 1)
        
        return seed
    
    def _get_segment_index(self, i):
        return i % self.gf_segments_num

    def detect(self, generated_text_list: list[str]):
        count = {}
        
        for i in range(self.gf_segments_num):
            count[i] = [0] * 2**self.segment_bit

        print("=== Detection Pass: Collecting Token Statistics ===")
        for text_id, text in enumerate(generated_text_list):

            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)[0]

            for i in range(1, input_ids.shape[0]):
                
                
                prev_token = input_ids[i - 1].item()
                curr_token = input_ids[i].item()
                
                segment_idx_p = (self.mapping[prev_token % 32000]) % self.gf_segments_num
                
                for j in range(2**self.segment_bit):
                    # print(f'Processing Token {i}, segment {j}')
                    current_seed = self.get_seed_rng(prev_token, j)
                    self.rng.manual_seed(current_seed)
                    # r_length = self.vocab_size
                    vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
                    greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * payload_max, index of greenlist token
                    
                    if curr_token in greenlist:
                        
                        count[segment_idx_p][j] += 1
                                
        max_indices = [values.index(max(values)) for key, values in sorted(count.items())]

        try:
            payload_in_segs = self.rs.decode(max_indices)
        except (ZeroDivisionError, IndexError) as e:
            # directly return non-corrected payload when error
            print(f"{e} in RS decode!")
            payload_in_segs = max_indices[:self.segments_num]
        
        payload = 0
        for i in range(self.segments_num):
            payload += int(payload_in_segs[i]) << (i * self.segment_bit)
            
        code = f"{payload:0{self.segments_num * self.segment_bit}b}"
        return code
                    
                