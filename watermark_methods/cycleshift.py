"""
This code is modified from the code in

https://zenodo.org/records/14729410

"""

from transformers import LogitsProcessor
import torch
import numpy as np

from typing import List
import numpy as np
from numpy._core.multiarray import array as array
from scipy import special
import torch
import logging


class CycleShiftProcessor(LogitsProcessor):
    def __init__(self, vocab, delta, payload, bits, gamma, hash_key=15485863):
        
        self.payload = payload
        self.bits = bits
        self.gamma = gamma
        self.delta = delta
        self.vocab_size = len(vocab)
        self.salt_key = hash_key
        
        self.seed = 1
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

        self.topk_records = {'top1': [], 'top5': [], 'top10': []}
        
        self.token_count = 0
        
        
    def get_seed_rng(self, input_ids):
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        seed = self.seed
        # for i in input_ids:
        seed = (seed * self.salt_key + input_ids.item()) % (2 ** 64 - 1)
        
        return seed
    

    def __call__(self, input_ids, scores):
        
        top1_ids = torch.topk(scores[0], k=1).indices.tolist()
        top5_ids = torch.topk(scores[0], k=5).indices.tolist()
        top10_ids = torch.topk(scores[0], k=10).indices.tolist()


        self.topk_records['top1'].append(top1_ids)
        self.topk_records['top5'].append(top5_ids)
        self.topk_records['top10'].append(top10_ids)

        scores = scores.clone()
            
        seed = self.get_seed_rng(input_ids[0, -1])
        
        # print('previes_token:', input_ids[0, -1], 'seed', seed)
        self.rng.manual_seed(seed)
        # r_length = self.vocab_size
        r_length = max(2**self.bits, self.vocab_size)
        vocab_permutation = torch.randperm(r_length, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * r_length)] # gamma * payload_max, index of greenlist token
        bias = torch.zeros(r_length).to(scores.device) # payload_max
        bias[greenlist] = self.delta
        bias = bias.roll(-self.payload) # roll the bias to the left by payload, i.e. payload_max
        scores += bias[:self.vocab_size] # add bias to the scores, only for vocab size

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



class CycleShiftDetector(WmDetector):
    def __init__(
            self,
            gamma,
            bits,
            device,
            **kwargs):
        super().__init__(**kwargs)


        self.bits = bits
        self.gamma = gamma # gamma is the ratio of payloads in the greenlist, default 0.5
        self.device = device
        self.ngram = 1 # ngram is the number of previous tokens to consider, default 1

        
    def get_seed_rng(self, input_ids):
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        seed = self.seed
        # for i in input_ids:
        seed = (seed * self.salt_key + input_ids) % (2 ** 64 - 1)
        
        return seed
    
    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens[-1].item())
        self.rng.manual_seed(seed)
        r_length = max(2**self.bits, self.vocab_size)
        scores = torch.zeros(r_length) # scores tensor of all possible payloads
        vocab_permutation = torch.randperm(r_length, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * r_length)] # gamma * n toks in the greenlist
        # print(scores)
        scores[greenlist] = 1
        # print(scores)
        scores = scores.roll(-token_id.item())
        return scores
    
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a binomial distribution """
        pvalue = special.betainc(score, 1 + ntoks - score, self.gamma)
        return max(pvalue, eps)
    
    
    def get_aggregate_scores(
        self, 
        texts: List[str], 
        payload_max: int = 0
    ):
        """
        Get score for each payload in list of texts (aggregated across all tokens generated using sum)
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            payload_max: maximum number of messages 
        Output:
            score_lists: list of [np array of score for each payload] for each text
            ntoks_arr: list of [# of generated tokens] for each text
        """
        bsz = len(texts)
        # print(bsz)
        tokens_id = [self.tokenizer(x, return_tensors="pt").input_ids.to(self.device)[0] for x in texts]

        # print(tokens_id)
        # for item in tokens_id:
        #     print(len(item))

        score_lists = []
        ntoks_arr = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            rt_aggr = torch.zeros(payload_max) # init aggregate scores as all 0
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos]) 
                rt = rt[:payload_max] # rt: contribution of token t on each payloads
                rt_aggr += rt # add contribution of token t to rt_aggr
            score_lists.append(rt_aggr.numpy())
            ntoks_arr.append(total_len - start_pos) 
        return score_lists, np.asarray(ntoks_arr) 
    
    
    def get_pvalues_from_aggr_scores(
        self,
        scores: np.array,
        ntoks_arr: np.array,
        eps: float=1e-200
    ) -> np.array:
        """
        Get p-value for each text.
        Args:
            scores: list of [list of scores (aggr by sum) for each payload] for each text
            ntoks_arr: list of num of tokens generated for each text
        Output:
            pvalues: np array of p-values for each text and payload
        """
        pvalues = []
        scores = np.asarray(scores) # bsz x payload_max
        ntoks_arr = np.asarray(ntoks_arr) # bsz
        for ss, ntoks in zip(scores, ntoks_arr):
            pvalues_by_payload = [self.get_pvalue(score, ntoks, eps=eps) for score in ss]
            pvalues.append(pvalues_by_payload)
        return np.asarray(pvalues) # bsz x payload_max
    

    def detect(self, generated_text_list: list[str]):
        
        logging.info('Decoding payloads from generated texts...')
        
        scores, num_tokens = self.get_aggregate_scores(generated_text_list, payload_max=2**self.bits)
        pvalues = self.get_pvalues_from_aggr_scores(scores, num_tokens)
        
        payloads = np.argmin(pvalues, axis=1).tolist() # decoded payloads, shape: (bsz)
        
        payload = payloads[0]
        
        code = f"{payload:0{self.bits}b}"
        
        return code
                    
                
    

    
