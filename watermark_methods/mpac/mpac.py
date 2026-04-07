"""
This code modified based on the official codebase of MPAC, which can be found at:
https://github.com/bangawayoo/mb-lm-watermarking

"""


from __future__ import annotations
import collections
import math
from math import sqrt, ceil, floor, log2, log
from itertools import chain, tee
from functools import lru_cache
import time
import random

import scipy.stats
from scipy.stats import chisquare, entropy, binom
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from watermark_methods.mpac.normalizers import normalization_strategy_lookup
from watermark_methods.mpac.alternative_prf_schemes import prf_lookup, seeding_scheme_lookup


class WatermarkBase:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",  # simple default, find more schemes in alternative_prf_schemes.py
        select_green_tokens: bool = True,  # should always be the default if not running in legacy mode
        base: int = 2,  # base (radix) of each message
        message_length: int = 4,
        code_length: int = 4,
        use_position_prf: bool = True,
        use_fixed_position: bool = False,
        device: str = "cuda",
        **kwargs
    ):
        # patch now that None could now maybe be passed as seeding_scheme
        if seeding_scheme is None:
            seeding_scheme = "simple_1"
        self.device = device
        # Vocabulary setup
        self.vocab = vocab
        self.vocab_size = len(vocab)
        # print(f"Watermarking vocabulary size: {self.vocab_size}")

        # Watermark behavior:
        self.gamma = gamma
        self.delta = delta
        self.rng = None
        self._initialize_seeding_scheme(seeding_scheme)
        # Legacy behavior:
        self.select_green_tokens = select_green_tokens

        ### Parameters for multi-bit watermarking ###
        self.original_msg_length = message_length
        self.message_length = max(message_length, code_length)
        decimal = int("1" * message_length, 2)
        self.converted_msg_length = len(self._numberToBase(decimal, base))

        # if message bit width is leq to 2, no need to increase base
        if message_length <= 2:
            base = 2
        self.message = None
        self.bit_position = None
        self.base = base
        # self.chunk = int(ceil(log2(base)))
        assert floor(1 / self.gamma) >= base, f"Only {floor(1 / self.gamma)} chunks available " \
                                              f"with current gamma={self.gamma}," \
                                              f"But base is {self.base}"
        self.converted_message = None
        self.message_char = None
        self.use_position_prf = use_position_prf
        self.use_fixed_position = use_fixed_position
        self.bit_position_list = []
        self.position_increment = 0
        self.green_cnt_by_position = {i: [0 for _ in range(self.base)] for i
                                      in range(1, self.converted_msg_length + 1)}

    def _initialize_seeding_scheme(self, seeding_scheme: str) -> None:
        """Initialize all internal settings of the seeding strategy from a colloquial, "public" name for the scheme."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = seeding_scheme_lookup(
            seeding_scheme
        )

    def _seed_rng(self, input_ids: torch.LongTensor) -> None:
        """Seed RNG from local context. Not batched, because the generators we use (like cuda.random) are not batched."""
        # Need to have enough context for seed generation
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(
                f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG."
            )

        prf_key = prf_lookup[self.prf_type](
            input_ids[-self.context_width :], salt_key=self.hash_key
        )
        if self.use_position_prf:
            position_prf_key = prf_lookup["anchored_minhash_prf"](
                input_ids[-2:], salt_key=self.hash_key
            )
        else:
            position_prf_key = prf_key
        self.prf_key = prf_key

        # seeding for bit position
        random.seed(position_prf_key % (2**64 - 1))
        if self.use_fixed_position:
            self.bit_position = list(
                                range(1, self.converted_msg_length + 1)
                                )[self.position_increment % self.converted_msg_length]
        else:
            self.bit_position = random.randint(1, self.converted_msg_length)
        self.message_char = self.get_current_bit(self.bit_position)
        # enable for long, interesting streams of pseudorandom numbers: print(prf_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))  # safeguard against overflow from long

    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=input_ids.device, generator=self.rng
        )
        candidate_greenlist = torch.chunk(vocab_permutation, floor(1 / self.gamma))
        return candidate_greenlist[self.message_char]

    def _get_colorlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Seed rng based on local context width and use this information to generate ids on the green list."""
        self._seed_rng(input_ids)
        vocab_permutation = torch.randperm(
            self.vocab_size, device=input_ids.device, generator=self.rng
        )
        colorlist = torch.chunk(vocab_permutation, floor(1 / self.gamma))
        return colorlist

    # @lru_cache(maxsize=2**32)
    def _get_ngram_score_cached(self, prefix: tuple[int], target: int, cand_msg=None):
        """Expensive re-seeding and sampling is cached."""
        ######################
        # self.converted_message = str(cand_msg) * self.converted_msg_length
        # Handle with care, should ideally reset on __getattribute__ access to self.prf_type, self.context_width, self.self_salt, self.hash_key
        # greenlist_ids = self._get_greenlist_ids(torch.as_tensor(prefix, device=self.device))
        # return True if target in greenlist_ids else False, self.get_current_position()
        ######################
        colorlist_ids = self._get_colorlist_ids(torch.as_tensor(prefix, device=self.device))
        colorlist_flag = []
        for cl in colorlist_ids[:self.base]:
            if target in cl:
                colorlist_flag.append(True)
            else:
                colorlist_flag.append(False)

        return colorlist_flag, self.get_current_position()

    def get_current_bit(self, bit_position):
        # embedding stage
        if self.converted_message:
            return int(self.converted_message[bit_position - 1])
        # extraction stage
        else:
            return 0

    def get_current_position(self):
        return self.bit_position

    def set_message(self, binary_msg: str = ""):
        self.message = binary_msg
        self.converted_message = self._convert_binary_to_base(binary_msg)

    def _convert_binary_to_base(self, binary_msg: str):
        decimal = int(binary_msg, 2)
        converted_msg = self._numberToBase(decimal, self.base)
        converted_msg = "0" * (self.converted_msg_length - len(converted_msg)) + converted_msg
        return converted_msg

    def _numberToBase(self, n, b):
        """
        https://stackoverflow.com/a/28666223
        """
        if n == 0:
            return str(0)
        digits = []
        while n:
            digits.append(int(n % b))
            n //= b
        return "".join(map(str, digits[::-1]))
    def flush_position(self):
        positions = "".join(list(map(str, self.bit_position_list)))
        self.bit_position_list = []
        self.green_cnt_by_position = {i: [0 for _ in range(self.base)] for i
                                      in range(1, self.converted_msg_length + 1)}
        return [positions]


class MPACProcessor(WatermarkBase, LogitsProcessor):
    """LogitsProcessor modifying model output scores in a pipe. Can be used in any HF pipeline to modify scores to fit the watermark,
    but can also be used as a standalone tool inserted for any model producing scores in between model outputs and next token sampler.
    """

    def __init__(self, *args, store_spike_ents: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.store_spike_ents = store_spike_ents
        self.spike_entropies = None
        self.feedback = kwargs.get('use_feedback', False)
        self.feedback_args = kwargs.get('feedback_args', {})
        if self.store_spike_ents:
            self._init_spike_entropies()
            
        self.topk_records = {'top1': [], 'top5': [], 'top10': []}


    def _init_spike_entropies(self):
        alpha = torch.exp(torch.tensor(self.delta)).item()
        gamma = self.gamma

        self.z_value = ((1 - gamma) * (alpha - 1)) / (1 - gamma + (alpha * gamma))
        self.expected_gl_coef = (gamma * alpha) / (1 - gamma + (alpha * gamma))

        # catch for overflow when bias is "infinite"
        if alpha == torch.inf:
            self.z_value = 1.0
            self.expected_gl_coef = 1.0

    def _get_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]
        for b_idx, ent_tensor_list in enumerate(self.spike_entropies):
            for ent_tensor in ent_tensor_list:
                spike_ents[b_idx].append(ent_tensor.item())
        return spike_ents

    def _get_and_clear_stored_spike_ents(self):
        spike_ents = self._get_spike_entropies()
        self.spike_entropies = None
        return spike_ents

    def _compute_spike_entropy(self, scores):
        # precomputed z value in init
        probs = scores.softmax(dim=-1)
        denoms = 1 + (self.z_value * probs)
        renormed_probs = probs / denoms
        sum_renormed_probs = renormed_probs.sum()
        return sum_renormed_probs

    def _calc_greenlist_mask(
        self, scores: torch.FloatTensor, greenlist_token_ids
    ) -> torch.BoolTensor:
        # Cannot lose loop, greenlists might have different lengths
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for b_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[b_idx][greenlist] = True
        return green_tokens_mask

    def _bias_greenlist_logits(
        self, scores: torch.Tensor, colorlist_mask: torch.Tensor, greenlist_bias: float,
            denylist_flag=False
    ) -> torch.Tensor:
        
        if denylist_flag:
            scores[colorlist_mask] = 0
        else:
            scores[colorlist_mask] = scores[colorlist_mask] + greenlist_bias
            
        # print(colorlist_mask[0].sum().item(), "tokens in greenlist")
        return scores

    def _score_rejection_sampling(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, tail_rule="fixed_compute"
    ) -> list[int]:
        """Generate greenlist based on current candidate next token. Reject and move on if necessary. Method not batched.
        This is only a partial version of Alg.3 "Robust Private Watermarking", as it always assumes greedy sampling. It will still (kinda)
        work for all types of sampling, but less effectively.
        To work efficiently, this function can switch between a number of rules for handling the distribution tail.
        These are not exposed by default.
        """
        sorted_scores, greedy_predictions = scores.sort(dim=-1, descending=True)

        final_greenlist = []
        for idx, prediction_candidate in enumerate(greedy_predictions):
            greenlist_ids = self._get_greenlist_ids(
                torch.cat([input_ids, prediction_candidate[None]], dim=0)
            )  # add candidate to prefix
            if prediction_candidate in greenlist_ids:  # test for consistency
                final_greenlist.append(prediction_candidate)

            # What follows below are optional early-stopping rules for efficiency
            if tail_rule == "fixed_score":
                if sorted_scores[0] - sorted_scores[idx + 1] > self.delta:
                    break
            elif tail_rule == "fixed_list_length":
                if len(final_greenlist) == 10:
                    break
            elif tail_rule == "fixed_compute":
                if idx == 40:
                    break
            else:
                pass  # do not break early
        return torch.as_tensor(final_greenlist, device=input_ids.device)


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""
        
        top1_ids = torch.topk(scores[0], k=1).indices.tolist()
        top5_ids = torch.topk(scores[0], k=5).indices.tolist()
        top10_ids = torch.topk(scores[0], k=10).indices.tolist()


        self.topk_records['top1'].append(top1_ids)
        self.topk_records['top5'].append(top5_ids)
        self.topk_records['top10'].append(top10_ids)

        # this is lazy to allow us to co-locate on the watermarked model's device
        self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng
        feedback_bias = self.feedback_args.get("feedback_bias", -1)

        #TODO: batchify ecc with feedback
        list_of_greenlist_ids = [None for _ in input_ids]  # Greenlists could differ in length
        list_of_blacklist_ids = [[] for _ in input_ids]
        feedback_flag = False
        for b_idx, input_seq in enumerate(input_ids):
            if self.self_salt:
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                greenlist_ids = self._get_greenlist_ids(input_seq)
            list_of_greenlist_ids[b_idx] = greenlist_ids
            if self.use_fixed_position:
                self.position_increment += 1
            self.bit_position_list.append(self.bit_position)

            # logic for computing and storing spike entropies for analysis
            if self.store_spike_ents:
                if self.spike_entropies is None:
                    self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[b_idx].append(self._compute_spike_entropy(scores[b_idx]))

            # logic for whether to expand the greenlist and suppress blacklist
            if self.feedback:
                # increment the colorlist for the current token
                context_length = self.context_width + 1 - self.self_salt
                if input_seq.shape[-1] < context_length:
                    continue
                ngram = input_seq[-context_length:]
                ngram = tuple(ngram.tolist())
                target = ngram[-1]
                prefix = ngram if self.self_salt else ngram[:-1]
                colorlist_ids = self._get_colorlist_ids(torch.as_tensor(prefix, device=self.device))
                pos = self.get_current_position()
                for c_idx, cl in enumerate(colorlist_ids[:self.base]):
                    if target in cl:
                        self.green_cnt_by_position[pos][c_idx] += 1
                colorlist_flag, pos = self._get_ngram_score_cached(prefix, target)

                eta = self.feedback_args.get("eta", 3)
                tau = self.feedback_args.get("tau", 2)
                msg = int(self.converted_message[pos-1])
                preliminary_cond = sum(self.green_cnt_by_position[pos]) >= eta
                if preliminary_cond:
                    max_color = np.argmax(self.green_cnt_by_position[pos])
                    cond_1 = max_color != msg
                    colorlist_ids = list(colorlist_ids)
                    if cond_1:
                        # feedback_flag = True
                        list_of_blacklist_ids[b_idx] = colorlist_ids[max_color]
                        continue
                    if tau == -1:
                        continue
                    top2_color = np.argpartition(self.green_cnt_by_position[pos], -2)[-2]
                    color_cnt_diff = self.green_cnt_by_position[pos][max_color] - \
                                     self.green_cnt_by_position[pos][top2_color]

                    cond_2 = color_cnt_diff < tau + 1
                    if cond_2:
                        list_of_blacklist_ids[b_idx] = colorlist_ids[top2_color]


        green_tokens_mask = self._calc_greenlist_mask(
            scores=scores, greenlist_token_ids=list_of_greenlist_ids
        )
        scores = self._bias_greenlist_logits(
            scores=scores, colorlist_mask=green_tokens_mask,
            greenlist_bias=self.delta
        )
        if self.feedback:
            # suppress the black list when condition is satisfied
            black_tokens_mask = self._calc_greenlist_mask(
                scores=scores, greenlist_token_ids=list_of_blacklist_ids
            )
            scores = self._bias_greenlist_logits(
                scores=scores, colorlist_mask=black_tokens_mask,
                greenlist_bias=feedback_bias, denylist_flag=True
            )

        return scores



class MPACDetector(WatermarkBase):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["unicode"],
        ignore_repeated_ngrams: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"
        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)
        self.normalizers = [normalization_strategy_lookup(n) for n in normalizers]
        self.ignore_repeated_ngrams = ignore_repeated_ngrams
        


    def detect(self, text: str = None, tokenized_text: list[int] = None, **kwargs) -> str:
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        for normalizer in self.normalizers:
            text = normalizer(text)

        if tokenized_text is None:
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            if self.tokenizer is not None and tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]

        return ''.join(map(str, self._score_sequence(tokenized_text, **kwargs)))

    def _score_sequence(self, input_ids: torch.Tensor, message: str = "", **kwargs):
        _, frequencies_table, ngram_to_position_lookup, green_cnt_by_position, _ = self._score_ngrams_in_passage_sequential(input_ids)

        position_cnt = {}
        for k, v in ngram_to_position_lookup.items():
            freq = frequencies_table[k]
            position_cnt[v] = position_cnt.get(v, 0) + freq

        list_decoded_msg, _, _ = self._predict_message(position_cnt, green_cnt_by_position, None)
        return ''.join(map(str, list_decoded_msg[0]))

    def _compute_max_multinomial_p_val(self, observed_count, T):
        if T <= 0:
            return 1
        poiss = scipy.stats.poisson
        normal = scipy.stats.norm
        k = self.base
        a = observed_count - 1
        poiss_cdf_X = poiss.cdf(a, T / k)
        normal_approx_W = normal.cdf(0.5 / np.sqrt(T)) - normal.cdf(-0.5 / np.sqrt(T))
        # print(f"T={T}, poiss_cdf_X={poiss_cdf_X}, normal_approx_W={normal_approx_W}")

        log_max_multi_cdf = math.log(np.sqrt(2 * math.pi * T)) + k * math.log(poiss_cdf_X) + math.log(normal_approx_W)
        max_multi_cdf = math.exp(log_max_multi_cdf)
        return 1 - min(1, max_multi_cdf)

    def _predict_message(self, position_cnt, green_cnt_by_position, p_val_per_pos, num_candidates=16):
        msg_prediction = []
        for pos in range(1, self.converted_msg_length + 1):
            if position_cnt.get(pos) is None:
                preds = random.sample(list(range(self.base)), 2)
                pred = preds[0]
            else:
                green_counts = green_cnt_by_position[pos]
                pred, _ = max(enumerate(green_counts), key=lambda x: (x[1], x[0]))
            msg_prediction.append(pred)
        return [msg_prediction], [], 0

    def _score_ngrams_in_passage_sequential(self, input_ids: torch.Tensor):
        frequencies_table = {}
        ngram_to_position_lookup = {}
        green_cnt_by_position = {i: [0 for _ in range(self.base)] for i in range(1, self.converted_msg_length + 1)}
        increment = self.context_width - self.self_salt
        for idx in range(self.context_width, len(input_ids) + self.self_salt):
            pos = increment % self.converted_msg_length + 1
            ngram = input_ids[idx - self.context_width: idx + 1 - self.self_salt]
            ngram = tuple(ngram.tolist())
            frequencies_table[ngram] = frequencies_table.get(ngram, 0) + 1
            target = ngram[-1]
            prefix = ngram if self.self_salt else ngram[:-1]
            colorlist_flag, pos = self._get_ngram_score_cached(prefix, target)
            for f_idx, flag in enumerate(colorlist_flag):
                if flag:
                    green_cnt_by_position[pos][f_idx] += 1
            ngram_to_position_lookup[ngram] = pos
            increment += 1
        return None, frequencies_table, ngram_to_position_lookup, green_cnt_by_position, []

    def _get_ngram_score_cached(self, prefix: tuple[int], target: int, cand_msg=None):
        colorlist_ids = self._get_colorlist_ids(torch.as_tensor(prefix, device=self.device))
        colorlist_flag = [(target in cl) for cl in colorlist_ids[:self.base]]
        return colorlist_flag, self.get_current_position()

##########################################################################
# Ngram iteration from nltk, extracted to remove the dependency
# Natural Language Toolkit: Utility functions
#
# Copyright (C) 2001-2023 NLTK Project
# Author: Steven Bird <stevenbird1@gmail.com>
#         Eric Kafe <kafe.eric@gmail.com> (acyclic closures)
# URL: <https://www.nltk.org/>
# For license information, see https://github.com/nltk/nltk/blob/develop/LICENSE.txt
##########################################################################


def ngrams(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n - 1))
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.


from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))