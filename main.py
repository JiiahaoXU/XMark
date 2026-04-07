import numpy as np
import torch
import logging

from args import parse_args
from functools import partial
from bert_score import score as bert_score
from transformers import LogitsProcessorList
from transformers import logging as t_logging
t_logging.set_verbosity_error()

from dataset import load_dataiter_by_name, load_text_by_iter
from model import load_model
from utils import setup_logging, compute_perplexity_from_decoded_texts, compute_bit_accuracy, compute_rouge_distortion
from generate_code import load_or_generate_base_code   
from processor_detector import get_detector, get_decoded_message, get_processor
from generation_helper import get_token_input_ids, get_generation_kwargs


def text_detection(user_id, binary_code, watermarked_texts, tokenizer, args, device):
    
    watermark_detector = get_detector(args.wm_method, args, tokenizer, device)
    decoded_message = get_decoded_message(args.wm_method, watermark_detector, watermarked_texts)
    
    logging.info(f"User {user_id}/{args.num_users} - Recovered message:     {' '.join(decoded_message[i:i+4] for i in range(0, len(decoded_message), 4))}")
    logging.info(f"User {user_id}/{args.num_users} - Ground-truth message:  {' '.join(binary_code[i:i+4] for i in range(0, len(binary_code), 4))}")
    

    bit_acc = compute_bit_accuracy(decoded_message, binary_code)
    logging.info(f"User {user_id}/{args.num_users} - Bit Accuracy: {bit_acc:.2f}")
    
    return bit_acc


def text_generation(prompt, args, model=None, device=None, tokenizer=None, current_binary_code=None):
    
    watermark_processor = get_processor(args.wm_method, args, tokenizer, current_binary_code, device)

    gen_kwargs = get_generation_kwargs(args)

    LLM_without_watermarking = partial(
        model.generate,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, return_legacy_cache=True, no_repeat_ngram_size=5,
        **gen_kwargs
    )
    
    LLM_with_watermarking = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]),
        return_dict_in_generate=True, 
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_legacy_cache=True,
        no_repeat_ngram_size=5,
        **gen_kwargs
    )
    
    # Get tokenized input IDs for both watermarked and non-watermarked generation
    token_inputs = get_token_input_ids(args, model, tokenizer, prompt, device)  

    # Generate text with and without watermarking, ensuring the same random seed for fair comparison
    torch.manual_seed(args.generation_seed)
    output_wo_watermark = LLM_without_watermarking(**token_inputs)

    output_w_watermark_ = LLM_with_watermarking(**token_inputs)                                                                                               

    logit = output_w_watermark_[1]
    logit = torch.stack(logit)

    output_w_watermark = output_w_watermark_[0]
    output_without_watermark = output_wo_watermark[:, token_inputs["input_ids"].shape[-1]:]
    output_with_watermark = output_w_watermark[:, token_inputs["input_ids"].shape[-1]:]
    
    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
    
    # Analyze top-k hit ratios for the generated tokens
    generated_ids = output_w_watermark_.sequences[0]  
    gen_token_ids = generated_ids[token_inputs["input_ids"].shape[-1]:]  

    topk_records = watermark_processor.topk_records
    topk_hit_ratio = {}

    for k in topk_records.keys(): 
        topk_list = topk_records[k]
        assert len(gen_token_ids) == len(topk_list), f"Length mismatch for {k}"

        hit_count = 0
        for token_id, topk in zip(gen_token_ids, topk_list):
            if token_id.item() in topk:
                hit_count += 1
        hit_ratio = hit_count / len(gen_token_ids)
        topk_hit_ratio[k] = hit_ratio 

    return (output_without_watermark,
            output_with_watermark,
            decoded_output_without_watermark,
            decoded_output_with_watermark, topk_hit_ratio)
    

def running(args, model, tokenizer, device, ds_iterator, code_dict):
    bit_acc_list = []
    all_user_ratios = []
    generated_texts = {}
    precision_all_users = []
    recall_all_users = []
    f1_all_users = []
    all_watermark_ppl = []
    all_non_watermark_ppl = []
    all_rouge_1 = []
    all_rouge_2 = []
    all_rouge_L = []
    all_rouge_Lsum =[]   
    user_perplexity_stats = {}
    code = []

    for user_id in range(1, 1 + args.num_users):
        
        current_binary_code = code_dict[str(args.bits)][str(user_id)]

        code.append(current_binary_code)
        prompts = []
        while len(prompts) < args.num_prompts:
            prompt = load_text_by_iter(args.dataset, ds_iterator, args)
            if "Proclus of Athens (*412–485 C.E.)" in prompt:
                continue
            prompts.append(prompt)

        watermarked_texts = []
        non_watermarked_texts = []
        watermarked_tokens = []
        non_watermarked_tokens = []
        user_prompt_ratios = []

        for prompt in prompts:
            non_watermarked_token, watermarked_token, decoded_non_watermarked, decoded_watermarked, topk_hit_ratio = text_generation(
                prompt,
                args,
                model=model,
                device=device,
                tokenizer=tokenizer,
                current_binary_code=current_binary_code
            )
            watermarked_texts.append(decoded_watermarked)
            non_watermarked_texts.append(decoded_non_watermarked)
            user_prompt_ratios.append(topk_hit_ratio)
            watermarked_tokens.append(watermarked_token)
            non_watermarked_tokens.append(non_watermarked_token)

        generated_texts[user_id] = {
            "prompts": prompts,
            "watermarked_texts": watermarked_texts,
            "non_watermared_texts": non_watermarked_texts
        }
    
        if args.evaluation:
            
            # Bit-level Watermark Detection
            bit_acc = text_detection(
                user_id=user_id,
                binary_code=current_binary_code, watermarked_texts=watermarked_texts, tokenizer=tokenizer, args=args, device=device)
            bit_acc_list.append(bit_acc)
            
            
            # Top-k Hit Ratio Aggregation
            user_avg_ratio = {}
            for k in user_prompt_ratios[0].keys():
                user_avg_ratio[k] = sum(r[k] for r in user_prompt_ratios) / len(user_prompt_ratios)
            all_user_ratios.append(user_avg_ratio)
            logging.info(
                f"User {user_id}/{args.num_users} - Avg Top-k Hit Ratio - Top-1: {user_avg_ratio['top1']:.2f}, "
                f"Top-5: {user_avg_ratio['top5']:.2f}, Top-10: {user_avg_ratio['top10']:.2f}"
            )

            # BERTScore Evaluation
            P, R, F1 = bert_score(watermarked_texts, non_watermarked_texts, lang='en')
            precision_all_users.append(P.mean().item())
            recall_all_users.append(R.mean().item())
            f1_all_users.append(F1.mean().item())
            logging.info(f"User {user_id}/{args.num_users} - BERTScore F1: {F1.mean().item():.4f} "
                        f"(P: {P.mean().item():.4f}, R: {R.mean().item():.4f})")
            
            rogue_score = compute_rouge_distortion(watermarked_texts, non_watermarked_texts)
            
            all_rouge_1.append(rogue_score['rouge1'])
            all_rouge_2.append(rogue_score['rouge2'])
            all_rouge_L.append(rogue_score['rougeL'])
            all_rouge_Lsum.append(rogue_score['rougeLsum'])
            
            logging.info(f"User {user_id}/{args.num_users} - ROUGE-1: {rogue_score['rouge1']:.4f}, ROUGE-2: {rogue_score['rouge2']:.4f}, ROUGE-L: {rogue_score['rougeL']:.4f}, ROUGE-Lsum: {rogue_score['rougeLsum']:.4f}")

            # PPL Evaluation
            if user_id == 1:
                oracle_model = load_model(args, is_oracle=True)
            wm_ppl = compute_perplexity_from_decoded_texts(prompts, watermarked_texts, tokenizer, oracle_model, args, device)
            non_wm_ppl = compute_perplexity_from_decoded_texts(prompts, non_watermarked_texts, tokenizer, oracle_model, args, device)
            all_watermark_ppl.append(wm_ppl)
            all_non_watermark_ppl.append(non_wm_ppl)
            user_perplexity_stats[user_id] = {
                "watermarked_perplexity": wm_ppl,
                "non_watermarked_perplexity": non_wm_ppl
            }
            logging.info(f"User {user_id}/{args.num_users} - Watermarked PPL = {wm_ppl:.2f}, Non-Watermarked PPL = {non_wm_ppl:.2f}")
        

    if args.evaluation:
        bit_acc_list = [bit_acc * 100 for bit_acc in bit_acc_list]
        detection_success_rate = [100 if acc == 100.0 else 0 for acc in bit_acc_list]

        avg_ppl_stats = {
            "average_watermarked_perplexity": all_watermark_ppl,
            "average_non_watermarked_perplexity": all_non_watermark_ppl
        }
        
        
        top1_list = [r['top1'] for r in all_user_ratios]
        top5_list = [r['top5'] for r in all_user_ratios]
        top10_list = [r['top10'] for r in all_user_ratios]
        
        logging.info(f"[Top-k Hit Ratio] - Top-1:  {np.mean(top1_list) * 100:.2f}% ± {np.std(top1_list) * 100:.2f}%, "
            f"Top-5: {np.mean(top5_list) * 100:.2f}% ± {np.std(top5_list) * 100:.2f}%, "
            f"Top-10: {np.mean(top10_list) * 100:.2f}% ± {np.std(top10_list) * 100:.2f}%"
        )
            
        logging.info(f"[Detection]       - Avg Success Rate: {np.mean(detection_success_rate):.2f}% +- {np.std(detection_success_rate):.2f}%, "
                    f"Bit Accuracy: {np.mean(bit_acc_list):.2f}% +- {np.std(bit_acc_list):.2f}%")
        logging.info(f"[Evaluation]      - Avg BERTScore F1: {np.mean(f1_all_users):.4f} ± {np.std(f1_all_users):.4f}, "
                    f"Precision: {np.mean(precision_all_users):.4f}, Recall: {np.mean(recall_all_users):.4f}")
        logging.info(f"[Evaluation]      - Avg WMed PPL    : {np.mean(avg_ppl_stats['average_watermarked_perplexity']):.2f} ± {np.std(avg_ppl_stats['average_watermarked_perplexity']):.2f}, "
                    f"Avg noWMed PPL : {np.mean(avg_ppl_stats['average_non_watermarked_perplexity']):.2f} ± {np.std(avg_ppl_stats['average_non_watermarked_perplexity']):.2f}")
        logging.info(f"[Evaluation]      - Avg ROUGE-1     : {np.mean(all_rouge_1):.4f} ± {np.std(all_rouge_1):.4f}, Avg ROUGE-2: {np.mean(all_rouge_2):.4f} ± {np.std(all_rouge_2):.4f}")
        logging.info(f"[Evaluation]      - Avg ROUGE-L     : {np.mean(all_rouge_L):.4f} ± {np.std(all_rouge_L):.4f}, Avg ROUGE-Lsum: {np.mean(all_rouge_Lsum):.4f} ± {np.std(all_rouge_Lsum):.4f}")
        
    return generated_texts

    
def main():
    
    args = parse_args()
    
    args.num_token_detection = args.tokens_for_detection_per_prompt * args.num_prompts
        
    _ = setup_logging(args)

    model, tokenizer, device = load_model(args)
    ds_iterator = load_dataiter_by_name(args.dataset)
    code_dict = load_or_generate_base_code('./base_codes.json')

    _ = running(args, model, tokenizer, device, ds_iterator, code_dict)
        

if __name__ == "__main__":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    main()



