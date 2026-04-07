import torch
import logging
import time
import glob
import shutil
import math
import evaluate
import os
evaluate.logging.set_verbosity_error()


def compute_bit_accuracy(pred_bits: str, true_bits: str) -> float:
    assert len(pred_bits) == len(true_bits), "Bits must be of equal length"
    correct = sum(p == t for p, t in zip(pred_bits, true_bits))
    return correct / len(true_bits)


def setup_logging(args):
    """
    Sets up the logging environment and creates necessary directories.
    
    Args:
        args: Arguments object containing logging parameters like non_iid, alpha, data, and aggr.
        
    Returns:
        dir_path: The directory path where logs and backup files are stored.
    """
    log_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    logPath = "logs"
    time_str = time.strftime("%Y-%m-%d-%H-%M")

    fileName = "evaluation"

    dir_path = '%s/%s/bits_%d/%s_%s/delta_%.1f_token_%d_bs_%d/%s_%s/' % (logPath, args.dataset, args.bits, args.wm_method, args.model_name_or_path.replace('/', '_'), float(args.delta), args.num_token_detection, args.block_size, time_str, 'log')
    file_path = dir_path + 'backup_file/'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for file in glob.glob("*.py"):
        shutil.copy(file, os.path.join(file_path, os.path.basename(file)))
        
    src_dir = "./watermark_methods"
    dst_dir = os.path.join(file_path, "watermark_methods")

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    
    # Set up file handler for logging
    file_handler = logging.FileHandler(os.path.join(dir_path, f"{fileName}.log"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    
    # Set up console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    # Log initial arguments
    logging.info(args)
    return dir_path


def compute_perplexity_from_decoded_texts(prompts, responses, tokenizer, model, args, device="cuda"):
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for prompt, response in zip(prompts, responses):
        full_input = prompt + response

        # Tokenize full input
        inputs = tokenizer(full_input, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(device)

        # Get prompt token length
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        prompt_length = prompt_ids.shape[1]

        # Construct labels: mask prompt tokens
        labels = input_ids.clone()
        labels[0, :prompt_length] = -100  # Do not compute loss on prompt

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss  # Averaged over unmasked tokens

        # Count number of tokens used for loss
        num_effective_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_effective_tokens
        total_tokens += num_effective_tokens

    if total_tokens == 0:
        return float("inf")  # Avoid division by zero

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def compute_rouge_distortion(watermarked_texts, non_watermarked_texts):
    
    rouge_metric = evaluate.load("rouge")

    results = rouge_metric.compute(
        predictions=watermarked_texts, 
        references=non_watermarked_texts, 
        use_stemmer=True
    )
    
    return results

