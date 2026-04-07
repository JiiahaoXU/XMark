from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ORACLE model is used for calculating perplexity.
ORACLE_MODEL_MAP = {
    "meta-llama/Llama-2-7b-hf": "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-70B",
    "Qwen/Qwen2.5-7B": "Qwen/Qwen2.5-32B",
    "google/gemma-2b": "google/gemma-7b",
    "openai-community/gpt2-large": "openai-community/gpt2-xl",
    "meta-llama/Llama-2-7b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
}


def get_device(args):
    return "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"


def load_model(args, is_oracle=False):
    device = get_device(args)

    model_name = args.model_name_or_path
    if is_oracle:
        if model_name not in ORACLE_MODEL_MAP:
            raise ValueError(f"Please specify the oracle model for {model_name}.")
        model_name = ORACLE_MODEL_MAP[model_name]

    model_kwargs = {}
    if args.load_fp16:
        model_kwargs["torch_dtype"] = torch.float16
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if "device_map" not in model_kwargs:
        model.to(device)

    model.eval()

    if is_oracle:
        return model

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer, device
