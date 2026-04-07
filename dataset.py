import time
import requests
from datasets import load_dataset, config as hf_datasets_config

def load_dataiter_by_name(dataset_name):
    if dataset_name == 'c4':
        dataset_config_name = "realnewslike"

        hf_datasets_config.DOWNLOADER_TIMEOUT = 60
        hf_datasets_config.DOWNLOADER_MAX_RETRIES = 10

        max_retries = 5
        for attempt in range(max_retries):
            try:
                dataset = load_dataset(
                    'allenai/c4',
                    dataset_config_name,
                    split="validation",
                    streaming=True,
                    trust_remote_code=True
                )
                ds_iterator = iter(dataset)
                break
            except requests.exceptions.ReadTimeout as e:
                print(f"[Retry {attempt+1}/{max_retries}] Timeout: {e}")
                time.sleep(5)
        else:
            raise RuntimeError("Failed to load 'c4' dataset after multiple retries.")

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    return ds_iterator


def load_text_by_iter(dataset_name, ds_iterator, args):
    if dataset_name == 'c4':
        sample = next(ds_iterator)
        text = sample['text']
        if args.prompt_max_length is not None and len(text) >= args.prompt_max_length:
            input_text = text[:args.prompt_max_length]
        else:
            input_text = text[:]

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    return input_text
