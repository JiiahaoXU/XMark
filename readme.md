# XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts

This repository contains the official implementation for the ACL'26 paper **"XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts"**.

## Overview

We evaluate multi-bit text watermarking methods for LLM-generated text. The main pipeline:

1. loads prompts from a Hugging Face dataset,
2. generates text with and without watermarking,
3. detects the embedded binary message from watermarked generations,
4. reports bit accuracy and optional quality metrics such as top-k hit ratio, BERTScore, ROUGE, and perplexity.

The current code supports the following watermark methods through `--wm_method`:

- `xmark` (Ours)
- `loso` (Ours)
- `mpac`
- `rsbh`
- `cycleshift`
- `depthw`
- `stealthink`

The currently supported dataset loader is:

- `c4`, using `allenai/c4` with the `realnewslike` validation split in streaming mode.

You can easily integrate other datasets into our codebase.

## Repository Structure

- `main.py`: main generation, detection, and evaluation entry point.
- `args.py`: command-line argument definitions.
- `model.py`: Hugging Face model/tokenizer loading and oracle-model mapping for perplexity evaluation.
- `dataset.py`: dataset loading utilities.
- `processor_detector.py`: factory functions for watermark processors and detectors.
- `generation_helper.py`: generation kwargs and prompt tokenization helpers.
- `generate_code.py`: binary message generation/loading utilities.
- `watermark_methods/`: watermark implementations and detectors.
- `run.sh`: example batch script.

## Installation

We provide a copy of our environment in `xmark.yml`. You can create and activate the environment with:

```bash
conda env create -f xmark.yml
conda activate xmark
```

Some Hugging Face models, such as Llama-family models, require access approval and authentication. If needed, log in before running:

```bash
huggingface-cli login
```

## Quick Start

Run XMark on C4 with the default Llama-2-7B model:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --wm_method xmark \
    --dataset c4 \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --bits 8 \
    --delta 2 \
    --num_users 50 \
    --num_prompts 2 \
    --tokens_for_detection_per_prompt 75 \
    --evaluation
```

You can also use the provided script:

```bash
bash run.sh
```

Logs and source-code backups are written under `logs/`.

## Key Arguments

- `--wm_method`: watermark method, for example `xmark`, `mpac`, or `loso`.
- `--dataset`: dataset name. 
- `--model_name_or_path`: Hugging Face model name or local model path.
- `--bits`: number of bits for the message.
- `--delta`: watermarking bias.
- `--num_users`: number of users to evaluate.
- `--num_prompts`: number of prompts per user.
- `--tokens_for_detection_per_prompt`: number of generated tokens per prompt.

## Notes

- `base_codes.json` is loaded automatically. If it is missing, `generate_code.py` will regenerate binary messages for supported bit lengths.
- Perplexity evaluation uses an oracle model defined in `model.py`. If you change the main model, make sure an oracle mapping exists or add one to `ORACLE_MODEL_MAP`.

## Citation

Coming soon
