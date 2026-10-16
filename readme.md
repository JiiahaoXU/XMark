# XMark and MajorMark: Multi-Bit Watermarking for LLM-Generated Texts

This repository hosts the official implementations of both papers:

- **"XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts"**, ACL 2026.
- **"Majority Bit-Aware Watermarking For Large Language Models"**, NeurIPS 2026, including MajorMark and MajorMark+.

The MajorMark implementation has been merged into this XMark repository. Readers of either paper can use this repository to run the methods under a shared generation and evaluation pipeline.

If you have any issues using this repo or want to discuss anything related to our work, feel free to contact Jiahao @ jiahaox@unr.edu.

## Overview

We evaluate multi-bit text watermarking methods for LLM-generated text. The main pipeline:

1. loads prompts from a Hugging Face dataset,
2. generates text with and without watermarking,
3. detects the embedded binary message from watermarked generations,
4. reports bit accuracy and optional quality metrics such as top-k hit ratio, BERTScore, ROUGE, and perplexity.

The current code supports the following watermark methods through `--wm_method`:

- `xmark` (Ours)
- `loso` (Ours)
- `majormark` (Ours)
- `majormarkplus` (Ours)
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

We provide a copy of our environment in `llmwm.yml`. You can create and activate the environment with:

```bash
conda env create -f llmwm.yml
conda activate llmwm
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

To run MajorMark or MajorMark+, replace `--wm_method xmark` above with `--wm_method majormark` or `--wm_method majormarkplus`. For MajorMark+, the default `--block_size -1` uses half the message length; use an integer such as `--block_size 4` to set it explicitly.

You can also use the provided script and set its `methods` array to the methods you want to compare, for example `methods=(xmark majormark majormarkplus)`:

```bash
bash run.sh
```

Logs and source-code backups are written under `logs/`.

## Key Arguments

- `--wm_method`: watermark method, for example `xmark`, `majormark`, or `majormarkplus`.
- `--dataset`: dataset name. 
- `--model_name_or_path`: Hugging Face model name or local model path.
- `--bits`: number of bits for the message.
- `--block_size`: MajorMark+ block size, at least 2 and a divisor of `bits`; `-1` (default) uses `bits // 2`. Each block must contain both 0 and 1.
- `--hash_key`: watermark hash key, shared by the processor and detector for XMark, MajorMark, and MajorMark+.
- `--delta`: watermarking bias.
- `--num_users`: number of users to evaluate.
- `--num_prompts`: number of prompts per user.
- `--tokens_for_detection_per_prompt`: number of generated tokens per prompt.

## Notes

- `base_codes.json` is loaded automatically. If it is missing, `generate_code.py` will regenerate binary messages for supported bit lengths.
- Perplexity evaluation uses an oracle model defined in `model.py`. If you change the main model, make sure an oracle mapping exists or add one to `ORACLE_MODEL_MAP`.

## Citation

We provide the BibTeX records of our papers for your convenience!

```
@inproceedings{xmark,
    title = "{XM}ark: Reliable Multi-Bit Watermarking for {LLM}-Generated Texts",
    author = "Xu, Jiahao  and
      Hu, Rui  and
      Kotevska, Olivera  and
      Zhang, Zikai",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.672/",
    pages = "14747--14763",
    ISBN = "979-8-89176-390-6"
}

@misc{majormark,
      title={Majority Bit-Aware Watermarking For Large Language Models}, 
      author={Jiahao Xu and Rui Hu and Olivera Kotevska and Zikai Zhang},
      year={2026},
      eprint={2508.03829},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.03829}, 
}

```
