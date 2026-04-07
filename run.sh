#!/bin/bash

num_prompts=2
num_users=50
base_model=meta-llama/Llama-2-7b-hf
device=0

deltas=(2 4)
bits=(8)
tokens_for_detection_per_prompt=(75)
datasets=(c4)
methods=(xmark)

# enumerate all combinations of parameters and run the script
for delta in "${deltas[@]}"; do
    for bit in "${bits[@]}"; do
        for tokens_for_detection in "${tokens_for_detection_per_prompt[@]}"; do
            for dataset in "${datasets[@]}"; do
                for wm in "${methods[@]}"; do
                    echo "==== Running bit=$bit, wm_method=$wm, dataset=$dataset, tokens_for_detection=$tokens_for_detection ===="
                    CUDA_VISIBLE_DEVICES=$device \
                    python main.py \
                        --bits $bit \
                        --wm_method $wm \
                        --delta $delta \
                        --num_users $num_users \
                        --num_prompts $num_prompts \
                        --tokens_for_detection_per_prompt $tokens_for_detection \
                        --model_name_or_path $base_model \
                        --evaluation \
                        --dataset $dataset
                done
            done
        done
    done
done