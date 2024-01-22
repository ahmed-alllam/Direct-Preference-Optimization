#!/usr/bin/env sh

python src/train.py \
    --epochs 10 \
    --batch_size 64 \
    --max_length 512 \
    --lr 1e-6 \
    --beta 0.1 \
    --seed 2003 \
    --model_name "microsoft/phi-2" \
    --dataset_name "jondurbin/truthy-dpo-v0.1" \
    --wandb_project "truthy-dpo"
