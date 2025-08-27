#!/bin/bash
CUDA_VISIBLE_DEVICES='0' \
python main_single.py \
    --lr=0.0001 \
    --backbone="vgg16_bn" \
    --ce_loss_coef=1.0 \
    --point_loss_coef=5.0 \
    --bce_loss_coef=10.0 \
    --smoothl1_loss_coef=1.0 \
    --eos_coef=0.5 \
    --dec_layers=2 \
    --hidden_dim=256 \
    --dim_feedforward=512 \
    --nheads=8 \
    --dropout=0.0 \
    --epochs=1500 \
    --dataset_file="SHA" \
    --eval_freq=5 \
    --output_dir='pet_model' \
    --force_single_gpu
