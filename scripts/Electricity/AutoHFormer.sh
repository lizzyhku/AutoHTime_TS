#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

python3 -u run_longExp.py \
    --is_training 1 \
    --model_id "Electricity_autoregressive" \
    --model "AutoHFormer" \
    --data "custom" \
    --random_seed 2021 \
    --root_path "./datasets/" \
    --data_path "electricity/electricity.csv" \
    --features M \
    --enc_in 321 \
    --hidden_dim 64 \
    --seq_len 336 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.1 \
    --window_size 32 \
    --gamma 0.5 \
    --d_k 128 \
    --d_v 128 \
    --alpha 0.1 \
    --beta 0.1 \
    --attn_decay_type "powerLaw" \
    --attn_decay_scale 0.5 \
    --patch_len 4 \
    --stride 8 \
    --train_epochs 100 \
    --patience 20 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --channel_independence 0 \
    --dec_in 321 \
    --c_out 321 \
    --itr 1 \
    "$@" >> logs/LongForecasting/AutoHFormer_Electricity_336_96.log 