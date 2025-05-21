#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2

# Run training with DeepSpeed
deepspeed ../../scripts/training/patched_train_gemma.py \
  --deepspeed ../../configs/training/ds_config_zero3.json \
  --model_name_or_path google/gemma-2-2b \
  --train_file ../../datasets/test_dataset/test_data.json \
  --output_dir ../../outputs/gemma-2-2b-finetuned \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --warmup_steps 2 \
  --logging_steps 1 \
  --save_steps 5 \
  --use_lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_4bit True \
  --max_seq_length 512 \
  --fp16 False \
  --report_to none