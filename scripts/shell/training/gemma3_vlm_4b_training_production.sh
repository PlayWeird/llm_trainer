#!/bin/bash

# Training script for Gemma-3-4B Vision-Language Model on Flickr8k dataset
# This smaller model should run more comfortably on 3x RTX 3090 GPUs

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2  # Use all 3 GPUs
export NCCL_DEBUG=INFO  # Helpful for debugging NCCL issues
export NCCL_P2P_DISABLE=1  # Can help with certain multi-GPU configuration
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer parallelism warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1  # Reduce noise from warnings

# Create output directory if it doesn't exist
mkdir -p ../../../outputs/gemma3-4b-vlm-flickr8k

# Run training with DeepSpeed ZeRO-3
deepspeed ../../training/gemma3_vlm_training_production.py \
  --deepspeed ../../../configs/training/ds_config_zero3.json \
  --model_name_or_path google/gemma-3-4b-it \
  --dataset_path ../../../datasets/processed/vlm/flickr8k/flickr8k_train.json \
  --image_dir ../../../datasets/processed/vlm/flickr8k/images \
  --output_dir ../../../outputs/gemma3-4b-vlm-flickr8k \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --warmup_steps 100 \
  --logging_steps 10 \
  --save_steps 500 \
  --use_lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --use_8bit \
  --max_seq_length 2048 \
  --fp16 False \
  --bf16 True \
  --no-gradient_checkpointing \
  --optim adamw_torch \
  --report_to tensorboard