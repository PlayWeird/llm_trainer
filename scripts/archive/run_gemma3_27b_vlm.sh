#!/bin/bash

# Training script for Gemma-3-27B Vision-Language Model on Flickr8k dataset
# WARNING: This model is very large and may struggle on 3x RTX 3090 GPUs
# Uses aggressive memory optimization techniques

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2  # Use all 3 GPUs
export NCCL_DEBUG=INFO  # Helpful for debugging NCCL issues
export NCCL_P2P_DISABLE=1  # Can help with certain multi-GPU configuration
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer parallelism warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1  # Reduce noise from warnings

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Create output directory if it doesn't exist
mkdir -p ../../../outputs/gemma3-27b-vlm-flickr8k

# Run training with DeepSpeed ZeRO-3 and aggressive memory optimization
deepspeed ../../training/gemma3_vlm_training_production.py \
  --deepspeed ../../../configs/training/ds_config_zero3_memory_opt.json \
  --model_name_or_path google/gemma-3-27b-it \
  --dataset_path ../../../datasets/processed/vlm/flickr8k/flickr8k_train.json \
  --image_dir ../../../datasets/processed/vlm/flickr8k/images \
  --output_dir ../../../outputs/gemma3-27b-vlm-flickr8k \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 1e-5 \
  --warmup_steps 100 \
  --logging_steps 10 \
  --save_steps 500 \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --use_8bit \
  --max_seq_length 512 \
  --fp16 False \
  --bf16 True \
  --gradient_checkpointing \
  --optim adamw_8bit \
  --report_to tensorboard