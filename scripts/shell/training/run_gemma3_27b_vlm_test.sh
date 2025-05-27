#!/bin/bash

# Test script for Gemma-3-27B VLM - minimal configuration with extreme memory optimization
# WARNING: Even with these optimizations, may still run out of memory on 3x RTX 3090

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2  # Use all 3 GPUs
export NCCL_DEBUG=INFO  # Helpful for debugging NCCL issues
export NCCL_P2P_DISABLE=1  # Can help with certain multi-GPU configuration
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer parallelism warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1  # Reduce noise from warnings

# Aggressive memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:64
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Create output directory if it doesn't exist
mkdir -p ../../../outputs/gemma3-27b-vlm-test

# Run training with DeepSpeed ZeRO-3 - extreme memory optimization
deepspeed ../../training/train_gemma3_vlm.py \
  --deepspeed ../../../configs/training/ds_config_zero3_memory_opt.json \
  --model_name_or_path google/gemma-3-27b-it \
  --dataset_path ../../../datasets/test_dataset/vlm/flickr8k/flickr8k_test_data.json \
  --image_dir ../../../datasets/test_dataset/vlm/flickr8k/images \
  --output_dir ../../../outputs/gemma3-27b-vlm-test \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --warmup_steps 2 \
  --logging_steps 1 \
  --save_steps 10 \
  --use_lora True \
  --lora_r 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  --use_8bit True \
  --max_seq_length 256 \
  --fp16 False \
  --bf16 True \
  --gradient_checkpointing True \
  --optim adamw_8bit \
  --report_to none