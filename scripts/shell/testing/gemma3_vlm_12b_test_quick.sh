#!/bin/bash

# Quick test script for Gemma-3-12B VLM - runs for just 20 steps to verify setup
# This avoids checkpoint saving issues during testing

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
mkdir -p ../../../outputs/gemma3-12b-vlm-quick-test

# Run training with DeepSpeed ZeRO-3 - just 20 steps
deepspeed ../../training/gemma3_vlm_training_production.py \
  --deepspeed ../../../configs/training/ds_config_zero3.json \
  --model_name_or_path google/gemma-3-12b-it \
  --dataset_path ../../../datasets/test_dataset/vlm/flickr8k/flickr8k_test_data.json \
  --image_dir ../../../datasets/test_dataset/vlm/flickr8k/images \
  --output_dir ../../../outputs/gemma3-12b-vlm-quick-test \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 20 \
  --learning_rate 2e-5 \
  --warmup_steps 2 \
  --logging_steps 1 \
  --save_strategy no \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_8bit \
  --max_seq_length 512 \
  --fp16 False \
  --bf16 True \
  --gradient_checkpointing \
  --report_to none