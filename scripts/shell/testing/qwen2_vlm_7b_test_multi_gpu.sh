#!/bin/bash

# Quick test to confirm VLM multi-GPU training works
# This will run just a few steps to verify the setup

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set environment variables for multi-GPU training (same as working VLM)
export CUDA_VISIBLE_DEVICES=0,1,2  # Use all 3 GPUs
export NCCL_DEBUG=INFO  # Helpful for debugging NCCL issues
export NCCL_P2P_DISABLE=1  # Can help with certain multi-GPU configuration
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer parallelism warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1  # Reduce noise from warnings

# Create output directory
mkdir -p ../../../outputs/test_vlm_multi_gpu

echo "=========================================="
echo "Testing VLM Multi-GPU Setup"
echo "=========================================="
nvidia-smi
echo "=========================================="

# Run training with DeepSpeed - just 5 steps to test
deepspeed ../../training/qwen2_vlm_training_production.py \
  --deepspeed ../../../configs/training/ds_config_zero3.json \
  --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
  --dataset_path ../../../datasets/test_dataset/vlm/flickr8k/flickr8k_test_data.json \
  --image_dir ../../../datasets/test_dataset/vlm/flickr8k/images \
  --output_dir ../../../outputs/test_vlm_multi_gpu \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_steps 5 \
  --learning_rate 2e-5 \
  --warmup_steps 2 \
  --logging_steps 1 \
  --save_strategy no \
  --use_lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_4bit True \
  --max_seq_length 256 \
  --fp16 False \
  --bf16 True \
  --gradient_checkpointing True \
  --report_to none \
  --overwrite_output_dir True

echo "=========================================="
echo "VLM Multi-GPU test completed!"
echo "=========================================="