#!/bin/bash

# Training script for Idefics3-8B-Llama3 Vision-Language Model on Flickr8k dataset
# Optimized for 3x RTX 3090 GPUs

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
mkdir -p ../../../outputs/idefics3-8b-vlm-flickr8k

# Run training with DeepSpeed ZeRO-3
deepspeed ../../training/idefics3_vlm_training_production.py \
  --deepspeed ../../../configs/training/ds_config_zero3.json \
  --model_name_or_path HuggingFaceM4/Idefics3-8B-Llama3 \
  --dataset_path ../../../datasets/processed/vlm/flickr8k/flickr8k_train.json \
  --image_dir ../../../datasets/processed/vlm/flickr8k/images \
  --output_dir ../../../outputs/idefics3-8b-vlm-flickr8k \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_steps 100 \
  --logging_steps 10 \
  --save_steps 1000 \
  --save_total_limit 3 \
  --use_lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_4bit True \
  --max_seq_length 512 \
  --fp16 False \
  --bf16 True \
  --gradient_checkpointing True \
  --report_to tensorboard