#!/bin/bash

# Ultra-lite test script for Gemma-3-27B VLM - absolute minimal configuration
# This uses the most extreme memory optimizations possible
# Only processes 10 examples for proof of concept

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set environment variables for multi-GPU training
export CUDA_VISIBLE_DEVICES=0,1,2  # Use all 3 GPUs
export NCCL_DEBUG=INFO  # Helpful for debugging NCCL issues
export NCCL_P2P_DISABLE=1  # Can help with certain multi-GPU configuration
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer parallelism warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1  # Reduce noise from warnings

# Maximum memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:32
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
export CUDA_LAUNCH_BLOCKING=1  # May help with memory fragmentation

# Create a minimal test dataset (just first 10 examples)
echo "Creating minimal test dataset..."
python -c "
import json
with open('../../../datasets/test_dataset/vlm/flickr8k/flickr8k_test_data.json', 'r') as f:
    data = json.load(f)
with open('../../../datasets/test_dataset/vlm/flickr8k/flickr8k_ultra_lite.json', 'w') as f:
    json.dump(data[:10], f, indent=2)
print(f'Created ultra-lite dataset with {len(data[:10])} examples')
"

# Create output directory if it doesn't exist
mkdir -p ../../../outputs/gemma3-27b-vlm-ultra-lite

# Run training with DeepSpeed ZeRO-3 - ultra minimal configuration
deepspeed ../../training/train_gemma3_vlm.py \
  --deepspeed ../../../configs/training/ds_config_zero3_memory_opt.json \
  --model_name_or_path google/gemma-3-27b-it \
  --dataset_path ../../../datasets/test_dataset/vlm/flickr8k/flickr8k_ultra_lite.json \
  --image_dir ../../../datasets/test_dataset/vlm/flickr8k/images \
  --output_dir ../../../outputs/gemma3-27b-vlm-ultra-lite \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --learning_rate 1e-5 \
  --warmup_steps 0 \
  --logging_steps 1 \
  --save_steps 10 \
  --save_total_limit 1 \
  --use_lora True \
  --lora_r 2 \
  --lora_alpha 4 \
  --lora_dropout 0.0 \
  --use_8bit True \
  --max_seq_length 128 \
  --fp16 False \
  --bf16 True \
  --gradient_checkpointing True \
  --optim adamw_8bit \
  --report_to none