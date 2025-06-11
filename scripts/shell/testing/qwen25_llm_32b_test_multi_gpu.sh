#!/bin/bash

# Multi-GPU training script for Qwen2.5-32B using working VLM-style setup
# This uses the exact same configuration that works for VLM training

# Exit on error
set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set environment variables for multi-GPU training (EXACT SAME AS WORKING VLM)
export CUDA_VISIBLE_DEVICES=0,1,2  # Use all 3 GPUs
export NCCL_DEBUG=INFO  # Helpful for debugging NCCL issues
export NCCL_P2P_DISABLE=1  # Can help with certain multi-GPU configuration
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer parallelism warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1  # Reduce noise from warnings

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/qwen25_llm_training_production.py"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen25-32b-multi-gpu-working"
DATASET_PATH="$PROJECT_ROOT/datasets/test_dataset/llm/dolly_test_data.json"
DEEPSPEED_CONFIG="$PROJECT_ROOT/configs/training/ds_config_zero3_memory_opt.json"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Starting Qwen2.5-32B Multi-GPU Training (Working VLM-style setup)"
echo "=========================================="
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Output directory: $OUTPUT_DIR"
echo "Using memory-optimized DeepSpeed config for 32B model"
echo "=========================================="
nvidia-smi
echo "=========================================="

# Run training with DeepSpeed (EXACT SAME COMMAND STRUCTURE AS WORKING VLM)
deepspeed "$TRAINING_SCRIPT" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_steps 20 \
  --learning_rate 2e-5 \
  --warmup_steps 5 \
  --logging_steps 2 \
  --save_steps 10 \
  --save_total_limit 2 \
  --use_lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_4bit True \
  --max_seq_length 512 \
  --fp16 False \
  --bf16 True \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --trust_remote_code True \
  --do_eval False \
  --overwrite_output_dir True

echo "=========================================="
echo "Multi-GPU 32B training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="

# Display final GPU status
nvidia-smi

# Check if model was saved
if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
    echo "✅ Model checkpoint saved successfully!"
    ls -la "$OUTPUT_DIR"
else
    echo "❌ Model checkpoint not found!"
fi