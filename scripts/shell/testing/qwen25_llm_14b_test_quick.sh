#!/bin/bash

# Quick test script for Qwen2.5-14B-Instruct
# Minimal configuration for testing

# Exit on error
set -e

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llm_trainer_env

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/qwen25_llm_training_production.py"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen25-14b-quick-test"
DATASET_PATH="$PROJECT_ROOT/datasets/test_dataset/llm/dolly_test_data.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log system info
echo "Starting Qwen2.5-14B-Instruct quick test..."
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Output directory: $OUTPUT_DIR"

# Run minimal training with DeepSpeed
deepspeed --num_gpus=3 "$TRAINING_SCRIPT" \
    --deepspeed "$PROJECT_ROOT/configs/training/ds_config_zero3.json" \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --warmup_steps 0 \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing True \
    --optim "paged_adamw_32bit" \
    --max_seq_length 128 \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --report_to "none" \
    --trust_remote_code True \
    --preprocessing_num_workers 1

echo "Quick test completed!"