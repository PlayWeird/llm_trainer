#!/bin/bash

# Multi-GPU DeepSpeed training script for Qwen2.5-32B-Instruct
# Uses DeepSpeed ZeRO-3 for distributed training

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
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen25-32b-deepspeed-finetuned"
DATASET_PATH="$PROJECT_ROOT/datasets/test_dataset/llm/dolly_test_data.json"
DEEPSPEED_CONFIG="$PROJECT_ROOT/configs/training/ds_config_zero3.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log system info
echo "Starting Qwen2.5-32B-Instruct DeepSpeed training..."
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Output directory: $OUTPUT_DIR"
nvidia-smi

# Run with DeepSpeed
deepspeed --num_gpus=3 "$TRAINING_SCRIPT" \
    --model_name_or_path "Qwen/Qwen2.5-32B-Instruct" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --bf16 True \
    --gradient_checkpointing True \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --optim "paged_adamw_32bit" \
    --max_seq_length 2048 \
    --use_lora True \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --report_to "tensorboard" \
    --trust_remote_code True \
    --ddp_find_unused_parameters False

echo "DeepSpeed training completed!"