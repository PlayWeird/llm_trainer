#!/bin/bash

# Test training script for Qwen2.5-14B using torchrun instead of deepspeed
# This avoids potential NCCL issues with deepspeed launcher

# Exit on error
set -e

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/train_qwen25_full.py"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen25-14b-torchrun-test"
DATASET_PATH="$PROJECT_ROOT/datasets/test_dataset/llm/dolly_test_data.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log system info
echo "=========================================="
echo "Starting Qwen2.5-14B TEST (torchrun)"
echo "=========================================="
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
nvidia-smi
echo "=========================================="

# Run with torchrun - single GPU first to test
torchrun --standalone --nproc_per_node=1 "$TRAINING_SCRIPT" \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --max_steps 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --do_eval False \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing True \
    --optim "adamw_torch" \
    --max_seq_length 512 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --use_4bit True \
    --report_to "none" \
    --trust_remote_code True \
    --preprocessing_num_workers 1

echo "=========================================="
echo "TEST completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="