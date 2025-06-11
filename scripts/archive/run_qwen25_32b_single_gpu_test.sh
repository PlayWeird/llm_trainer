#!/bin/bash

# Single GPU test training script for Qwen2.5-32B-Instruct
# Uses aggressive memory optimization without DeepSpeed for initial testing

# Exit on error
set -e

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use only first GPU for initial test
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/train_qwen25_full.py"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen25-32b-single-gpu-test"
DATASET_PATH="$PROJECT_ROOT/datasets/test_dataset/llm/dolly_test_data.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log system info
echo "=========================================="
echo "Starting Qwen2.5-32B-Instruct SINGLE GPU TEST"
echo "=========================================="
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "Output directory: $OUTPUT_DIR"
echo "Dataset: $DATASET_PATH"
echo "=========================================="
nvidia-smi
echo "=========================================="

# Run without DeepSpeed - single GPU with aggressive optimization
python "$TRAINING_SCRIPT" \
    --model_name_or_path "Qwen/Qwen2.5-32B-Instruct" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --do_eval False \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing True \
    --optim "paged_adamw_8bit" \
    --max_seq_length 256 \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --use_4bit True \
    --report_to "none" \
    --trust_remote_code True \
    --preprocessing_num_workers 1 \
    --dataloader_num_workers 0

echo "=========================================="
echo "SINGLE GPU TEST completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="

# Display final GPU status
nvidia-smi