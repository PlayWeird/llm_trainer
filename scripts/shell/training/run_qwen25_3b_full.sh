#!/bin/bash

# Training script for Qwen2.5-3B-Instruct
# Excellent performance-to-size ratio, fits on single GPU

# Exit on error
set -e

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Single GPU
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info

# Activate conda environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate llm_trainer_env

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/train_qwen25_full.py"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen25-3b-full-finetuned"
DATASET_PATH="$PROJECT_ROOT/datasets/test_dataset/llm/dolly_test_data.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log system info
echo "Starting Qwen2.5-3B-Instruct training..."
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Output directory: $OUTPUT_DIR"
nvidia-smi

# Run training - 3B model can handle larger batches
python "$TRAINING_SCRIPT" \
    --model_name_or_path "Qwen/Qwen2.5-3B-Instruct" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 3e-5 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --bf16 True \
    --gradient_checkpointing True \
    --optim "paged_adamw_32bit" \
    --max_seq_length 2048 \
    --use_lora True \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --report_to "tensorboard" \
    --trust_remote_code True

echo "Training completed!"