#!/bin/bash

# Test training script for Qwen2.5-14B-Instruct
# Optimized for 3x RTX 3090 GPUs with proven configuration

# Exit on error
set -e

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2
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
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen25-14b-test-$(date +%Y%m%d_%H%M%S)"
DATASET_PATH="$PROJECT_ROOT/datasets/test_dataset/llm/dolly_test_data.json"
DEEPSPEED_CONFIG="$PROJECT_ROOT/configs/training/ds_config_zero3.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log system info
echo "=========================================="
echo "Starting Qwen2.5-14B-Instruct TEST training"
echo "=========================================="
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Output directory: $OUTPUT_DIR"
echo "Dataset: $DATASET_PATH"
echo "DeepSpeed config: $DEEPSPEED_CONFIG"
echo "=========================================="
nvidia-smi
echo "=========================================="

# Run with DeepSpeed - using proven configuration
deepspeed --num_gpus=3 "$TRAINING_SCRIPT" \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --max_steps 50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps 25 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --logging_steps 5 \
    --bf16 True \
    --gradient_checkpointing True \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --optim "paged_adamw_32bit" \
    --max_seq_length 1024 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --use_4bit True \
    --report_to "tensorboard" \
    --trust_remote_code True \
    --ddp_find_unused_parameters False \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 0

echo "=========================================="
echo "TEST training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="

# Display final GPU status
nvidia-smi

# Display training results if available
if [ -f "$OUTPUT_DIR/train_results.json" ]; then
    echo "=========================================="
    echo "Training Results:"
    cat "$OUTPUT_DIR/train_results.json"
    echo "=========================================="
fi