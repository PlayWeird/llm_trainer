#!/bin/bash

# LLaVA-NeXT 13B VLM Quick Test Script
# Fast test with minimal samples for validation

set -e

# Configuration
MODEL_NAME="llava-hf/llava-v1.6-vicuna-13b-hf"
DATASET_PATH="/home/user/llm_trainer/datasets/test_dataset/vlm/flickr8k"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/home/user/llm_trainer/outputs/llava-next-13b-quick-test-${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/llava_next_vlm_training_production.py"
DEEPSPEED_CONFIG="$PROJECT_ROOT/configs/training/ds_config_zero3_vlm.json"

echo "========================================"
echo "LLaVA-NeXT 13B VLM Quick Test"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Change to project directory
cd "$PROJECT_ROOT"

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_trainer_env

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo ""

# Quick test configuration
echo "Quick Test Configuration:"
echo "- Model: $MODEL_NAME"
echo "- Max training samples: 50"
echo "- Batch size per device: 1"
echo "- Gradient accumulation: 4"
echo "- Learning rate: 2e-5"
echo "- Max steps: 10"
echo "- Using 4-bit quantization: Yes"
echo "- Using LoRA: Yes"
echo ""

# Start quick test
echo "Starting LLaVA-NeXT 13B quick test..."

deepspeed "$TRAINING_SCRIPT" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length 1024 \
    --max_train_samples 50 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 5 \
    --logging_steps 2 \
    --save_steps 10 \
    --save_total_limit 1 \
    --use_4bit \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1

echo ""
echo "========================================"
echo "Quick test completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"