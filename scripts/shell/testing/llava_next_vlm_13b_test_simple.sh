#!/bin/bash

# Simple LLaVA-NeXT 13B test without DeepSpeed for debugging
# This will help isolate the issue from the multi-GPU DeepSpeed setup

set -e

# Configuration
MODEL_NAME="llava-hf/llava-v1.6-vicuna-13b-hf"
DATASET_PATH="/home/user/llm_trainer/datasets/test_dataset/vlm/flickr8k"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/home/user/llm_trainer/outputs/llava-next-13b-simple-test-${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/llava_next_vlm_training_production.py"

echo "========================================"
echo "LLaVA-NeXT 13B Simple Test (No DeepSpeed)"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Environment setup - single GPU only for debugging
export CUDA_VISIBLE_DEVICES=0
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

# Simple test configuration
echo "Simple Test Configuration:"
echo "- Model: $MODEL_NAME"
echo "- Max training samples: 10"
echo "- Batch size per device: 1"
echo "- Gradient accumulation: 2"
echo "- Learning rate: 2e-5"
echo "- Max steps: 5"
echo "- Using 4-bit quantization: Yes"
echo "- Using LoRA: Yes"
echo "- NO DeepSpeed (single GPU)"
echo ""

# Start simple test (no DeepSpeed)
echo "Starting LLaVA-NeXT 13B simple test..."

python "$TRAINING_SCRIPT" \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length 512 \
    --max_train_samples 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 1 \
    --save_steps 5 \
    --save_total_limit 1 \
    --use_4bit \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1

echo ""
echo "========================================"
echo "Simple test completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"