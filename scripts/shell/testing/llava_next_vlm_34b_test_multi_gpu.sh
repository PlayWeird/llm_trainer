#!/bin/bash

# LLaVA-NeXT 34B VLM Multi-GPU Test Script
# Tests the largest model that should fit on 3x RTX 3090 with aggressive optimization

set -e

# Configuration
MODEL_NAME="llava-hf/llava-v1.6-34b-hf"
DATASET_PATH="/home/user/llm_trainer/datasets/test_dataset/vlm/flickr8k"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/home/user/llm_trainer/outputs/llava-next-34b-multi-gpu-test-${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/llava_next_vlm_training_production.py"
DEEPSPEED_CONFIG="$PROJECT_ROOT/configs/training/ds_config_zero3_vlm.json"

echo "========================================"
echo "LLaVA-NeXT 34B Multi-GPU Test"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Target: All 3 GPUs should be heavily utilized"
echo "========================================"

# Environment setup - all GPUs
export CUDA_VISIBLE_DEVICES=0,1,2
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd "$PROJECT_ROOT"

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_trainer_env

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check initial GPU state
echo "Initial GPU memory usage:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo ""

# Test configuration optimized for 34B model
echo "34B Model Test Configuration:"
echo "- Model: $MODEL_NAME (34B parameters)"
echo "- Max training samples: 20"
echo "- Batch size per device: 1"
echo "- Gradient accumulation: 8"
echo "- Sequence length: 768 (reduced for memory)"
echo "- Learning rate: 1e-5"
echo "- Epochs: 1"
echo "- Using 4-bit quantization: Yes"
echo "- Using LoRA: Yes (aggressive settings)"
echo "- Using DeepSpeed ZeRO-3: Yes"
echo ""

echo "🚀 Starting LLaVA-NeXT 34B multi-GPU test..."
echo "📊 Monitor all 3 GPUs - you should see high utilization across all!"
echo ""

# Start 34B model test
deepspeed "$TRAINING_SCRIPT" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length 768 \
    --max_train_samples 20 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --warmup_steps 3 \
    --logging_steps 1 \
    --save_steps 10 \
    --save_total_limit 1 \
    --use_4bit \
    --use_lora \
    --lora_r 4 \
    --lora_alpha 8 \
    --lora_dropout 0.1

echo ""
echo "========================================"
echo "34B Multi-GPU test completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"

# Display final GPU memory usage
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader,nounits