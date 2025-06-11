#!/bin/bash

# LLaVA-NeXT 13B VLM Loading Test Script
# Tests loading and basic inference with LLaVA-NeXT 13B vision-language model

set -e

# Configuration
MODEL_NAME="llava-hf/llava-v1.6-vicuna-13b-hf"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
INFERENCE_SCRIPT="$PROJECT_ROOT/scripts/inference/llava_next_vlm_13b_test_loading.py"

echo "========================================"
echo "LLaVA-NeXT 13B VLM Loading Test"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Script: $INFERENCE_SCRIPT"
echo "Project root: $PROJECT_ROOT"
echo "========================================"

# Environment setup
export CUDA_VISIBLE_DEVICES=0,1,2
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Change to project directory
cd "$PROJECT_ROOT"

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_trainer_env

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo ""

# Test different configurations
echo "Testing LLaVA-NeXT 13B model loading..."

echo ""
echo "Test 1: Default configuration (no quantization)"
python "$INFERENCE_SCRIPT" \
    --model_name="$MODEL_NAME" || echo "Default config failed, trying quantized versions..."

echo ""
echo "Test 2: 8-bit quantization"
python "$INFERENCE_SCRIPT" \
    --model_name="$MODEL_NAME" \
    --use_8bit || echo "8-bit config failed, trying 4-bit..."

echo ""
echo "Test 3: 4-bit quantization"
python "$INFERENCE_SCRIPT" \
    --model_name="$MODEL_NAME" \
    --use_4bit || echo "4-bit config failed"

echo ""
echo "Test 4: 4-bit quantization with Flash Attention (if available)"
python "$INFERENCE_SCRIPT" \
    --model_name="$MODEL_NAME" \
    --use_4bit \
    --use_flash_attn || echo "4-bit + Flash Attention config failed"

echo ""
echo "========================================"
echo "LLaVA-NeXT 13B VLM loading test completed"
echo "========================================"