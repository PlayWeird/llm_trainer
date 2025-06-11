#!/bin/bash

# LLaVA-NeXT 13B VLM Production Training Script
# Multi-GPU training using DeepSpeed ZeRO-3 for memory efficiency

set -e

# Configuration
MODEL_NAME="llava-hf/llava-v1.6-vicuna-13b-hf"
DATASET_PATH="/home/user/llm_trainer/datasets/test_dataset/vlm/flickr8k"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/home/user/llm_trainer/outputs/llava-next-13b-${TIMESTAMP}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/llava_next_vlm_training_production.py"
DEEPSPEED_CONFIG="$PROJECT_ROOT/configs/training/ds_config_zero3_vlm.json"

echo "========================================"
echo "LLaVA-NeXT 13B VLM Production Training"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "DeepSpeed config: $DEEPSPEED_CONFIG"
echo "Training script: $TRAINING_SCRIPT"
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

# Verify dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset directory not found: $DATASET_PATH"
    exit 1
fi

if [ ! -f "$DATASET_PATH/flickr8k_test_data.json" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH/flickr8k_test_data.json"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
echo ""

# Verify DeepSpeed config exists
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Error: DeepSpeed config not found: $DEEPSPEED_CONFIG"
    exit 1
fi

# Log configuration
echo "Training Configuration:"
echo "- Model: $MODEL_NAME"
echo "- Dataset: $DATASET_PATH"
echo "- Output: $OUTPUT_DIR"
echo "- Max training samples: 1000"
echo "- Batch size per device: 1"
echo "- Gradient accumulation: 16"
echo "- Learning rate: 2e-5"
echo "- Epochs: 3"
echo "- Using 4-bit quantization: Yes"
echo "- Using LoRA: Yes"
echo "- Using DeepSpeed ZeRO-3: Yes"
echo ""

# Start training
echo "Starting LLaVA-NeXT 13B training with DeepSpeed..."

deepspeed "$TRAINING_SCRIPT" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --max_seq_length 2048 \
    --max_train_samples 1000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 2 \
    --use_4bit \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05

echo ""
echo "========================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "========================================"

# Display final GPU memory usage
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader,nounits