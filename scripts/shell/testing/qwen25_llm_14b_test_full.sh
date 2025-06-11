#!/bin/bash

# Full test training script for Qwen2.5-14B-Instruct
# Single GPU training with model saving

# Exit on error
set -e

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=info

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/qwen25_llm_training_production.py"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen25-14b-full-test-$(date +%Y%m%d_%H%M%S)"
DATASET_PATH="$PROJECT_ROOT/datasets/test_dataset/llm/dolly_test_data.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log system info
echo "=========================================="
echo "Starting Qwen2.5-14B Full Test Training"
echo "=========================================="
echo "CUDA device: $CUDA_VISIBLE_DEVICES"
echo "Output directory: $OUTPUT_DIR"
echo "Dataset: $DATASET_PATH"
echo "=========================================="
nvidia-smi
echo "=========================================="

# Run training with model saving
python "$TRAINING_SCRIPT" \
    --model_name_or_path "Qwen/Qwen2.5-14B-Instruct" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --max_steps 20 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --do_eval False \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 5 \
    --logging_steps 2 \
    --logging_dir "$OUTPUT_DIR/logs" \
    --bf16 True \
    --gradient_checkpointing True \
    --optim "adamw_torch" \
    --max_seq_length 512 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --use_4bit True \
    --report_to "tensorboard" \
    --trust_remote_code True \
    --preprocessing_num_workers 2 \
    --ddp_find_unused_parameters False \
    --overwrite_output_dir True

echo "=========================================="
echo "Training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="

# Display final GPU status
nvidia-smi

# Check if model was saved
if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
    echo "✅ Model checkpoint saved successfully!"
    ls -la "$OUTPUT_DIR"
else
    echo "❌ Model checkpoint not found!"
fi

# Display training results if available
if [ -f "$OUTPUT_DIR/train_results.json" ]; then
    echo "=========================================="
    echo "Training Results:"
    cat "$OUTPUT_DIR/train_results.json"
    echo "=========================================="
fi

# Check TensorBoard logs
if [ -d "$OUTPUT_DIR/logs" ]; then
    echo "TensorBoard logs available at: $OUTPUT_DIR/logs"
    echo "Run: tensorboard --logdir $OUTPUT_DIR/logs"
fi