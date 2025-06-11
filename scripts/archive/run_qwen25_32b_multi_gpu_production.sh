#!/bin/bash

# Production Multi-GPU training script for Qwen2.5-32B
# Uses the proven VLM-style multi-GPU setup with longer training

# Exit on error
set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set environment variables for multi-GPU training (PROVEN WORKING)
export CUDA_VISIBLE_DEVICES=0,1,2  # Use all 3 GPUs
export NCCL_DEBUG=INFO  # Helpful for debugging NCCL issues
export NCCL_P2P_DISABLE=1  # Can help with certain multi-GPU configuration
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer parallelism warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1  # Reduce noise from warnings

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAINING_SCRIPT="$PROJECT_ROOT/scripts/training/qwen25_llm_training_production.py"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen25-32b-production-$(date +%Y%m%d_%H%M%S)"
DATASET_PATH="$PROJECT_ROOT/datasets/test_dataset/llm/dolly_test_data.json"
DEEPSPEED_CONFIG="$PROJECT_ROOT/configs/training/ds_config_zero3_memory_opt.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Starting Qwen2.5-32B PRODUCTION Multi-GPU Training"
echo "=========================================="
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "Output directory: $OUTPUT_DIR"
echo "Using memory-optimized DeepSpeed ZeRO-3"
echo "=========================================="
nvidia-smi
echo "=========================================="

# Production training with longer run
deepspeed "$TRAINING_SCRIPT" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_steps 100 \
  --logging_steps 10 \
  --save_steps 500 \
  --save_total_limit 3 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --use_lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_4bit True \
  --max_seq_length 1024 \
  --fp16 False \
  --bf16 True \
  --gradient_checkpointing True \
  --report_to tensorboard \
  --trust_remote_code True \
  --do_eval False \
  --overwrite_output_dir True \
  --logging_dir "$OUTPUT_DIR/logs" \
  --dataloader_num_workers 0 \
  --preprocessing_num_workers 4

echo "=========================================="
echo "PRODUCTION 32B training completed!"
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="

# Display final GPU status
nvidia-smi

# Check if model was saved
if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
    echo "✅ Model checkpoint saved successfully!"
    echo "Model size:"
    ls -lh "$OUTPUT_DIR/adapter_model.safetensors"
else
    echo "❌ Model checkpoint not found!"
fi

# Show TensorBoard command
echo "=========================================="
echo "To view training progress:"
echo "tensorboard --logdir $OUTPUT_DIR/logs"
echo "========================================"