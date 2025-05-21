#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set environment variables to help with distributed operations
export CUDA_VISIBLE_DEVICES=0,1,2  # Use all 3 GPUs
export NCCL_DEBUG=INFO  # Helpful for debugging NCCL issues
export NCCL_P2P_DISABLE=1  # Can help with certain multi-GPU configurations
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer parallelism warnings

# Run the script with the gpt2-xl model, without quantization to force distribution
python ../../scripts/test_smaller_model.py \
  --model_name "gpt2-xl" \
  --prompt "In a world where AI models are distributed across multiple GPUs, the researchers discovered that "