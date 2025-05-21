#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate gemma3_env

# Set environment variables to help with distributed operations
export CUDA_VISIBLE_DEVICES=0,1,2  # Use all 3 GPUs
export NCCL_DEBUG=INFO  # Helpful for debugging NCCL issues
export NCCL_P2P_DISABLE=1  # Can help with certain multi-GPU configurations
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer parallelism warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1  # Reduce noise from warnings

# Run the script with 4-bit quantization to fit the model on available GPUs
python ../../scripts/load_gemma_27b.py \
  --model_name google/gemma-3-27b-it \
  --use_4bit \
  --prompt "Explain how to distribute a large language model across multiple GPUs, and what are the key considerations for memory optimization." \
  --max_new_tokens 200