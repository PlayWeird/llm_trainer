#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set environment variables to help with distributed operations
export CUDA_VISIBLE_DEVICES=0,1,2  # Use all 3 GPUs
export NCCL_DEBUG=INFO  # Helpful for debugging NCCL issues
export NCCL_P2P_DISABLE=1  # Can help with certain multi-GPU configurations
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer parallelism warnings

# Run compatibility test
python ../../scripts/test_multi_gpu_compatibility.py