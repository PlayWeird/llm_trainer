#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set visible GPUs to just one
export CUDA_VISIBLE_DEVICES=0

# Run training with simplified setup (no DeepSpeed)
python ../../scripts/training/gemma_training_test.py