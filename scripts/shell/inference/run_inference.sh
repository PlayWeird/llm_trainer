#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate gemma3_env

# Run inference with Gemma
echo "=== Running inference with Gemma 2-2B model ==="
python ../../scripts/inference/gemma_inference_test.py --use_4bit --prompt "Explain the concept of fine-tuning large language models in simple terms."

echo ""
echo "All tests completed successfully!"