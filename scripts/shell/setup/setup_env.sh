#!/bin/bash

# Create the conda environment from the yml file
conda env create -f environment.yml

# Activate the environment
echo "To activate the environment, run: conda activate llm_trainer_env"

# Install additional dependencies
echo "After activating, you might need to install additional packages:"
echo "pip install -r requirements.txt"

# For flash attention, we need to do a separate install with the right CUDA version
echo "For flash attention: pip install flash-attn --no-build-isolation"