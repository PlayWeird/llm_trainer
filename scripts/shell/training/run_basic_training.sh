#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set visible GPUs to just one
export CUDA_VISIBLE_DEVICES=0

# Run basic training
python ../../scripts/training/basic_lora_train.py \
  --model_name google/gemma-2-2b \
  --train_file ../../datasets/test_dataset/test_data.json \
  --output_dir ../../outputs/gemma-2-2b-basic-finetuned \
  --num_train_epochs 3 \
  --batch_size 1 \
  --learning_rate 2e-5 \
  --use_4bit \
  --max_seq_length 512 \
  --gradient_accumulation_steps 4