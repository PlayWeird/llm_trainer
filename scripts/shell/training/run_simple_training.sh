#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate llm_trainer_env

# Set visible GPUs to just one
export CUDA_VISIBLE_DEVICES=0

# Run training with simplified setup
python ../../scripts/training/simple_lora_train.py \
  --model_name google/gemma-2-2b \
  --train_file ../../datasets/test_dataset/test_data.json \
  --output_dir ../../outputs/gemma-2-2b-finetuned \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --learning_rate 2e-5 \
  --use_4bit \
  --max_seq_length 512 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --save_steps 5