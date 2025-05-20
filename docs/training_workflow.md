# Gemma 3 27B Training Workflow

This document outlines the end-to-end workflow for training Gemma 3 27B models using this project.

## Overview

The training workflow consists of the following steps:
1. Data preparation
2. Model configuration
3. Training setup
4. Execution
5. Evaluation and visualization

## 1. Data Preparation

### Dataset Requirements

For general text fine-tuning:
- High-quality text in instruction format
- Prompt/response pairs
- Recommended 10,000+ examples

For multimodal training:
- Image/text pairs
- Image descriptions or captions
- QA pairs about images

### Processing Pipeline

```bash
# Create dataset directory
mkdir -p datasets/gemma_instruction_dataset

# Run preprocessing script
python utils/data_preprocessing.py \
  --input_file your_raw_data.json \
  --output_file datasets/gemma_instruction_dataset/processed_data.json \
  --format alpaca
```

### Dataset Format

Example dataset JSON format:
```json
[
  {
    "instruction": "Explain the concept of neural networks.",
    "input": "",
    "output": "Neural networks are computational models inspired by the human brain..."
  },
  {
    "instruction": "Describe this image.",
    "input": "",
    "output": "The image shows a sunset over mountains with vibrant orange and purple colors...",
    "image": "path/to/image.jpg"
  }
]
```

## 2. Model Configuration

### Configuration Files

Create a model configuration file in `configs/model/`:

```json
{
  "model_name_or_path": "google/gemma-3-27b-it",
  "use_lora": true,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "use_4bit": true,
  "use_8bit": false
}
```

### DeepSpeed Configuration

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": 16,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": 1,
  "wall_clock_breakdown": false
}
```

## 3. Training Setup

### Launch Script

Create a training launch script:

```bash
#!/bin/bash
# save as run_training.sh

export CUDA_VISIBLE_DEVICES=0,1,2

deepspeed scripts/training/train_gemma3_27b.py \
  --deepspeed configs/training/ds_config_zero3.json \
  --model_name_or_path google/gemma-3-27b-it \
  --dataset_name datasets/gemma_instruction_dataset \
  --output_dir outputs/gemma-3-27b-finetuned \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_steps 500 \
  --logging_steps 10 \
  --save_steps 500 \
  --use_lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --use_4bit True
```

### Parameter Explanation

- `--model_name_or_path`: Hugging Face model ID or local path
- `--dataset_name`: Path to the dataset
- `--output_dir`: Directory to save model checkpoints
- `--per_device_train_batch_size`: Batch size per GPU
- `--gradient_accumulation_steps`: Number of steps to accumulate gradients
- `--num_train_epochs`: Number of training epochs
- `--learning_rate`: Learning rate for optimizer
- `--warmup_steps`: Number of warmup steps for learning rate scheduler
- `--logging_steps`: Log metrics every X steps
- `--save_steps`: Save model every X steps
- `--use_lora`: Whether to use LoRA for fine-tuning
- `--lora_r`: LoRA rank
- `--lora_alpha`: LoRA alpha parameter
- `--lora_dropout`: LoRA dropout probability
- `--use_4bit`: Whether to use 4-bit quantization

## 4. Execution

### Starting Training

```bash
# Make the script executable
chmod +x run_training.sh

# Run the training
./run_training.sh
```

### Monitoring

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f outputs/gemma-3-27b-finetuned/logs/training_log.txt

# Start TensorBoard
tensorboard --logdir outputs/gemma-3-27b-finetuned/logs
```

### Checkpointing

By default, model checkpoints are saved to:
- `outputs/gemma-3-27b-finetuned/checkpoint-{step}`

For LoRA models, the adapter weights are saved, not the full model.

## 5. Evaluation and Visualization

### Evaluation Script

```bash
python scripts/evaluation/evaluate_model.py \
  --model_path outputs/gemma-3-27b-finetuned/checkpoint-best \
  --test_file datasets/gemma_instruction_dataset/test.json \
  --output_file evaluation/results/evaluation_results.json \
  --batch_size 4
```

### Visualization

```bash
python visualizations/training/plot_training_metrics.py \
  --tensorboard_dir outputs/gemma-3-27b-finetuned/logs \
  --output_dir visualizations/training/plots
```

## Common Issues and Solutions

### Out of Memory Errors

If you encounter OOM errors:
1. Reduce batch size
2. Increase gradient accumulation steps
3. Use more aggressive quantization
4. Reduce context length

### Slow Training

To improve training speed:
1. Install flash-attention
2. Optimize DeepSpeed configuration
3. Ensure NVLink is properly configured
4. Use local datasets instead of loading from Hugging Face

### Training Instability

If you experience instability:
1. Lower learning rate
2. Increase warmup steps
3. Clip gradients (set in DeepSpeed config)
4. Check for bad samples in dataset