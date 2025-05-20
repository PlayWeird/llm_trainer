# Gemma 3 27B Training

This directory contains the code and configuration for training and fine-tuning the Gemma 3 27B model, a powerful vision-language model from Google.

## Model Information

- **Parameters**: 27 billion
- **Context Window**: 128k tokens
- **Capabilities**: Multimodal (text + images), multilingual
- **HuggingFace ID**: google/gemma-3-27b

## Hardware Requirements

Our setup utilizes:
- 3x NVIDIA RTX 3090 GPUs (24GB VRAM each)
- DeepSpeed ZeRO-3 for distributed training
- 4-bit quantization for memory efficiency

## Training Options

1. **Full Fine-tuning**: Requires substantial GPU memory and compute
2. **LoRA Fine-tuning**: Memory-efficient adapter-based tuning
3. **QLoRA**: Quantized LoRA for maximum memory efficiency

## Usage

The training script supports the following configurations:

```bash
python scripts/training/train_gemma3_27b.py \
    --model_name_or_path google/gemma-3-27b-base \
    --dataset_name your_dataset \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --use_lora True \
    --output_dir ./outputs/gemma-3-27b-finetuned
```

## Distributed Training

To run on multiple GPUs with DeepSpeed:

```bash
deepspeed scripts/training/train_gemma3_27b.py \
    --deepspeed configs/training/ds_config_zero3.json \
    --model_name_or_path google/gemma-3-27b-base \
    --dataset_name your_dataset \
    --per_device_train_batch_size 1 \
    --use_lora True
```