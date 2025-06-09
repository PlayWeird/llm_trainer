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

### Basic Training (Single GPU)

For basic LoRA fine-tuning on a single GPU:

```bash
python scripts/training/train_gemma3_27b.py \
    --model_name_or_path google/gemma-3-27b-it \
    --train_file datasets/test_dataset/llm/dolly_formatted.json \
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
    --lora_dropout 0.05
```

### Memory-Efficient QLoRA Training

For maximum memory efficiency with 4-bit quantization:

```bash
python scripts/training/train_gemma3_27b.py \
    --model_name_or_path google/gemma-3-27b-it \
    --train_file datasets/test_dataset/llm/dolly_formatted.json \
    --output_dir outputs/gemma-3-27b-qlora \
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

Note: The 4-bit quantization settings (bnb_4bit_compute_dtype=bfloat16, bnb_4bit_use_double_quant=True) are configured in the script.

### Distributed Training with DeepSpeed

For multi-GPU training with DeepSpeed ZeRO-3 optimization:

```bash
deepspeed scripts/training/train_gemma3_27b.py \
    --deepspeed configs/training/ds_config_zero3.json \
    --model_name_or_path google/gemma-3-27b-it \
    --train_file datasets/test_dataset/llm/dolly_formatted.json \
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

### Vision-Language Model Training

For training with image-text datasets (e.g., Flickr8k):

```bash
python scripts/training/train_gemma3_vlm.py \
    --model_name_or_path google/gemma-3-27b-it \
    --dataset_path datasets/test_dataset/vlm/flickr8k/flickr8k_test_data.json \
    --image_dir datasets/test_dataset/vlm/flickr8k/images \
    --output_dir outputs/gemma-3-27b-vlm \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --save_steps 500 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32
```

Note: The VLM script requires the `--remove_unused_columns False` parameter in the training arguments to handle image data properly.

### Shell Script Execution

You can also use the provided shell scripts for convenient execution:

```bash
# For single GPU training
bash scripts/shell/training/run_training_single_gpu.sh

# For multi-GPU training with DeepSpeed
bash scripts/shell/training/run_gemma_27b.sh

# For VLM training
bash scripts/shell/training/run_vlm_training.sh
```

## Important Notes

### Dataset Format
- **LLM Training**: The training script expects a JSON file with a "text" field. Use the formatted dataset at `datasets/test_dataset/llm/dolly_formatted.json` or prepare your data accordingly.
- **VLM Training**: The VLM script expects a JSON file with "instruction", "output", and "image" fields. Image paths should be relative to the image_dir parameter.

### Common Issues
1. **DeepSpeed Configuration**: The DeepSpeed config uses bf16 by default. The training scripts have been updated to use bf16 instead of fp16 to avoid conflicts.
2. **Gradient Accumulation**: When using DeepSpeed, ensure the gradient_accumulation_steps matches the value in the DeepSpeed config (16 by default).
3. **Memory Management**: For the full 27B model, you'll need multiple GPUs. Use the 2B model (`google/gemma-2-2b-it`) for testing on limited hardware.
4. **VLM Image Paths**: Ensure image paths in your dataset are correct relative to the image_dir parameter.