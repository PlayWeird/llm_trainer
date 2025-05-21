# Compatibility Notes for Gemma Training

This document outlines compatibility issues and solutions for training Gemma models with PyTorch 2.2.0 and Transformers 4.53.0.

## Issues Identified

1. **Missing `torch.get_default_device()` function**:
   - Transformers 4.53.0 expects this function but it's missing in PyTorch 2.2.0
   - Solution: Patch the torch module by adding the function

2. **PyTorch Vulnerability Restriction**:
   - Newer transformers require PyTorch 2.6+ for security reasons when loading models
   - Solution: Use `use_safetensors=True` to work around this restriction

3. **Device Mapping Issues**:
   - Auto device mapping can cause device mismatches
   - Solution: Use explicit device mapping strategies

## Patched Scripts

1. **`test_gemma_inference.py`**: 
   - Tests loading and running inference with Gemma models
   - Includes fixes for all identified issues
   - Supports 4-bit quantization

2. **`test_gemma_training.py`**:
   - Tests basic model loading, LoRA setup, and dataset handling
   - Validates the compatibility fixes work for training setup

3. **`train_gemma3_27b.py`**:
   - Full training script with all compatibility fixes
   - Includes DeepSpeed integration

## How to Use

### Testing Inference

```bash
python scripts/inference/test_gemma_inference.py --use_4bit
```

### Testing Training Setup

```bash
python scripts/training/test_gemma_training.py
```

### Full Training

```bash
deepspeed scripts/training/train_gemma3_27b.py \
  --deepspeed configs/training/ds_config_zero3.json \
  --model_name_or_path google/gemma-2-2b \
  --train_file datasets/my_dataset.json \
  --output_dir outputs/gemma-2-2b-finetuned \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --warmup_steps 500 \
  --logging_steps 10 \
  --save_steps 500 \
  --use_lora True \
  --use_4bit True
```

## Key Patches

The main patch that enables compatibility is:

```python
# PATCH: Add get_default_device to torch module if it doesn't exist
if not hasattr(torch, 'get_default_device'):
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
    # Add the function to the torch module
    torch.get_default_device = get_default_device
```

Combined with using safetensors and explicit device mapping, this allows Gemma models to be trained with the current environment configuration.