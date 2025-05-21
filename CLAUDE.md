# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Guidelines

- **Always work strictly within the conda environment** - All code should run within the `llm_trainer_env` conda environment
- **Do not modify any system files** - Only modify files within the project directory
- **Keep all model artifacts in the designated directories** - Store datasets, models, and outputs in their respective directories

## Commands

### Environment Setup

```bash
# Create and activate conda environment (always work in this environment)
conda env create -f environment.yml
conda activate llm_trainer_env

# Install additional requirements
pip install -r requirements.txt

# Install flash-attention (optional, for faster training)
pip install flash-attn --no-build-isolation
```

### Verify GPU Setup

```bash
# Check NVIDIA driver and GPU detection
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# Test model loading with different configurations
python scripts/inference/test_gemma_inference.py --use_4bit
```

### Data Preprocessing

```bash
# Process instruction-tuning dataset
python utils/data_preprocessing.py \
  --input_file your_raw_data.json \
  --output_file datasets/gemma_instruction_dataset/processed_data.json \
  --format dolly
```

### Training

```bash
# Run training with DeepSpeed ZeRO-3
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

### Monitoring Training

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Start TensorBoard to monitor metrics
tensorboard --logdir outputs/gemma-3-27b-finetuned/logs
```

### Visualization

```bash
# Visualize training metrics
python visualizations/training/plot_training_metrics.py \
  --tensorboard_dir outputs/gemma-3-27b-finetuned/logs \
  --output_dir visualizations/training/plots
```

## Architecture Overview

This project is designed for training and fine-tuning large language models (LLMs) and vision-language models (VLMs) with a current focus on Gemma 3 27B.

### Key Components

1. **Training Pipeline**: Uses DeepSpeed ZeRO-3 for distributed training across multiple GPUs with memory optimization.

2. **Memory Management**:
   - 4-bit quantization via BitsAndBytes
   - Parameter-Efficient Fine-Tuning (PEFT) with LoRA
   - Gradient accumulation to increase effective batch size

3. **Data Processing**:
   - Support for instruction-tuning formats
   - Multimodal dataset handling for image-text pairs
   - Various text formatting options for different model architectures

4. **Hardware Requirements**:
   - Designed for 3x RTX 3090 GPUs (24GB VRAM each, 72GB total)
   - Requires CUDA 12.1 and appropriate NVIDIA drivers
   - BF16 mixed precision support

### Training Process

The training workflow follows these steps:
1. Data preparation in appropriate format (instruction-tuning, multimodal)
2. Model configuration (quantization, LoRA parameters)
3. Training setup using DeepSpeed
4. Training execution with monitoring
5. Evaluation and visualization of results

### Memory-Efficient Training

The codebase is optimized for training large models with limited GPU resources:
- ZeRO-3 offloading parameters and optimizer states to CPU
- 4-bit quantization to reduce memory footprint
- LoRA fine-tuning to train only a small subset of parameters
- Flash Attention 2 for more efficient attention computation

## Common Issues

1. **Environment Issues**:
   - Always ensure you're in the `llm_trainer_env` conda environment
   - If you encounter dependency conflicts, resolve them within the conda environment without modifying system packages
   - Use `pip install --user` within the conda environment for additional packages

2. **Out of Memory Errors**:
   - Reduce batch size
   - Increase gradient accumulation steps
   - Use more aggressive quantization
   - Reduce context length

2. **Slow Training**:
   - Install and use flash-attention
   - Optimize DeepSpeed configuration
   - Use local datasets instead of loading from Hugging Face

3. **CUDA/GPU Issues**:
   - Check NVIDIA driver version compatibility
   - Verify CUDA installation with `nvcc --version`
   - Set appropriate environment variables: `CUDA_VISIBLE_DEVICES=0,1,2`

4. **PyTorch/Transformers Compatibility Issues**:
   - Use patched scripts in `scripts/` directory that handle compatibility issues
   - See `scripts/README_COMPATIBILITY.md` for details on the compatibility fixes
   - The main issues include missing `torch.get_default_device()` function and PyTorch security restrictions
   - For immediate results, use `scripts/inference/test_gemma_inference.py` which has all necessary patches
   - See `RESULTS.md` for a summary of working components and remaining issues