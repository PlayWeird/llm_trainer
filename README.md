# LLM Trainer

A project for training and fine-tuning local Large Language Models (LLMs) and Vision-Language Models (VLMs). The project focuses on testing various model architectures from Hugging Face, including but not limited to LLaVA, Llama, Gemma, and Deepseek R1.

## Project Purpose

This project aims to:
- Train LLMs from scratch and fine-tune existing models
- Experiment with various model architectures from Hugging Face
- Test VLM training methods and techniques
- Store and manage datasets for different training scenarios
- Evaluate model performance with custom metrics
- Visualize training progress and results

## Current Focus

The current focus is on Gemma 3 27B, a powerful multimodal model that supports both text and image inputs.

## Directory Structure

```
llm_trainer/
├── configs/             # Configuration files
│   ├── model/           # Model architecture configurations
│   └── training/        # Training hyperparameters
├── datasets/            # Training and testing datasets
├── evaluation/          # Evaluation code and results
│   ├── analysis/        # Analysis tools
│   └── metrics/         # Performance metrics
├── logs/                # Training logs
├── models/              # Model architecture definitions
│   ├── llm/             # Text-only language models
│   └── vlm/             # Vision-language models
├── notebooks/           # Jupyter notebooks for exploration
├── outputs/             # Saved model outputs and checkpoints
├── scripts/             # Python scripts
│   ├── inference/       # Code for running inference
│   ├── preprocessing/   # Data preprocessing
│   └── training/        # Training code
├── utils/               # Utility functions
└── visualizations/      # Visualization tools
    ├── inference/       # Inference result visualizations
    └── training/        # Training metric visualizations
```

## Environment Setup

1. Create the conda environment:
```bash
conda env create -f environment.yml
```

2. Activate the environment:
```bash
conda activate gemma3_env
```

3. Install additional requirements:
```bash
pip install -r requirements.txt
```

## GPU Setup Requirements

This project is designed to work with 3x RTX 3090 GPUs (24GB VRAM each, 72GB total). The Gemma 3 27B model can be trained using:
- DeepSpeed ZeRO-3 for distributed training
- 4-bit quantization for memory efficiency
- LoRA for parameter-efficient fine-tuning

## Key Scripts

### Training Scripts
- `scripts/training/train_gemma3_27b.py`: Main training script for Gemma 3 27B
- `scripts/training/train_gemma_lora.py`: Fine-tuning script for Gemma models with LoRA
- `scripts/training/train_gemma_basic_lora.py`: Simpler training script with manual training loop
- `scripts/training/test_gemma_training.py`: Test script to verify basic model and training setup

### Inference Scripts
- `scripts/inference/test_model_loading.py`: Test script to verify Gemma 3 27B loading on GPUs
- `scripts/inference/test_gemma3_27b_loading.py`: Script for loading Gemma 3 27B with optimal settings
- `scripts/inference/test_gemma_inference.py`: Test script for basic Gemma inference
- `scripts/inference/test_cpu_setup.py`: Script to verify the Python environment setup
- `scripts/inference/test_multi_gpu_compatibility.py`: Test script for multi-GPU compatibility

### Utility Scripts
- `utils/data_preprocessing.py`: Utilities for data preparation
- `visualizations/training/plot_training_metrics.py`: Visualization tools for training metrics

## Troubleshooting

### CUDA Issues

If you encounter CUDA errors:
1. Check your NVIDIA driver version with `nvidia-smi`
2. Check CUDA version compatibility with PyTorch
3. Resolve any driver/library version mismatches

## Next Steps

1. Create and test datasets for model training
2. Run a small-scale training test with the Gemma 2B model
3. Scale up to full Gemma 3 27B model training
4. Implement and test a proper evaluation pipeline

## Recent Updates

- Organized and standardized file naming conventions
- Improved documentation across all scripts
- Removed redundant test files
- Ensured proper file organization
- Enhanced README with comprehensive script descriptions