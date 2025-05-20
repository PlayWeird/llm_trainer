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

- `scripts/training/train_gemma3_27b.py`: Main training script for Gemma 3 27B
- `scripts/inference/test_model_loading.py`: Test script to verify model loading on GPUs
- `utils/data_preprocessing.py`: Utilities for data preparation
- `visualizations/training/plot_training_metrics.py`: Visualization tools for training metrics

## Troubleshooting

### CUDA Issues

If you encounter CUDA errors:
1. Check your NVIDIA driver version with `nvidia-smi`
2. Check CUDA version compatibility with PyTorch
3. Resolve any driver/library version mismatches

## Next Steps

1. Fix GPU/CUDA setup issues
2. Create a test dataset for initial training
3. Run a small-scale training test
4. Scale up to full model training