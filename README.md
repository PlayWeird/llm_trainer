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

## Getting Started

(Instructions to be added)

## Requirements

See `requirements.txt` for a list of dependencies.