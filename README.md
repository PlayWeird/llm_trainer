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
conda activate llm_trainer_env
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
- `scripts/training/train_gemma_lora.py`: Fine-tuning script for Gemma models with LoRA using Dolly dataset
- `scripts/training/train_gemma_basic_lora.py`: Simpler training script with manual training loop
- `scripts/training/train_gemma3_vlm.py`: Vision-Language Model training script for Gemma 3 27B with Flickr8k
- `scripts/training/test_gemma_training.py`: Test script to verify basic model and training setup

### Inference Scripts
- `scripts/inference/test_model_loading.py`: Test script to verify Gemma 3 27B loading on GPUs
- `scripts/inference/test_gemma3_27b_loading.py`: Script for loading Gemma 3 27B with optimal settings
- `scripts/inference/test_gemma_inference.py`: Test script for basic Gemma inference
- `scripts/inference/test_cpu_setup.py`: Script to verify the Python environment setup
- `scripts/inference/test_multi_gpu_compatibility.py`: Test script for multi-GPU compatibility

### Preprocessing Scripts
- `scripts/preprocessing/download_test_datasets.py`: Script for downloading test datasets (Dolly for LLM and Flickr8k for VLM)
- `scripts/preprocessing/process_flickr8k.py`: Script for processing Flickr8k dataset for VLM training with train/val/test splits

### Shell Scripts
- `scripts/shell/training/run_training.sh`: Main script for running Gemma 3 27B training
- `scripts/shell/training/run_vlm_training.sh`: Script for running VLM training with Flickr8k dataset
- `scripts/shell/utils/download_test_datasets.sh`: Utility script for downloading datasets

### Utility Scripts
- `utils/data_preprocessing.py`: Utilities for data preparation including instruction-tuning format for Dolly and multimodal support for Flickr8k
- `visualizations/training/plot_training_metrics.py`: Visualization tools for training metrics

## Vision-Language Model (VLM) Training

The project now supports training Vision-Language Models using the Flickr8k dataset.

### VLM Training Workflow

1. **Download the Dataset**:
   ```bash
   bash scripts/shell/utils/download_test_datasets.sh --vlm-only --vlm flickr8k --vlm-samples 8000
   ```

2. **Process the Dataset**:
   ```bash
   python scripts/preprocessing/process_flickr8k.py \
     --input_dir datasets/test_dataset/vlm/flickr8k \
     --output_dir datasets/processed/vlm/flickr8k \
     --sample_size 8000
   ```

3. **Run VLM Training**:
   ```bash
   bash scripts/shell/training/run_vlm_training.sh \
     --input-dir datasets/test_dataset/vlm/flickr8k \
     --processed-dir datasets/processed/vlm/flickr8k \
     --output-dir outputs/gemma-3-27b-vlm-finetuned \
     --model-name google/gemma-3-27b-it \
     --sample-size 8000
   ```

### VLM Training Options

The `run_vlm_training.sh` script supports multiple options:

- `--input-dir`: Directory containing raw Flickr8k data
- `--processed-dir`: Directory to save processed data
- `--output-dir`: Output directory for training results
- `--model-name`: Model name or path
- `--sample-size`: Number of examples to include
- `--num-epochs`: Number of training epochs
- `--batch-size`: Batch size per GPU
- `--accum-steps`: Gradient accumulation steps
- `--learning-rate`: Learning rate
- `--ds-config`: DeepSpeed config file
- `--skip-processing`: Skip dataset processing
- `--skip-training`: Skip model training
- `--no-lora`: Disable LoRA fine-tuning
- `--no-4bit`: Disable 4-bit quantization

## Troubleshooting

### CUDA Issues

If you encounter CUDA errors:
1. Check your NVIDIA driver version with `nvidia-smi`
2. Check CUDA version compatibility with PyTorch
3. Resolve any driver/library version mismatches

### VLM Training Issues

If VLM training fails:
1. Check if the dataset was correctly processed
2. Ensure the image paths in the processed dataset are correct
3. Try with a smaller sample size (e.g., 10-100) for testing
4. Make sure you have enough GPU memory (reduce batch size or use more GPUs if needed)
5. Use the `--skip-processing` flag to reuse already processed data

## Next Steps

1. Create and test additional datasets for model training
2. Run small-scale training tests with the Gemma 2B model
3. Scale up to full Gemma 3 27B model training
4. Implement and test a proper evaluation pipeline
5. Expand VLM training to larger datasets and more diverse tasks

## Recent Updates

- Added VLM training capabilities with Flickr8k dataset
- Created specialized processing scripts for VLM data
- Enhanced data preprocessing utilities to support multimodal inputs with Flickr8k
- Updated data utils to use Dolly dataset format for instruction tuning
- Removed unused datasets (MMVP, Alpaca) and standardized on Dolly and Flickr8k
- Added shell scripts for end-to-end VLM training
- Organized and standardized file naming conventions
- Improved documentation across all scripts
- Ensured proper file organization