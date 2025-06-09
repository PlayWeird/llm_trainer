# LLM Trainer

A comprehensive framework for training and fine-tuning Large Language Models (LLMs) and Vision-Language Models (VLMs). Successfully implements multi-GPU training with memory optimization for large models like Gemma 3 27B using DeepSpeed ZeRO-3, 4-bit quantization, and LoRA fine-tuning.

## Project Purpose

This project aims to:
- Train LLMs from scratch and fine-tune existing models
- Experiment with various model architectures from Hugging Face
- Test VLM training methods and techniques
- Store and manage datasets for different training scenarios
- Evaluate model performance with custom metrics
- Visualize training progress and results

## Current Status

✅ **Working**: Multi-GPU Gemma 3 27B training with DeepSpeed ZeRO-3  
✅ **Tested**: Qwen2.5-14B successful training run completed  
✅ **Compatible**: PyTorch 2.2.0 + Transformers 4.53.0 with compatibility patches  
✅ **Optimized**: Memory-efficient training on 3x RTX 3090 GPUs (72GB total VRAM)

See `docs/project_status.md` for comprehensive project status and technical details.

## Directory Structure

```
llm_trainer/
├── configs/             # Configuration files
│   └── training/        # DeepSpeed and training configurations
├── datasets/            # Training and testing datasets
│   ├── processed/       # Processed datasets ready for training
│   └── test_dataset/    # Raw test datasets (Dolly, Flickr8k)
├── docs/                # Documentation and guides
│   ├── gemma3_setup.md  # Gemma 3 setup and training guide
│   ├── gpu_setup.md     # GPU configuration guide
│   ├── project_status.md # Comprehensive project status
│   └── training_workflow.md # Training workflow documentation
├── evaluation/          # Evaluation code and results
├── models/              # Model artifacts and checkpoints
│   ├── llm/             # Text-only language models
│   └── vlm/             # Vision-language models
├── outputs/             # Successful training outputs only
│   └── qwen25-14b-quick-test/ # Working model from successful run
├── scripts/             # Organized by functionality
│   ├── inference/       # Model testing and inference scripts
│   ├── preprocessing/   # Data preprocessing utilities
│   ├── shell/           # Shell scripts organized by purpose
│   │   ├── inference/   # Inference shell scripts
│   │   ├── setup/       # Environment setup scripts
│   │   ├── testing/     # Testing automation scripts
│   │   ├── training/    # Training shell scripts
│   │   └── utils/       # Utility shell scripts
│   ├── testing/         # Comprehensive test suites
│   └── training/        # Model-specific training scripts
├── utils/               # Reusable utility modules
│   ├── data_preprocessing.py # Data processing utilities
│   ├── model_utils.py   # Model loading and configuration
│   ├── training_config.py # Training configuration management
│   ├── training_utils.py # Training helper functions
│   └── vlm_data_utils.py # VLM-specific data utilities
└── visualizations/      # Training and inference visualization tools
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

4. Optional - Install flash attention for faster training:
```bash
pip install flash-attn --no-build-isolation
```

See `docs/gpu_setup.md` for detailed GPU configuration and `CLAUDE.md` for complete setup instructions.

## GPU Setup Requirements

This project is designed to work with 3x RTX 3090 GPUs (24GB VRAM each, 72GB total). The Gemma 3 27B model can be trained using:
- DeepSpeed ZeRO-3 for distributed training
- 4-bit quantization for memory efficiency
- LoRA for parameter-efficient fine-tuning

## Quick Start

### Basic Training Example
```bash
# Train Qwen2.5-14B (proven working configuration)
deepspeed scripts/training/train_qwen25_full.py \
  --deepspeed configs/training/ds_config_zero3_memory_opt.json \
  --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
  --train_file datasets/test_dataset/llm/dolly_formatted.json \
  --output_dir outputs/qwen25-14b-production \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --use_lora True \
  --use_4bit True
```

### Test Environment
```bash
# Verify GPU setup and model loading
python scripts/inference/test_gemma_inference.py --use_4bit

# Run compatibility tests
bash scripts/shell/utils/run_compatibility_test.sh
```

## Key Components

### Proven Training Scripts
- **`scripts/training/train_qwen25_full.py`** - Tested and working Qwen2.5 training
- **`scripts/training/train_gemma3_vlm.py`** - Vision-Language Model training with modular components
- **`scripts/training/train_idefics3_vlm.py`** - Idefics3 VLM training implementation

### Comprehensive Testing Suite
- **`scripts/inference/test_gemma_inference.py`** - Most comprehensive inference test with patches
- **`scripts/testing/test_refactored_training.py`** - Complete training integration test
- **`scripts/testing/validate_refactoring.py`** - Validation for modular components

### Modular Utilities
- **`utils/training_config.py`** - Centralized configuration management
- **`utils/vlm_data_utils.py`** - Flexible VLM data processing
- **`utils/model_utils.py`** - Reusable model loading functions

### Shell Script Automation
- **`scripts/shell/training/`** - Production training scripts for different models
- **`scripts/shell/testing/`** - Automated testing workflows
- **`scripts/shell/utils/`** - Utility scripts for monitoring and setup

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

### Common Issues

**PyTorch Compatibility**: The project includes patches for PyTorch 2.2.0 + Transformers 4.53.0 compatibility issues. Use the provided test scripts which include necessary patches.

**Memory Issues**: 
- Reduce `per_device_train_batch_size` 
- Increase `gradient_accumulation_steps`
- Ensure you're using 4-bit quantization (`--use_4bit True`)

**Model Loading**: Use `scripts/inference/test_gemma_inference.py` for the most robust model loading test with all compatibility patches.

**GPU Setup**: Run `nvidia-smi` and verify all 3 GPUs are detected. See `docs/gpu_setup.md` for detailed configuration.

For comprehensive troubleshooting, see `docs/project_status.md` and `scripts/README_COMPATIBILITY.md`.

## Architecture Highlights

### Memory Optimization
- **DeepSpeed ZeRO-3**: Offloads parameters and optimizer states to CPU
- **4-bit Quantization**: Reduces memory footprint by ~75%
- **LoRA Fine-tuning**: Trains only small subset of parameters
- **Gradient Accumulation**: Increases effective batch size without memory overhead

### Multi-GPU Training
- **Model Sharding**: Automatic distribution across 3x RTX 3090 GPUs
- **Memory Efficiency**: 72GB total VRAM for large model training
- **Communication Optimization**: Efficient gradient synchronization

### Modular Design
- **Reusable Components**: Centralized configuration and utility modules
- **Flexible Data Processing**: Auto-detection of dataset formats
- **Comprehensive Testing**: Integration tests and compatibility validation

## Contributing

1. Follow the project organization rules in `CLAUDE.md`
2. Use the established directory structure
3. Remove test artifacts and failed training runs regularly
4. Consolidate documentation rather than creating duplicates
5. Test changes with the provided test suites

## Documentation

- **`CLAUDE.md`** - Project instructions and organization rules
- **`docs/project_status.md`** - Comprehensive project status and achievements
- **`docs/gemma3_setup.md`** - Gemma 3 specific setup and training guide
- **`docs/gpu_setup.md`** - GPU configuration and troubleshooting
- **`docs/training_workflow.md`** - Training workflow documentation