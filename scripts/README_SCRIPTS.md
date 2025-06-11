# Training Scripts Organization

This document describes the organized training and testing scripts with consistent naming conventions.

## 📋 Naming Convention

**Format**: `{model_family}_{size}_{type}_{purpose}.{ext}`

- **model_family**: 
  - LLM: `qwen25_llm`, `gemma_llm`
  - VLM: `qwen2_vlm`, `gemma3_vlm`, `llava_vlm`, `idefics3_vlm`
- **size**: `3b`, `7b`, `14b`, `32b`, etc.
- **type**: `training`, `test`, `inference`
- **purpose**: `production`, `quick`, `validation`, `multi_gpu`, `full`

## 🚀 Production Training Scripts

### LLM Training (Language Models Only)

#### Python Training Scripts (`scripts/training/`)
- **`qwen25_llm_training_production.py`** ✅ **WORKING**
  - Main production LLM training script
  - Supports Qwen2.5 models (3B, 7B, 14B, 32B)
  - Features: DeepSpeed ZeRO-3, LoRA, 4-bit quantization, multi-GPU

#### Shell Scripts (`scripts/shell/training/`)
- **`qwen25_llm_3b_training_production.sh`** ✅
- **`qwen25_llm_14b_training_production.sh`** ✅
- **`qwen25_llm_32b_training_production.sh`** ✅
- **`qwen25_llm_32b_training_multi_gpu.sh`** ✅ **PROVEN WORKING**
  - Uses VLM-style multi-GPU setup that works reliably

### VLM Training (Vision-Language Models)

#### Python Training Scripts (`scripts/training/`)
- **`qwen2_vlm_training_production.py`** ✅ **WORKING**
- **`gemma3_vlm_training_production.py`** ✅ **WORKING**
- **`llava_vlm_training_production.py`** ✅ **WORKING**
- **`idefics3_vlm_training_production.py`** ✅ **WORKING**

#### Shell Scripts (`scripts/shell/training/`)
- **`qwen2_vlm_7b_training_production.sh`** ✅
- **`gemma3_vlm_4b_training_production.sh`** ✅
- **`gemma3_vlm_12b_training_production.sh`** ✅
- **`gemma3_vlm_27b_training_production.sh`** ✅
- **`llava_vlm_13b_training_production.sh`** ✅
- **`idefics3_vlm_8b_training_production.sh`** ✅

## 🧪 Testing Scripts

### LLM Testing (`scripts/testing/`)
- **`qwen25_llm_14b_test_quick.py`** ✅ - Quick model loading validation
- **`qwen25_llm_32b_test_quick.py`** ✅ - 32B model loading validation
- **`qwen25_llm_14b_test_validation.py`** ✅ - Comprehensive validation
- **`qwen25_llm_14b_test_inference.py`** ✅ - Fine-tuned model inference
- **`gemma_llm_test_training.py`** ✅ - Basic training verification

### Shell Testing Scripts (`scripts/shell/testing/`)

#### LLM Testing
- **`qwen25_llm_14b_test_quick.sh`** ✅
- **`qwen25_llm_14b_test_multi_gpu.sh`** ✅ **WORKING**
- **`qwen25_llm_32b_test_multi_gpu.sh`** ✅ **WORKING**
- **`qwen25_llm_14b_test_full.sh`** ✅ **WORKING**

#### VLM Testing
- **`qwen2_vlm_7b_test_multi_gpu.sh`** ✅
- **`gemma3_vlm_12b_test_quick.sh`** ✅

## 🔍 Inference Scripts (`scripts/inference/`)
- **`gemma_llm_inference_production.py`** ✅ - Production inference with patches
- **`general_test_model_loading.py`** ✅ - General model loading tests
- **`general_test_multi_gpu.py`** ✅ - Multi-GPU compatibility tests
- **`gemma3_llm_27b_test_loading.py`** ✅ - Large model loading tests
- **`general_test_cpu_setup.py`** ✅ - CPU setup validation

## 🗃️ Archived Scripts (`scripts/archive/`)

The following obsolete/broken scripts have been moved to archive:
- `run_qwen25_14b_torchrun.sh` - Had PyTorch compatibility issues
- `run_qwen25_32b_single_gpu_test.sh` - 32B too large for single GPU
- `run_qwen25_32b_test.sh` - NCCL timeout issues (fixed in new scripts)
- `run_qwen25_14b_test.sh` - NCCL timeout issues (fixed in new scripts)

## 🎯 Quick Start Guide

### For LLM Training:

#### Test Qwen2.5-14B (Recommended first step):
```bash
# Quick validation
python scripts/testing/qwen25_llm_14b_test_quick.py

# Multi-GPU test (PROVEN WORKING)
./scripts/shell/testing/qwen25_llm_14b_test_multi_gpu.sh
```

#### Train Qwen2.5-32B (Multi-GPU):
```bash
# Test first (PROVEN WORKING)
./scripts/shell/testing/qwen25_llm_32b_test_multi_gpu.sh

# Production training
./scripts/shell/training/qwen25_llm_32b_training_multi_gpu.sh
```

### For VLM Training:

#### Test Qwen2-VL-7B:
```bash
./scripts/shell/testing/qwen2_vlm_7b_test_multi_gpu.sh
```

#### Production VLM Training:
```bash
./scripts/shell/training/qwen2_vlm_7b_training_production.sh
```

## ✅ Verified Working Configurations

### Multi-GPU Setup (3x RTX 3090):
1. **Qwen2.5-14B LLM**: ✅ Confirmed working
   - Memory usage: ~12GB per GPU during training
   - Training speed: ~0.147 steps/second

2. **Qwen2.5-32B LLM**: ✅ Multi-GPU initialization confirmed
   - Successfully loads across 3 GPUs with DeepSpeed ZeRO-3
   - NCCL communication working
   - Uses memory-optimized DeepSpeed config

3. **VLM Training**: ✅ Multi-GPU setup confirmed working
   - All VLM scripts use proven multi-GPU configuration
   - Successfully tested with Qwen2-VL models

## 🔧 Key Technical Details

### Multi-GPU Success Factors:
1. **Command format**: Use `deepspeed script.py` (not `deepspeed --num_gpus=3`)
2. **Device mapping**: Let DeepSpeed handle placement (`device_map=None`)
3. **Environment variables**: 
   - `NCCL_P2P_DISABLE=1`
   - `CUDA_VISIBLE_DEVICES=0,1,2`
4. **Memory optimization**: Use `ds_config_zero3_memory_opt.json` for 32B models

### Quantization Support:
- **4-bit quantization**: BitsAndBytes NF4 with double quantization
- **LoRA**: Parameter-efficient fine-tuning (r=16, α=32)
- **Memory optimizations**: Gradient checkpointing, CPU offloading

## 📊 Model Size Guidelines

| Model Size | GPUs Needed | Memory per GPU | Recommended Config |
|------------|-------------|----------------|-------------------|
| 3B-7B      | 1 GPU       | 8-12GB        | Single GPU scripts |
| 14B        | 1-3 GPUs    | 12-16GB       | Multi-GPU for speed |
| 32B        | 3 GPUs      | 20-24GB       | Multi-GPU required |

All configurations use 4-bit quantization + LoRA for maximum efficiency.