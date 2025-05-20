# GPU Setup for Gemma 3 27B Training

This document provides instructions for setting up your 3x RTX 3090 GPUs for training Gemma 3 27B models.

## Hardware Requirements

- 3x NVIDIA RTX 3090 GPUs (24GB VRAM each)
- NVLink bridge (recommended but not required)
- Sufficient system RAM (128GB+ recommended)
- Adequate cooling solution for sustained load
- High-speed storage (SSD/NVMe)

## Software Requirements

1. **NVIDIA Drivers**: 
   - Minimum driver version: 525.105.17
   - Recommended driver version: Latest stable driver
   - Install via: `sudo apt install nvidia-driver-XXX` (where XXX is the version)

2. **CUDA Toolkit**:
   - Required version: CUDA 12.1
   - PyTorch 2.2.0 is built with CUDA 12.1
   - Installation: https://developer.nvidia.com/cuda-12-1-0-download-archive

3. **cuDNN**:
   - Recommended: cuDNN 8.9.2 for CUDA 12.1
   - Installation: https://developer.nvidia.com/cudnn

## Verifying GPU Setup

```bash
# Check NVIDIA driver version
nvidia-smi

# Check CUDA version
nvcc --version

# Check GPU detection in PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"
```

## Troubleshooting

### Driver/Library Version Mismatch

If you see an error like:
```
Failed to initialize NVML: Driver/library version mismatch
```

Solutions:
1. Reboot the system
2. Reinstall the NVIDIA driver
3. Ensure CUDA version compatibility

### No CUDA Device Found

If PyTorch doesn't detect your GPUs:
1. Check if NVIDIA driver is loaded: `lsmod | grep nvidia`
2. Verify GPU is recognized by the system: `lspci | grep -i nvidia`
3. Check CUDA installation: `ls -l /usr/local/cuda`

## Optimizing Multi-GPU Training

For distributed training across 3x RTX 3090 GPUs:

1. **NVLink Configuration** (if available):
   ```bash
   # Check NVLink status
   nvidia-smi nvlink -s
   ```

2. **DeepSpeed Configuration**:
   - Use ZeRO Stage 3 for maximum memory efficiency
   - Enable CPU offloading for parameter storage
   - Configuration file: `/configs/training/ds_config_zero3.json`

3. **Environment Variables**:
   ```bash
   # Set before training
   export NCCL_P2P_DISABLE=1  # May help with certain multi-GPU setups
   export CUDA_VISIBLE_DEVICES=0,1,2  # Specify GPUs to use
   ```

## Memory Management

To fit Gemma 3 27B across 3x 24GB GPUs:

1. **Quantization**: Use 4-bit quantization via BitsAndBytes
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "google/gemma-3-27b-it",
       quantization_config=BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_compute_dtype=torch.bfloat16
       )
   )
   ```

2. **Parameter-Efficient Fine-Tuning (PEFT)**:
   - Use LoRA for fine-tuning instead of full-parameter training
   - Target only attention layers to reduce memory requirements

3. **Gradient Accumulation**:
   - Increase `gradient_accumulation_steps` to effectively increase batch size
   - Recommended: 16-32 steps depending on available memory

## Monitoring Training

1. **GPU Utilization and Memory**: 
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Process Monitoring**:
   ```bash
   nvidia-smi pmon -i 0,1,2
   ```

3. **Temperature Monitoring**:
   ```bash
   nvidia-smi -q -d TEMPERATURE
   ```

## Performance Optimization

1. **Optimize Batch Size**:
   - Start with small batch sizes (1 per GPU)
   - Increase until out of memory errors

2. **Flash Attention**:
   - Install flash-attention for faster training
   - Configure in model loading:
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       "google/gemma-3-27b-it",
       attn_implementation="flash_attention_2"
   )
   ```

3. **Mixed Precision**:
   - Always use BF16 or FP16 for training
   - BF16 is recommended for Gemma models