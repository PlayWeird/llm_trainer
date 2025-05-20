#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify compatibility between PyTorch, Transformers, and multi-GPU setup
without downloading the full model. This script checks if all the necessary patches
and configurations are working correctly.
"""

import os
import torch
import logging
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig
from accelerate import init_empty_weights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# PATCH: Add get_default_device to torch module if it doesn't exist
if not hasattr(torch, 'get_default_device'):
    logger.info("Adding missing get_default_device to torch module")
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
    # Add the function to the torch module
    torch.get_default_device = get_default_device
    logger.info(f"Patched torch.get_default_device() returning {torch.get_default_device()}")

def print_gpu_info():
    """Print information about available GPUs and their memory."""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_props.name}")
            logger.info(f"  Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
            logger.info(f"  CUDA Capability: {gpu_props.major}.{gpu_props.minor}")
            logger.info(f"  Multi-processors: {gpu_props.multi_processor_count}")
    else:
        logger.info("No CUDA GPUs detected")

def test_accelerate_compatibility():
    """Test compatibility with Accelerate for multi-GPU setup."""
    try:
        from accelerate import Accelerator
        
        accelerator = Accelerator()
        logger.info(f"Accelerator device: {accelerator.device}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Distributed type: {accelerator.distributed_type}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
        
        logger.info("Accelerator compatibility check: PASS")
        return True
    except Exception as e:
        logger.error(f"Accelerator compatibility check: FAIL - {e}")
        return False

def test_transformers_compatibility():
    """Test compatibility with the Transformers library."""
    try:
        from transformers import AutoModel
        
        # Just check version and availability without loading a model
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Test if we can create a config
        config = AutoConfig.from_pretrained("bert-base-uncased")
        logger.info(f"Config test successful: {type(config).__name__}")
        
        logger.info("Transformers compatibility check: PASS")
        return True
    except Exception as e:
        logger.error(f"Transformers compatibility check: FAIL - {e}")
        return False

def test_device_map_functionality():
    """Test if the device map functionality works correctly."""
    try:
        from accelerate import infer_auto_device_map, init_empty_weights
        
        # Create a dummy config for a large model
        class LargeConfig(PretrainedConfig):
            model_type = "large_model"
            def __init__(self, **kwargs):
                self.hidden_size = 4096
                self.num_hidden_layers = 40
                self.num_attention_heads = 32
                super().__init__(**kwargs)
        
        # Create a dummy model class
        class LargeModel(PreTrainedModel):
            config_class = LargeConfig
            
            def __init__(self, config):
                super().__init__(config)
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(config.hidden_size, config.hidden_size) 
                    for _ in range(config.num_hidden_layers)
                ])
                self.head = torch.nn.Linear(config.hidden_size, config.hidden_size)
                
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.head(x)
        
        # Create config and initialize with empty weights
        config = LargeConfig()
        
        with init_empty_weights():
            model = LargeModel(config)
            
        # Infer device map
        max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory / 1024**3 - 1:.0f}GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "96GiB"  # Allow CPU offloading
        
        device_map = infer_auto_device_map(
            model, 
            max_memory=max_memory,
            no_split_module_classes=["Layer"]
        )
        
        # Check if multiple devices are used
        used_devices = set(device_map.values())
        logger.info(f"Device map uses {len(used_devices)} device(s): {used_devices}")
        
        # The device map values might be integers for CUDA devices
        # Need to check if they represent GPU devices
        cuda_devices = []
        for device in used_devices:
            if isinstance(device, int) and device >= 0 and device < torch.cuda.device_count():
                cuda_devices.append(device)
            elif isinstance(device, torch.device) and device.type == 'cuda':
                cuda_devices.append(device)
            elif isinstance(device, str) and device.startswith('cuda'):
                cuda_devices.append(device)
                
        logger.info(f"Found {len(cuda_devices)} CUDA devices in map: {cuda_devices}")
        test_passed = len(cuda_devices) > 0
        
        if test_passed:
            logger.info("Device map functionality check: PASS")
        else:
            logger.info("Device map functionality check: FAIL - No GPUs in device map")
        
        return test_passed
    except Exception as e:
        logger.error(f"Device map functionality check: FAIL - {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_quantization_compatibility():
    """Test compatibility with quantization libraries."""
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        
        logger.info(f"BitsAndBytes version: {bnb.__version__}")
        
        # Test creating 4-bit and 8-bit configs
        bnb_4bit_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        bnb_8bit_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        
        logger.info(f"4-bit config: {bnb_4bit_config}")
        logger.info(f"8-bit config: {bnb_8bit_config}")
        
        # Create a linear layer and test 4-bit quantization
        try:
            from bitsandbytes.nn import Linear4bit
            linear_4bit = Linear4bit(128, 128, bias=True, compute_dtype=torch.bfloat16)
            logger.info(f"4-bit linear layer created: {type(linear_4bit).__name__}")
            logger.info("Quantization compatibility check: PASS")
            return True
        except Exception as e:
            logger.error(f"Failed to create quantized layer: {e}")
            return False
            
    except ImportError as e:
        logger.error(f"Quantization compatibility check: FAIL - {e}")
        return False

def main():
    """Run all compatibility tests."""
    os.environ["NCCL_P2P_DISABLE"] = "1"  # Help with certain multi-GPU setups
    
    logger.info("===== Testing Multi-GPU Compatibility =====")
    
    # Print GPU information
    print_gpu_info()
    
    # Run tests
    tests = [
        ("Accelerate", test_accelerate_compatibility),
        ("Transformers", test_transformers_compatibility),
        ("Device Map", test_device_map_functionality),
        ("Quantization", test_quantization_compatibility),
    ]
    
    all_passed = True
    results = []
    
    for name, test_func in tests:
        logger.info(f"\n===== Testing {name} Compatibility =====")
        try:
            passed = test_func()
            results.append((name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            logger.error(f"Error during {name} test: {e}")
            all_passed = False
            results.append((name, False))
    
    # Print summary
    logger.info("\n===== Compatibility Test Summary =====")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{name:15} | {status}")
    
    logger.info(f"\nOverall compatibility status: {'✅ PASSED' if all_passed else '❌ FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())