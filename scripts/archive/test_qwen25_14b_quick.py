#!/usr/bin/env python3
"""Quick test to check if Qwen2.5-14B can be loaded"""

import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Patch for missing torch.get_default_device  
if not hasattr(torch, 'get_default_device'):
    torch.get_default_device = lambda: None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")
    
    try:
        logger.info(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("✓ Tokenizer loaded")
        
        logger.info(f"Loading model from {model_name}...")
        logger.info("This may take a few minutes for a 14B model...")
        
        # Try loading with device_map="auto" to distribute across GPUs
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        logger.info("✓ Model loaded successfully!")
        
        # Check memory usage
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i}: Allocated {mem_alloc:.2f}GB, Reserved {mem_reserved:.2f}GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)