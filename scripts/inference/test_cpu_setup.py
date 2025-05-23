#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify the Python environment is correctly set up for training LLMs.

This script checks for all required dependencies needed for Gemma 3 training:
- Python version
- PyTorch and CUDA availability
- Transformers library
- Datasets library
- Accelerate library for distributed training
- PEFT library for parameter-efficient fine-tuning
- BitsAndBytes for quantization
- NumPy

Usage:
    python test_cpu_setup.py

The script will log information about each dependency and report any missing
packages that need to be installed.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """
    Test the Python environment setup.
    
    This function checks for all required dependencies and their versions.
    It attempts to import each package and logs its version if available,
    or logs an error if the package is not installed.
    
    For PyTorch, it also checks CUDA availability and GPU information.
    """
    logger.info(f"Python version: {sys.version}")
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        logger.error("PyTorch not installed")
    
    try:
        import transformers
        logger.info(f"Transformers version: {transformers.__version__}")
    except ImportError:
        logger.error("Transformers not installed")
    
    try:
        import datasets
        logger.info(f"Datasets version: {datasets.__version__}")
    except ImportError:
        logger.error("Datasets not installed")
    
    try:
        import accelerate
        logger.info(f"Accelerate version: {accelerate.__version__}")
    except ImportError:
        logger.error("Accelerate not installed")
    
    try:
        import peft
        logger.info(f"PEFT version: {peft.__version__}")
    except ImportError:
        logger.error("PEFT not installed")
    
    try:
        import bitsandbytes
        logger.info(f"BitsAndBytes version: {bitsandbytes.__version__}")
    except ImportError:
        logger.error("BitsAndBytes not installed")
    
    try:
        import numpy
        logger.info(f"NumPy version: {numpy.__version__}")
    except ImportError:
        logger.error("NumPy not installed")
    
    logger.info("Environment check completed")

if __name__ == "__main__":
    main()