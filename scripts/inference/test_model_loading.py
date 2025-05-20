#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify if Gemma 3 27B can be loaded on the available GPUs.
This script attempts to load the model with different configurations to find
the most memory-efficient setup for your hardware.
"""

import os
import gc
import torch
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_gpu_memory():
    """Get the current GPU memory usage for all GPUs."""
    gpu_memory = []
    num_gpus = torch.cuda.device_count()
    
    for i in range(num_gpus):
        with torch.cuda.device(i):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)  # Convert to GB
            reserved_memory = torch.cuda.memory_reserved() / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            free_memory = total_memory - reserved_memory
            
            gpu_memory.append({
                "id": i,
                "name": gpu_properties.name,
                "total": round(total_memory, 2),
                "reserved": round(reserved_memory, 2),
                "allocated": round(allocated_memory, 2),
                "free": round(free_memory, 2),
            })
    
    return gpu_memory


def test_model_loading(model_name, load_in_4bit=False, load_in_8bit=False, use_flash_attn=False):
    """Test loading the model with different configurations."""
    logger.info(f"Testing model loading for: {model_name}")
    logger.info(f"Configuration: 4-bit: {load_in_4bit}, 8-bit: {load_in_8bit}, Flash Attention: {use_flash_attn}")
    
    try:
        # Clear CUDA cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log GPU memory before loading
        logger.info("GPU memory before loading model:")
        for gpu in get_gpu_memory():
            logger.info(f"GPU {gpu['id']} ({gpu['name']}): {gpu['free']:.2f}GB free of {gpu['total']:.2f}GB total")
        
        # Set device map to auto for multi-GPU usage
        device_map = "auto"
        
        # Configure quantization if specified
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set up model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Use torch_dtype for mixed precision
        model_kwargs["torch_dtype"] = torch.bfloat16
        
        if use_flash_attn:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                logger.warning("Flash Attention not available, skipping flash attention implementation")
        
        # Load model
        logger.info(f"Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Log GPU memory after loading
        logger.info("GPU memory after loading model:")
        for gpu in get_gpu_memory():
            logger.info(f"GPU {gpu['id']} ({gpu['name']}): {gpu['free']:.2f}GB free of {gpu['total']:.2f}GB total")
        
        # Test simple generation
        input_text = "What are the main capabilities of Gemma 3 models?"
        logger.info(f"Testing generation with input: {input_text}")
        
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Generated response: {response}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Model test successful!")
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test loading Gemma 3 27B model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-27b-it",
        help="Name or path of the model to test",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use Flash Attention implementation",
    )
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires GPU support.")
        return
    
    # Log GPU information
    logger.info(f"Found {torch.cuda.device_count()} GPUs")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test different configurations
    configs = [
        # Test default fp16 loading
        {"load_in_4bit": False, "load_in_8bit": False, "use_flash_attn": False},
        # Test 8-bit quantization
        {"load_in_4bit": False, "load_in_8bit": True, "use_flash_attn": False},
        # Test 4-bit quantization
        {"load_in_4bit": True, "load_in_8bit": False, "use_flash_attn": False},
        # Test 4-bit quantization with flash attention if requested
        {"load_in_4bit": True, "load_in_8bit": False, "use_flash_attn": args.use_flash_attn},
    ]
    
    # If specific configs were specified, only use those
    if args.use_4bit or args.use_8bit:
        configs = [
            {
                "load_in_4bit": args.use_4bit,
                "load_in_8bit": args.use_8bit,
                "use_flash_attn": args.use_flash_attn
            }
        ]
    
    # Test each configuration
    success = False
    for config in configs:
        try:
            logger.info(f"\nTesting configuration: {config}")
            if test_model_loading(args.model_name, **config):
                logger.info(f"Successfully loaded model with configuration: {config}")
                success = True
                break
        except RuntimeError as e:
            logger.error(f"Failed to load model with configuration {config}: {e}")
    
    if not success:
        logger.error("Failed to load model with any configuration")
    else:
        logger.info("Found a working configuration for your hardware!")


if __name__ == "__main__":
    main()