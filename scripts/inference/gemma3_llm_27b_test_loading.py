#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-GPU Gemma 3 27B loading script with proper compatibility patches.
Works around PyTorch 2.2 and Transformers compatibility issues.
"""

import os
import gc
import torch
import time
import logging
import argparse
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

def print_gpu_memory_stats():
    """Print current GPU memory status for all available GPUs."""
    num_gpus = torch.cuda.device_count()
    logger.info(f"GPU Memory Statistics ({num_gpus} GPUs):")
    
    for i in range(num_gpus):
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
        reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)  # GB
        allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        free_mem = total_mem - reserved_mem  # GB
        
        logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
        logger.info(f"  Total memory: {total_mem:.2f} GB")
        logger.info(f"  Reserved memory: {reserved_mem:.2f} GB")
        logger.info(f"  Allocated memory: {allocated_mem:.2f} GB")
        logger.info(f"  Free memory: {free_mem:.2f} GB")
        logger.info(f"  Utilization: {(allocated_mem / total_mem) * 100:.2f}%")

def get_model_size_info(model):
    """Get information about model size and parameter distribution."""
    device_map = model.hf_device_map if hasattr(model, "hf_device_map") else {}
    
    total_params = 0
    trainable_params = 0
    
    for n, p in model.named_parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    
    device_param_counts = {}
    for name, device in device_map.items():
        if device not in device_param_counts:
            device_param_counts[device] = 0
        
        # Find parameters in this module
        for param_name, param in model.named_parameters():
            if param_name.startswith(name):
                device_param_counts[device] += param.numel()
    
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Model has {trainable_params:,} trainable parameters ({trainable_params/total_params*100:.2f}%)")
    
    if device_param_counts:
        logger.info("Parameter distribution across devices:")
        for device, count in device_param_counts.items():
            logger.info(f"  {device}: {count:,} parameters ({count/total_params*100:.2f}%)")

def test_generation(model, tokenizer, prompts: List[str], max_new_tokens=100):
    """Test generation with the model on multiple prompts."""
    for i, prompt in enumerate(prompts):
        logger.info(f"Testing generation with prompt {i+1}: '{prompt}'")
        
        # Get model's actual device
        model_device = next(model.parameters()).device
        logger.info(f"Model's actual device: {model_device}")
        
        # Tokenize input and send to same device as model
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
        
        # Time generation
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True
            )
        gen_time = time.time() - start_time
        
        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generation completed in {gen_time:.2f} seconds")
        logger.info(f"Generated response: \n{response}\n")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load and test Gemma 3 27B with multiple GPUs")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-27b-it",
        help="Model name or path"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the concept of model quantization in simple terms.",
        help="Prompt for testing inference"
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="Use Flash Attention if available"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Print CUDA information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Print memory stats before loading
    logger.info("GPU memory before loading model:")
    print_gpu_memory_stats()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Start timing
    start_time = time.time()
    
    try:
        # Prepare for loading model
        logger.info(f"Loading model: {args.model_name}")
        
        # Setup quantization config
        quantization_config = None
        if args.use_4bit:
            logger.info("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif args.use_8bit:
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Set up model loading arguments
        model_kwargs = {
            "device_map": "auto",  # Distribute model across all available GPUs
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "use_safetensors": True,  # Use safetensors to avoid PyTorch vulnerability
        }
        
        # Add quantization config if specified
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            
        # Add flash attention if requested and available
        if args.flash_attn:
            try:
                import flash_attn
                logger.info("Flash Attention is available, enabling it for the model")
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning("Flash Attention not available, proceeding without it")
        
        # Load tokenizer
        logger.info("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Make sure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set padding token to EOS token")
        
        # Load model - this distributes across GPUs
        logger.info(f"Loading model with configuration: {model_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **model_kwargs
        )
        
        # Report loading time
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Print memory stats after loading
        logger.info("GPU memory after loading model:")
        print_gpu_memory_stats()
        
        # Print model distribution information
        logger.info("Model distribution across devices:")
        get_model_size_info(model)
        
        # Test generation
        logger.info("Testing model generation")
        test_generation(
            model, 
            tokenizer, 
            [args.prompt], 
            max_new_tokens=args.max_new_tokens
        )
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    # Set environment variables to help with multi-GPU setup
    os.environ["NCCL_P2P_DISABLE"] = "1"  # Help with certain multi-GPU setups
    
    exit_code = main()
    exit(exit_code)