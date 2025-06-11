#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patched test script for Gemma model that works around the 'get_default_device' issue 
and ensures correct device mapping.
"""

import os
import sys
import gc
import torch
import time
import logging
import argparse
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

def get_gpu_memory():
    """Get the current GPU memory usage for all GPUs."""
    gpu_memory = []
    num_gpus = torch.cuda.device_count()
    
    for i in range(num_gpus):
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        reserved_mem = torch.cuda.memory_reserved(i) / 1024**3
        allocated_mem = torch.cuda.memory_allocated(i) / 1024**3
        free_mem = total_mem - reserved_mem
        
        gpu_memory.append({
            "id": i,
            "name": torch.cuda.get_device_name(i),
            "total": round(total_mem, 2),
            "reserved": round(reserved_mem, 2),
            "allocated": round(allocated_mem, 2),
            "free": round(free_mem, 2),
        })
    
    return gpu_memory

def main():
    parser = argparse.ArgumentParser(description="Test Gemma model loading and inference")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="google/gemma-2-2b",  # Using Gemma 2-2B which is publicly available
        help="Model name or path"
    )
    parser.add_argument(
        "--use_4bit", 
        action="store_true", 
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Explain what large language models are in simple terms.",
        help="Prompt for testing inference"
    )
    
    args = parser.parse_args()
    
    # Print PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # Print initial GPU memory
    print("\nGPU memory before loading model:")
    for gpu in get_gpu_memory():
        print(f"GPU {gpu['id']} ({gpu['name']}): {gpu['free']:.2f}GB free, {gpu['allocated']:.2f}GB allocated, {gpu['total']:.2f}GB total")
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        # Set the device explicitly
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        # Configure quantization if requested
        quantization_config = None
        if args.use_4bit:
            print("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        
        # Load tokenizer
        print(f"\nLoading tokenizer for {args.model_name}...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds")
        
        # Make sure the tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set padding token to EOS token")
        
        # Load model
        print(f"\nLoading model {args.model_name}...")
        start_time = time.time()
        
        # Set up model loading kwargs
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "use_safetensors": True,  # Force safetensors format
            "trust_remote_code": True
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Load model - for multi-GPU we need to handle device mapping differently
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1 and not args.use_4bit:
            # Simple balanced device map for multi-GPU without quantization
            print(f"Using balanced device map for {num_gpus} GPUs")
            model_kwargs["device_map"] = "balanced"
        else:
            # For single GPU or 4-bit quantized model
            print(f"Will move model to {device} after loading")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            **model_kwargs
        )
        
        # Move model to device if needed (when not using device_map)
        if "device_map" not in model_kwargs:
            print(f"Moving model to {device}")
            model = model.to(device)
        
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Print GPU memory after loading
        print("\nGPU memory after loading model:")
        for gpu in get_gpu_memory():
            print(f"GPU {gpu['id']} ({gpu['name']}): {gpu['free']:.2f}GB free, {gpu['allocated']:.2f}GB allocated, {gpu['total']:.2f}GB total")
        
        # Test inference
        print(f"\nTesting inference with prompt: '{args.prompt}'")
        start_time = time.time()
        
        inputs = tokenizer(args.prompt, return_tensors="pt")
        
        # Make sure inputs are on the correct device
        if hasattr(model, "device"):
            device_to_use = model.device
        else:
            # If we can't determine model device, use first GPU
            device_to_use = device
            
        inputs = {k: v.to(device_to_use) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
        
        inference_time = time.time() - start_time
        
        # Decode and print response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generation completed in {inference_time:.2f} seconds")
        print(f"Generated response: \n{response}")
        
        # Clean up
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        # Clean up
        torch.cuda.empty_cache()
        gc.collect()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)