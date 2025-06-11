#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify LLaVA-NeXT multi-GPU distribution
This script loads the model and monitors GPU memory usage to ensure
proper distribution across all available GPUs.
"""

import os
import time
import torch
import logging
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_gpu_memory():
    """Get current GPU memory usage for all GPUs."""
    gpu_memory = []
    num_gpus = torch.cuda.device_count()
    
    for i in range(num_gpus):
        with torch.cuda.device(i):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
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

def print_gpu_usage(stage):
    """Print current GPU usage for monitoring."""
    print(f"\n=== GPU Memory Usage - {stage} ===")
    for gpu in get_gpu_memory():
        print(f"GPU {gpu['id']}: {gpu['allocated']:.2f}GB allocated, {gpu['reserved']:.2f}GB reserved, {gpu['free']:.2f}GB free")

def main():
    print("Testing LLaVA-NeXT Multi-GPU Distribution")
    print("="*50)
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    
    model_name = "llava-hf/llava-v1.6-vicuna-13b-hf"
    
    print_gpu_usage("Initial")
    
    # Test 1: Auto device mapping (should distribute across GPUs)
    print("\n🔍 Test 1: Loading with device_map='auto'")
    try:
        torch.cuda.empty_cache()
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        print("Loading processor...")
        processor = LlavaNextProcessor.from_pretrained(model_name)
        
        print("Loading model with auto device mapping...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        
        print_gpu_usage("After auto device_map loading")
        
        # Show device mapping
        if hasattr(model, 'hf_device_map'):
            print(f"\nDevice mapping: {model.hf_device_map}")
        
        # Wait for monitoring
        print("\n⏳ Holding model in memory for 10 seconds for GPU monitoring...")
        time.sleep(10)
        
        del model, processor
        torch.cuda.empty_cache()
        
        print_gpu_usage("After cleanup")
        
    except Exception as e:
        print(f"❌ Auto device mapping failed: {e}")
    
    # Test 2: Manual distribution
    print("\n🔍 Test 2: Loading with manual device distribution")
    try:
        torch.cuda.empty_cache()
        time.sleep(2)
        
        # Try manual device mapping
        device_map = {
            "vision_tower": 0,
            "multi_modal_projector": 1,
            "language_model.model.embed_tokens": 0,
            "language_model.model.layers.0": 0,
            "language_model.model.layers.1": 0,
            "language_model.model.layers.2": 0,
            "language_model.model.layers.3": 0,
            "language_model.model.layers.4": 0,
            "language_model.model.layers.5": 0,
            "language_model.model.layers.6": 0,
            "language_model.model.layers.7": 0,
            "language_model.model.layers.8": 0,
            "language_model.model.layers.9": 1,
            "language_model.model.layers.10": 1,
            "language_model.model.layers.11": 1,
            "language_model.model.layers.12": 1,
            "language_model.model.layers.13": 1,
            "language_model.model.layers.14": 1,
            "language_model.model.layers.15": 1,
            "language_model.model.layers.16": 1,
            "language_model.model.layers.17": 1,
            "language_model.model.layers.18": 2,
            "language_model.model.layers.19": 2,
            "language_model.model.layers.20": 2,
            "language_model.model.layers.21": 2,
            "language_model.model.layers.22": 2,
            "language_model.model.layers.23": 2,
            "language_model.model.layers.24": 2,
            "language_model.model.layers.25": 2,
            "language_model.model.layers.26": 2,
            "language_model.model.norm": 2,
            "language_model.lm_head": 2,
        }
        
        print("Loading processor...")
        processor = LlavaNextProcessor.from_pretrained(model_name)
        
        print("Loading model with manual device mapping...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        
        print_gpu_usage("After manual device mapping")
        
        # Wait for monitoring
        print("\n⏳ Holding model in memory for 10 seconds for GPU monitoring...")
        time.sleep(10)
        
        del model, processor
        torch.cuda.empty_cache()
        
        print_gpu_usage("After cleanup")
        
    except Exception as e:
        print(f"❌ Manual device mapping failed: {e}")
    
    print("\n✅ Multi-GPU distribution test completed")

if __name__ == "__main__":
    main()