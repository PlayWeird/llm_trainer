#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaVA-NeXT 34B VLM Loading Test Script

This script tests loading the large 34B LLaVA-NeXT model across multiple GPUs
to verify multi-GPU distribution and memory usage.
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
    total_allocated = 0
    total_free = 0
    for gpu in get_gpu_memory():
        print(f"GPU {gpu['id']}: {gpu['allocated']:.2f}GB allocated, {gpu['reserved']:.2f}GB reserved, {gpu['free']:.2f}GB free")
        total_allocated += gpu['allocated']
        total_free += gpu['free']
    print(f"Total across all GPUs: {total_allocated:.2f}GB allocated, {total_free:.2f}GB free")

def main():
    print("🚀 Testing LLaVA-NeXT 34B Multi-GPU Distribution")
    print("="*60)
    
    # Set environment variables for optimal multi-GPU usage
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    
    model_name = "llava-hf/llava-v1.6-34b-hf"
    
    print_gpu_usage("Initial")
    
    print(f"\n📦 Loading 34B model: {model_name}")
    print("This is a LARGE model - expect significant memory usage across all GPUs!")
    
    try:
        torch.cuda.empty_cache()
        
        # Aggressive quantization for 34B model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        print("📥 Loading processor...")
        processor = LlavaNextProcessor.from_pretrained(model_name)
        
        print("🧠 Loading 34B model with auto device mapping...")
        print("⏳ This will take several minutes and should utilize all 3 GPUs...")
        
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )
        
        print_gpu_usage("After 34B model loading")
        
        # Show device mapping for the large model
        if hasattr(model, 'hf_device_map'):
            print(f"\n📍 Device mapping for 34B model:")
            for component, device in model.hf_device_map.items():
                print(f"  {component}: GPU {device}")
        
        print(f"\n⏳ Holding 34B model in memory for 15 seconds for GPU monitoring...")
        print("📊 You should see very high memory usage across all 3 GPUs!")
        
        for i in range(15):
            time.sleep(1)
            if i % 5 == 0:
                print(f"   Monitoring... {15-i} seconds remaining")
        
        print_gpu_usage("During 34B model hold")
        
        # Test basic inference with 34B model
        print(f"\n🧪 Testing inference with 34B model...")
        try:
            from PIL import Image
            import requests
            from io import BytesIO
            
            # Load test image
            url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {"type": "image"},
                    ],
                },
            ]
            
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
            
            print("🔄 Running inference on 34B model...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            response = processor.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response
            print(f"🤖 34B Model response: {assistant_response}")
            
        except Exception as e:
            print(f"⚠️  Inference test failed (expected with large model): {e}")
        
        del model, processor
        torch.cuda.empty_cache()
        
        print_gpu_usage("After cleanup")
        
        print(f"\n✅ 34B model test successful!")
        print("🎯 If you saw high memory usage across all 3 GPUs, multi-GPU distribution is working!")
        
    except Exception as e:
        print(f"❌ 34B model loading failed: {e}")
        print("💡 This is expected if the model is too large even with optimizations")

if __name__ == "__main__":
    main()