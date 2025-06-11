#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaVA-NeXT 13B VLM Loading Test Script

This script tests loading and basic inference with the LLaVA-NeXT 13B vision-language model.
It verifies the model can be loaded on the available GPU hardware and performs basic
image captioning and visual question answering tasks.

Supported models:
- llava-hf/llava-v1.6-vicuna-13b-hf (recommended for 3x RTX 3090)
- llava-hf/llava-v1.6-vicuna-7b-hf (lighter alternative)
- llava-hf/llama3-llava-next-8b-hf (LLaMA-3 based)

Example usage:
    python llava_next_vlm_13b_test_loading.py --use_4bit
    python llava_next_vlm_13b_test_loading.py --model_name="llava-hf/llava-v1.6-vicuna-7b-hf"
"""

import os
import gc
import torch
import argparse
import logging
from PIL import Image
import requests
from io import BytesIO
from transformers import (
    LlavaNextProcessor, 
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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


def load_test_image():
    """Load a test image for inference."""
    try:
        url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        logger.info("Successfully loaded test image from URL")
        return image
    except Exception as e:
        logger.warning(f"Failed to load image from URL: {e}")
        
        try:
            test_image_path = "/home/user/llm_trainer/datasets/test_dataset/vlm/flickr8k/images/flickr8k_0.jpg"
            if os.path.exists(test_image_path):
                image = Image.open(test_image_path).convert('RGB')
                logger.info(f"Successfully loaded local test image: {test_image_path}")
                return image
        except Exception as e2:
            logger.warning(f"Failed to load local image: {e2}")
        
        logger.info("Creating a simple test image")
        image = Image.new('RGB', (224, 224), color='blue')
        return image


def test_llava_next_loading(model_name, load_in_4bit=False, load_in_8bit=False, use_flash_attn=False):
    """
    Test loading LLaVA-NeXT model with different configurations.
    
    Args:
        model_name (str): HuggingFace model name
        load_in_4bit (bool): Whether to use 4-bit quantization
        load_in_8bit (bool): Whether to use 8-bit quantization  
        use_flash_attn (bool): Whether to use Flash Attention 2
        
    Returns:
        bool: True if model loading and inference succeeded, False otherwise
    """
    logger.info(f"Testing LLaVA-NeXT model loading for: {model_name}")
    logger.info(f"Configuration: 4-bit: {load_in_4bit}, 8-bit: {load_in_8bit}, Flash Attention: {use_flash_attn}")
    
    try:
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("GPU memory before loading model:")
        for gpu in get_gpu_memory():
            logger.info(f"GPU {gpu['id']} ({gpu['name']}): {gpu['free']:.2f}GB free of {gpu['total']:.2f}GB total")
        
        device_map = "auto"
        
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": torch.bfloat16,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        if use_flash_attn:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                logger.warning("Flash Attention not available, using standard attention")
        
        logger.info(f"Loading LLaVA-NeXT processor for {model_name}...")
        processor = LlavaNextProcessor.from_pretrained(model_name)
        
        logger.info(f"Loading LLaVA-NeXT model {model_name}...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        logger.info("GPU memory after loading model:")
        for gpu in get_gpu_memory():
            logger.info(f"GPU {gpu['id']} ({gpu['name']}): {gpu['free']:.2f}GB free of {gpu['total']:.2f}GB total")
        
        image = load_test_image()
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image? Describe it in detail."},
                    {"type": "image"},
                ],
            },
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        logger.info(f"Generated prompt: {prompt}")
        
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
        
        logger.info("Running inference...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response = processor.decode(outputs[0], skip_special_tokens=True)
        
        assistant_response = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response
        logger.info(f"Model response: {assistant_response}")
        
        conversation_2 = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "How many objects can you identify in this image?"},
                    {"type": "image"},
                ],
            },
        ]
        
        prompt_2 = processor.apply_chat_template(conversation_2, add_generation_prompt=True)
        inputs_2 = processor(images=image, text=prompt_2, return_tensors="pt").to("cuda")
        
        logger.info("Running second inference test...")
        with torch.no_grad():
            outputs_2 = model.generate(
                **inputs_2,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        response_2 = processor.decode(outputs_2[0], skip_special_tokens=True)
        assistant_response_2 = response_2.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response_2 else response_2
        logger.info(f"Second response: {assistant_response_2}")
        
        del model, processor
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("LLaVA-NeXT model test successful!")
        return True
    
    except Exception as e:
        logger.error(f"Error loading LLaVA-NeXT model: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test loading LLaVA-NeXT VLM model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava-hf/llava-v1.6-vicuna-13b-hf",
        help="Name of the LLaVA-NeXT model to test",
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
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires GPU support.")
        return
    
    logger.info(f"Found {torch.cuda.device_count()} GPUs")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    configs = [
        {"load_in_4bit": False, "load_in_8bit": False, "use_flash_attn": False},
        {"load_in_4bit": False, "load_in_8bit": True, "use_flash_attn": False},
        {"load_in_4bit": True, "load_in_8bit": False, "use_flash_attn": False},
        {"load_in_4bit": True, "load_in_8bit": False, "use_flash_attn": args.use_flash_attn},
    ]
    
    if args.use_4bit or args.use_8bit:
        configs = [
            {
                "load_in_4bit": args.use_4bit,
                "load_in_8bit": args.use_8bit,
                "use_flash_attn": args.use_flash_attn
            }
        ]
    
    success = False
    for config in configs:
        try:
            logger.info(f"\nTesting configuration: {config}")
            if test_llava_next_loading(args.model_name, **config):
                logger.info(f"Successfully loaded LLaVA-NeXT with configuration: {config}")
                success = True
                break
        except RuntimeError as e:
            logger.error(f"Failed to load model with configuration {config}: {e}")
    
    if not success:
        logger.error("Failed to load LLaVA-NeXT model with any configuration")
    else:
        logger.info("Found a working configuration for LLaVA-NeXT on your hardware!")


if __name__ == "__main__":
    main()