#!/usr/bin/env python3
"""
Quick test script for Qwen2.5-14B model
Validates model loading and basic inference
"""

import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_14b_model():
    """Test Qwen2.5-14B model loading and inference"""
    logger.info("Testing Qwen2.5-14B model...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-14B-Instruct",
            trust_remote_code=True
        )
        logger.info("✓ Tokenizer loaded")
        
        # Check if model is already cached
        from huggingface_hub import snapshot_download
        try:
            cache_dir = snapshot_download(
                "Qwen/Qwen2.5-14B-Instruct",
                local_files_only=True,
                cache_dir=None
            )
            logger.info(f"✓ Model found in cache: {cache_dir}")
        except:
            logger.info("Model not in cache, will download...")
        
        # Load model
        logger.info("Loading model with 4-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-14B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        logger.info("✓ Model loaded successfully")
        
        # Prepare for training
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Test inference
        test_prompts = [
            "What is machine learning?",
            "Write a Python function to calculate fibonacci numbers:",
            "Explain quantum computing in simple terms."
        ]
        
        for prompt in test_prompts:
            logger.info(f"\nTesting prompt: {prompt}")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Response: {response[:200]}...")
        
        # Check memory usage
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"\nGPU {i}: Allocated: {mem_allocated:.2f}GB, Reserved: {mem_reserved:.2f}GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

def main():
    """Run the test"""
    logger.info("=" * 60)
    logger.info("Qwen2.5-14B Quick Test")
    logger.info("=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.error("No CUDA devices available!")
        return
    
    # Run test
    success = test_14b_model()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✅ All tests passed! Ready for training.")
        logger.info("\nTo run training:")
        logger.info("  cd /home/user/llm_trainer")
        logger.info("  ./scripts/shell/testing/run_qwen25_14b_test.sh")
    else:
        logger.info("❌ Tests failed. Check errors above.")

if __name__ == "__main__":
    main()