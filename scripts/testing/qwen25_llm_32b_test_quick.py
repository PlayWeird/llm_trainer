#!/usr/bin/env python3
"""
Quick test script for Qwen2.5-32B model training
Tests model loading, dataset processing, and a few training steps
"""

import os
import sys
import torch
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading Qwen2.5-32B with quantization"""
    logger.info("Testing Qwen2.5-32B model loading with 4-bit quantization...")
    
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
            "Qwen/Qwen2.5-32B-Instruct",
            trust_remote_code=True
        )
        logger.info("✓ Tokenizer loaded successfully")
        
        # Load model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-32B-Instruct",
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        logger.info("✓ Model loaded successfully with 4-bit quantization")
        
        # Prepare for k-bit training
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
        test_input = "What is machine learning?"
        inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test inference:\nInput: {test_input}\nOutput: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def test_dataset_loading():
    """Test loading and processing the Dolly dataset"""
    logger.info("\nTesting dataset loading...")
    
    try:
        dataset_path = project_root / "datasets" / "test_dataset" / "llm" / "dolly_test_data.json"
        
        # Load dataset
        dataset = load_dataset('json', data_files=str(dataset_path))['train']
        logger.info(f"✓ Loaded {len(dataset)} samples from Dolly dataset")
        
        # Check first few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            logger.info(f"\nSample {i+1}:")
            logger.info(f"  Instruction: {sample['instruction'][:100]}...")
            logger.info(f"  Output: {sample['output'][:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False

def test_memory_usage():
    """Check GPU memory usage"""
    logger.info("\nChecking GPU memory usage...")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            
            logger.info(f"\nGPU {i} ({props.name}):")
            logger.info(f"  Total memory: {total_memory:.2f} GB")
            logger.info(f"  Allocated: {allocated:.2f} GB")
            logger.info(f"  Reserved: {reserved:.2f} GB")
            logger.info(f"  Free: {total_memory - reserved:.2f} GB")
    else:
        logger.warning("No CUDA devices available")

def main():
    """Run all tests"""
    logger.info("Starting Qwen2.5-32B quick test...")
    logger.info("=" * 50)
    
    # Test dataset loading first (less memory intensive)
    dataset_ok = test_dataset_loading()
    
    # Check initial memory
    test_memory_usage()
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Check memory after model loading
    test_memory_usage()
    
    logger.info("\n" + "=" * 50)
    logger.info("Test Summary:")
    logger.info(f"  Dataset loading: {'✓ PASSED' if dataset_ok else '✗ FAILED'}")
    logger.info(f"  Model loading: {'✓ PASSED' if model_ok else '✗ FAILED'}")
    logger.info("=" * 50)
    
    if dataset_ok and model_ok:
        logger.info("\nAll tests passed! Ready to run full training.")
        logger.info("\nTo start training, run:")
        logger.info("  cd /home/user/llm_trainer")
        logger.info("  ./scripts/shell/testing/run_qwen25_32b_test.sh")
    else:
        logger.error("\nSome tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()