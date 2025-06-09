#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified test script to verify basic model loading and training setup.
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

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

def main():
    # Model settings
    model_name = "google/gemma-2-2b"
    train_file = "/home/user/llm_trainer/datasets/test_dataset/test_data.json"
    
    # Device settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=True)
    
    # Make sure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set padding token to EOS token")
    
    # Prepare 4-bit quantization config
    logger.info("Using 4-bit quantization")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    logger.info(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        device_map=0,  # Use first GPU only
        trust_remote_code=True,
    )
    
    # Apply LoRA
    logger.info("Setting up LoRA for fine-tuning")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load dataset
    logger.info(f"Loading dataset from {train_file}")
    datasets = load_dataset("json", data_files={"train": train_file})
    logger.info(f"Dataset loaded: {datasets}")
    
    # Check that we can access the examples
    logger.info(f"First example: {datasets['train'][0]}")
    
    # Simple tokenization test
    logger.info("Testing tokenization of first example")
    example = tokenizer(datasets['train'][0]['text'], return_tensors="pt", truncation=True, max_length=512)
    logger.info(f"Tokenized shape: {example['input_ids'].shape}")
    
    logger.info("Test passed successfully - model, tokenizer, and dataset are all configured correctly")

if __name__ == "__main__":
    main()