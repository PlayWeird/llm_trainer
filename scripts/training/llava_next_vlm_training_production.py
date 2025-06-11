#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaVA-NeXT VLM Production Training Script

This script implements production-ready training for LLaVA-NeXT vision-language models
using DeepSpeed ZeRO-3 for distributed training across multiple GPUs.

Features:
- DeepSpeed ZeRO-3 integration for multi-GPU training
- 4-bit quantization support for memory efficiency
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- Support for multiple LLaVA-NeXT model variants
- Comprehensive logging and monitoring
- Memory optimization techniques

Supported models:
- llava-hf/llava-v1.6-vicuna-7b-hf
- llava-hf/llava-v1.6-vicuna-13b-hf  
- llava-hf/llava-v1.6-34b-hf
- llava-hf/llama3-llava-next-8b-hf

Example usage:
    deepspeed llava_next_vlm_training_production.py \
        --deepspeed configs/training/ds_config_zero3_vlm.json \
        --model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf \
        --dataset_path datasets/test_dataset/vlm/flickr8k \
        --output_dir outputs/llava-next-13b-finetuned
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_dataset
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.vlm_data_utils import BaseVLMDataCollator, GenericVLMDatasetLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_model_and_processor(args):
    """
    Load and configure the LLaVA-NeXT model and processor.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        tuple: (model, processor) configured for training
    """
    logger.info(f"Loading LLaVA-NeXT model: {args.model_name_or_path}")
    
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Using 4-bit quantization")
    
    device_map = None if args.deepspeed else "auto"
    if args.deepspeed:
        logger.info("Using DeepSpeed for distributed training")
    else:
        logger.info("Using automatic device mapping")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    if args.use_flash_attn:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")
        except ImportError:
            logger.warning("Flash Attention not available, using standard attention")
    
    processor = LlavaNextProcessor.from_pretrained(args.model_name_or_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )
    
    if args.use_lora:
        logger.info("Applying LoRA configuration")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, processor


def load_dataset_for_training(args, processor):
    """
    Load and prepare the dataset for training.
    
    Args:
        args: Parsed command line arguments
        processor: LLaVA-NeXT processor for tokenization
        
    Returns:
        Dataset: Prepared training dataset
    """
    logger.info(f"Loading dataset from: {args.dataset_path}")
    
    # Use the generic VLM dataset loader
    if args.dataset_path.endswith('.json'):
        # Direct JSON file - use with image directory
        image_dir = os.path.join(os.path.dirname(args.dataset_path), 'images')
        if not os.path.exists(image_dir):
            image_dir = os.path.join(args.dataset_path, 'images')
        dataset = GenericVLMDatasetLoader.load(
            dataset_path=args.dataset_path,
            image_dir=image_dir,
            num_workers=1
        )
    else:
        # Directory path - look for dataset file
        dataset_file = os.path.join(args.dataset_path, 'flickr8k_test_data.json')
        image_dir = os.path.join(args.dataset_path, 'images')
        dataset = GenericVLMDatasetLoader.load(
            dataset_path=dataset_file,
            image_dir=image_dir,
            num_workers=1
        )
    
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    
    # Limit to max training samples
    if len(dataset) > args.max_train_samples:
        dataset = dataset.select(range(args.max_train_samples))
        logger.info(f"Limited dataset to {args.max_train_samples} examples")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train LLaVA-NeXT VLM with DeepSpeed")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-v1.6-vicuna-13b-hf")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    
    # Training arguments
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_train_samples", type=int, default=1000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    
    # Model configuration arguments
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # System arguments
    parser.add_argument("--deepspeed", type=str, help="DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Log configuration
    logger.info("=" * 50)
    logger.info("LLaVA-NeXT VLM Training Configuration")
    logger.info("=" * 50)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    logger.info(f"Batch size: {args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Epochs: {args.num_train_epochs}")
    logger.info(f"Using 4-bit: {args.use_4bit}")
    logger.info(f"Using LoRA: {args.use_lora}")
    logger.info(f"Using Flash Attention: {args.use_flash_attn}")
    if args.deepspeed:
        logger.info(f"DeepSpeed config: {args.deepspeed}")
    logger.info("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup model and processor
    model, processor = setup_model_and_processor(args)
    
    # Load dataset
    train_dataset = load_dataset_for_training(args, processor)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=args.logging_steps,
        optim="adamw_torch",
        logging_dir=f"{args.output_dir}/logs",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
        group_by_length=False,
        report_to="tensorboard",
        run_name=f"llava-next-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
    )
    
    # Create data collator
    data_collator = BaseVLMDataCollator(processor, max_length=args.max_seq_length)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    # Save training info
    training_info = {
        "model_name": args.model_name_or_path,
        "dataset_path": args.dataset_path,
        "training_args": vars(args),
        "final_output_dir": args.output_dir,
        "training_completed": datetime.now().isoformat(),
    }
    
    with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"Training completed! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()