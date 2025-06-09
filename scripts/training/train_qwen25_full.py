#!/usr/bin/env python3
"""
Training script for Qwen2.5 Language Models (Full Precision)
Supports multi-GPU training with DeepSpeed
"""

import os
import sys
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union
from pathlib import Path

# Patch for missing torch.get_default_device
if not hasattr(torch, 'get_default_device'):
    def get_default_device():
        return None
    torch.get_default_device = get_default_device

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from transformers.integrations import is_deepspeed_zero3_enabled
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
import deepspeed

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-14B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of module names to apply LoRA to. If None, will use model defaults."}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading model"}
    )
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8-bit quantization"}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to the data used for training and evaluation"""
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local dataset file (JSON or JSONL)"}
    )
    train_split: str = field(
        default="train",
        metadata={"help": "The name of the training data split"}
    )
    validation_split: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the validation data split"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for tokenization"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for preprocessing"}
    )

def format_dolly_dataset(example):
    """Format Dolly-style dataset into Qwen chat format"""
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    response = example.get("response", "")
    
    # Build the prompt
    if context:
        prompt = f"{instruction}\n\nContext: {context}"
    else:
        prompt = instruction
    
    # Format as Qwen chat
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    
    return {"messages": messages}

def format_alpaca_dataset(example):
    """Format Alpaca-style dataset into Qwen chat format"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    
    # Build the prompt
    if input_text:
        prompt = f"{instruction}\n\nInput: {input_text}"
    else:
        prompt = instruction
    
    # Format as Qwen chat
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": output}
    ]
    
    return {"messages": messages}

def format_simple_dataset(example):
    """Format simple instruction-response dataset into Qwen chat format"""
    instruction = example.get("instruction", example.get("text", ""))
    response = example.get("response", example.get("output", ""))
    
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ]
    
    return {"messages": messages}

def preprocess_function(examples, tokenizer, max_length):
    """Preprocess the data by tokenizing."""
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for messages in examples["messages"]:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Tokenize
        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = encoded["input_ids"].clone()
        # Mask padding tokens
        labels[labels == tokenizer.pad_token_id] = -100
        
        model_inputs["input_ids"].append(encoded["input_ids"].squeeze())
        model_inputs["attention_mask"].append(encoded["attention_mask"].squeeze())
        model_inputs["labels"].append(labels.squeeze())
    
    return model_inputs

def load_and_prepare_dataset(data_args, tokenizer):
    """Load and prepare the dataset for training"""
    # Load dataset
    if data_args.dataset_name:
        # Load from HuggingFace
        logger.info(f"Loading dataset from HuggingFace: {data_args.dataset_name}")
        raw_dataset = load_dataset(data_args.dataset_name)
    elif data_args.dataset_path:
        # Load from local file
        logger.info(f"Loading dataset from local file: {data_args.dataset_path}")
        if data_args.dataset_path.endswith('.json'):
            raw_dataset = load_dataset('json', data_files=data_args.dataset_path)
        elif data_args.dataset_path.endswith('.jsonl'):
            raw_dataset = load_dataset('json', data_files=data_args.dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {data_args.dataset_path}")
    else:
        raise ValueError("Either dataset_name or dataset_path must be specified")
    
    # Get train and validation splits
    if data_args.train_split in raw_dataset:
        train_dataset = raw_dataset[data_args.train_split]
    else:
        train_dataset = raw_dataset["train"]
    
    eval_dataset = None
    if data_args.validation_split and data_args.validation_split in raw_dataset:
        eval_dataset = raw_dataset[data_args.validation_split]
    
    # Detect and apply appropriate formatting
    sample = train_dataset[0]
    if "instruction" in sample and "response" in sample:
        logger.info("Detected Dolly-style dataset format")
        format_func = format_dolly_dataset
    elif "instruction" in sample and "output" in sample:
        logger.info("Detected Alpaca-style dataset format")
        format_func = format_alpaca_dataset
    elif "messages" in sample:
        logger.info("Dataset already in chat format")
        format_func = None
    else:
        logger.info("Using simple instruction-response format")
        format_func = format_simple_dataset
    
    # Apply formatting if needed
    if format_func:
        train_dataset = train_dataset.map(format_func, remove_columns=train_dataset.column_names)
        if eval_dataset:
            eval_dataset = eval_dataset.map(format_func, remove_columns=eval_dataset.column_names)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, data_args.max_seq_length),
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: preprocess_function(x, tokenizer, data_args.max_seq_length),
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval dataset"
        )
    
    return train_dataset, eval_dataset

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True
    )
    
    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(data_args, tokenizer)
    logger.info(f"Loaded {len(train_dataset)} training samples")
    if eval_dataset:
        logger.info(f"Loaded {len(eval_dataset)} evaluation samples")
    
    # Set up quantization if needed
    bnb_config = None
    if model_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif model_args.use_8bit:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    
    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    
    # Determine device map strategy
    if training_args.deepspeed:
        device_map = None  # Let DeepSpeed handle device placement
        logger.info("Using DeepSpeed for distributed training")
    else:
        device_map = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device_map=device_map,
        quantization_config=bnb_config,
    )
    
    # Prepare model for training
    if model_args.use_4bit or model_args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing if specified
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Set up LoRA if enabled
    if model_args.use_lora:
        logger.info("Setting up LoRA for PEFT")
        
        # Default target modules for Qwen models
        target_modules = model_args.lora_target_modules
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # Save the model
    logger.info("Saving model...")
    trainer.save_model()
    
    # Save training metrics
    if trainer.is_world_process_zero():
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # Save tokenizer
        tokenizer.save_pretrained(training_args.output_dir)
    
    # Evaluate if we have eval dataset
    if eval_dataset and training_args.do_eval:
        logger.info("Running evaluation...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()