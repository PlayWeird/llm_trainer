#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for Gemma 3 27B model using Hugging Face Transformers.
This script sets up distributed training using DeepSpeed for efficient multi-GPU usage.
"""

import os
import argparse
import logging
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    HfArgumentParser
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import deepspeed
from dataclasses import dataclass, field
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="google/gemma-3-27b-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8-bit quantization"}
    )

@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (from the HF hub)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use"}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file"}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation data file"}
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="./outputs/gemma-3-27b-finetuned",
        metadata={"help": "The output directory where model predictions and checkpoints will be written"}
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Overwrite the content of the output directory"}
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for training"}
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for AdamW optimizer"}
    )
    warmup_steps: int = field(
        default=500,
        metadata={"help": "Linear warmup over warmup_steps"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps"}
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints"}
    )
    deepspeed: Optional[str] = field(
        default="configs/training/ds_config_zero3.json",
        metadata={"help": "Path to deepspeed config file"}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"}
    )
    report_to: List[str] = field(
        default_factory=lambda: ["tensorboard", "wandb"],
        metadata={"help": "The list of integrations to report the results and logs to"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    # Make sure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization if specified
    load_in_4bit = model_args.use_4bit
    load_in_8bit = model_args.use_8bit
    
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Apply LoRA if specified
    if model_args.use_lora:
        logger.info("Using LoRA for fine-tuning")
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Load dataset
    if data_args.dataset_name is not None:
        # Load from hub
        datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )
    else:
        # Load from local files
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        
        extension = data_args.train_file.split(".")[-1] if data_args.train_file else "json"
        datasets = load_dataset(extension, data_files=data_files)
    
    # Preprocessing function
    def preprocess_function(examples):
        # Tokenize the texts
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_args.max_seq_length,
            padding="max_length",
        )
    
    # Process datasets
    tokenized_datasets = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    train_result = trainer.train()
    
    # Save the model
    trainer.save_model()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("Training completed")


if __name__ == "__main__":
    main()