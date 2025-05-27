#!/usr/bin/env python3
"""
Training script for Idefics3 Vision-Language Models
Supports Idefics3-8B-Llama3 with DeepSpeed and multi-GPU training
"""

import os
import sys
import json
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from pathlib import Path

from transformers import (
    AutoProcessor,
    Idefics3ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    set_seed,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
from PIL import Image
import deepspeed

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
        default="HuggingFaceM4/Idefics3-8B-Llama3",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning"}
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
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
    """Arguments pertaining to the data used for training and evaluation"""
    dataset_path: str = field(
        metadata={"help": "Path to the dataset JSON file"}
    )
    image_dir: str = field(
        metadata={"help": "Directory containing the images"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for tokenization"}
    )

@dataclass
class VLMTrainingArguments(TrainingArguments):
    """Training arguments specific to Vision-Language Model training"""
    optim: str = field(default="paged_adamw_32bit")
    gradient_checkpointing: bool = field(default=True)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    warmup_steps: int = field(default=100)
    num_train_epochs: float = field(default=3)
    learning_rate: float = field(default=2e-5)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=3)
    evaluation_strategy: str = field(default="no")
    remove_unused_columns: bool = field(default=False)
    push_to_hub: bool = field(default=False)
    report_to: str = field(default="tensorboard")
    deepspeed: Optional[str] = field(default=None)

class VLMDataCollator:
    """Data collator for Vision-Language Models"""
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, features):
        # Prepare texts with Idefics3 format
        texts = []
        images = []
        
        for feature in features:
            # Idefics3 uses a specific format for image-text pairs
            text = f"<image>User: Describe this image.\nAssistant: {feature['text']}"
            texts.append(text)
            images.append(feature["image"])
        
        # Process batch
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Handle labels - mask padding tokens
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

def load_dataset(data_path: str, image_dir: str) -> List[Dict]:
    """Load dataset from JSON file"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Process data
    processed_data = []
    for item in data:
        image_filename = item.get('image', '')
        
        # Handle image path
        if image_filename.startswith('images/'):
            image_filename = image_filename.replace('images/', '', 1)
        
        image_path = os.path.join(image_dir, image_filename)
        
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
                processed_data.append({
                    "image": image,
                    "text": item.get('caption', item.get('text', ''))
                })
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
        else:
            logger.warning(f"Image not found: {image_path}")
    
    return processed_data

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, VLMTrainingArguments))
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
    
    # Load dataset
    logger.info(f"Loading dataset from {data_args.dataset_path}")
    dataset_list = load_dataset(data_args.dataset_path, data_args.image_dir)
    
    if not dataset_list:
        raise ValueError("No valid data found in the dataset!")
    
    logger.info(f"Loaded {len(dataset_list)} samples")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(dataset_list)
    
    # Set up quantization config if needed
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
    
    # Load processor and model
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    
    # Ensure we have a pad token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model with appropriate config
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if not (model_args.use_4bit or model_args.use_8bit) else None,
        device_map="auto" if not training_args.deepspeed else None,
    )
    
    # Prepare model for training
    if model_args.use_4bit or model_args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Set up LoRA if enabled
    if model_args.use_lora:
        logger.info("Setting up LoRA")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Create data collator
    data_collator = VLMDataCollator(processor, max_length=data_args.max_seq_length)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save the model
    logger.info("Saving model...")
    trainer.save_model()
    
    # Save training metrics
    if trainer.is_world_process_zero():
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # Save processor
        processor.save_pretrained(training_args.output_dir)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()