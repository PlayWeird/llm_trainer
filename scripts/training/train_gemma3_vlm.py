#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for Gemma 3 27B VLM (Vision-Language Model) using Hugging Face Transformers.
This script extends train_gemma3_27b.py to support multimodal training with image-text pairs.

Features:
- Support for image-text pairs from datasets like Flickr8k
- Multimodal preprocessing with prepare_multimodal_dataset from utils
- Quantization support (4-bit or 8-bit) to reduce GPU memory requirements
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- DeepSpeed ZeRO-3 integration for distributed training across multiple GPUs

Example usage:
    python train_gemma3_vlm.py \
        --model_name_or_path="google/gemma-3-27b-it" \
        --dataset_path="/path/to/data.json" \
        --image_dir="/path/to/images" \
        --output_dir="./outputs/gemma3-27b-vlm-finetuned" \
        --num_train_epochs=3 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps=16 \
        --learning_rate=2e-5

Requirements:
    - PyTorch 2.0+
    - Transformers 4.30+
    - DeepSpeed
    - Accelerate
    - PEFT
    - BitsAndBytes
    - PIL for image processing
"""

import os
import argparse
import logging
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    set_seed,
    HfArgumentParser
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
import deepspeed
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import sys
import json
from PIL import Image

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_preprocessing import prepare_multimodal_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="google/gemma-3-27b-it",
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
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (JSON)"}
    )
    image_dir: str = field(
        default="datasets/test_dataset/vlm/flickr8k/images",
        metadata={"help": "Directory containing the image files"}
    )
    validation_split: float = field(
        default=0.1,
        metadata={"help": "Percentage of training data to use as validation"}
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization"}
    )
    prompt_template: str = field(
        default="<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
        metadata={"help": "Template for formatting prompts"}
    )
    response_template: str = field(
        default="{output}<end_of_turn>",
        metadata={"help": "Template for formatting responses"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="./outputs/gemma-3-27b-vlm-finetuned",
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
        default=None,
        metadata={"help": "Path to deepspeed config file"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 (mixed) precision instead of 32-bit"}
    )
    report_to: List[str] = field(
        default_factory=lambda: ["tensorboard", "wandb"],
        metadata={"help": "The list of integrations to report the results and logs to"}
    )

# Custom data collator for multimodal inputs
class MultimodalDataCollator:
    def __init__(self, tokenizer, processor=None):
        self.tokenizer = tokenizer
        self.processor = processor
        
    def __call__(self, examples):
        batch = {}
        
        # Process text inputs
        text_inputs = [example["text"] for example in examples]
        tokenized = self.tokenizer(
            text_inputs,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )
        
        batch.update(tokenized)
        
        # Process image inputs if present
        if all("image" in example and example.get("has_image", False) for example in examples):
            if self.processor:
                # If we have a processor, use it to process images
                images = [example["image"] for example in examples]
                if isinstance(images[0], str):
                    # Load images from file paths
                    images = [Image.open(img_path).convert("RGB") for img_path in images]
                
                image_features = self.processor(images=images, return_tensors="pt")
                batch["pixel_values"] = image_features["pixel_values"]
            else:
                # Otherwise just pass image paths
                batch["image_paths"] = [example["image"] for example in examples]
        
        # Create labels (shifted input_ids)
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
            
        return batch

def load_and_process_data(data_args, tokenizer, processor=None):
    """Load and preprocess the dataset."""
    logger.info("Loading and processing data")
    
    if data_args.dataset_name:
        # Load from Hugging Face Hub
        logger.info(f"Loading dataset {data_args.dataset_name} from Hugging Face Hub")
        datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )
        
        # Check if this is a multimodal dataset with 'image' and 'text'/'caption' fields
        # and convert to our desired format
        if "train" in datasets:
            train_data = datasets["train"]
            
            # Check if we need to convert the dataset format
            if "image" in train_data.features and ("caption" in train_data.features or "text" in train_data.features):
                logger.info("Converting Hugging Face dataset to multimodal format")
                
                # Determine the caption field
                caption_field = "caption" if "caption" in train_data.features else "text"
                
                # Convert to our expected format
                def convert_format(example):
                    return {
                        "instruction": "Describe this image in detail.",
                        "output": example[caption_field],
                        "image": example["image"],
                    }
                
                datasets = datasets.map(convert_format)
    
    elif data_args.dataset_path:
        # Load from local file
        logger.info(f"Loading dataset from {data_args.dataset_path}")
        
        with open(data_args.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create dataset
        dataset = Dataset.from_list(data)
        
        # Split into train and validation
        if data_args.validation_split > 0:
            splits = dataset.train_test_split(test_size=data_args.validation_split)
            datasets = {
                "train": splits["train"],
                "validation": splits["test"]
            }
        else:
            datasets = {"train": dataset}
    
    else:
        raise ValueError("You must specify either dataset_name or dataset_path")
    
    # Process the dataset into the multimodal format
    processed_datasets = {}
    
    for split, dataset in datasets.items():
        # Use our utility function to prepare the multimodal dataset
        processed_datasets[split] = prepare_multimodal_dataset(
            data_path=data_args.dataset_path,
            image_dir=data_args.image_dir,
            tokenizer=tokenizer,
            processor=processor,
            max_seq_length=data_args.max_seq_length
        )
        
    return processed_datasets

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Load tokenizer and processor
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    # Make sure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set padding token to EOS token")
    
    # Try to load the processor for image handling
    processor = None
    try:
        logger.info(f"Loading processor from {model_args.model_name_or_path}")
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        logger.info("Processor loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load processor: {e}. Will use basic PIL image handling.")

    # Load model with quantization if specified
    load_in_4bit = model_args.use_4bit
    load_in_8bit = model_args.use_8bit
    
    quantization_config = None
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig
        
        logger.info(f"Using {'4-bit' if load_in_4bit else '8-bit'} quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Model loading keywords
    model_kwargs = {
        "quantization_config": quantization_config if (load_in_4bit or load_in_8bit) else None,
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "use_safetensors": True,  # Use safetensors to avoid PyTorch vulnerability
    }
    
    # Handle device mapping based on GPU count
    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} GPUs")
    
    if num_gpus > 1:
        logger.info("Using balanced device mapping for multiple GPUs")
        model_kwargs["device_map"] = "auto"
    else:
        logger.info("Using single GPU")
        model_kwargs["device_map"] = 0  # Use first GPU explicitly
    
    # Filter out None values
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}
    
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    logger.info(f"Model loading parameters: {model_kwargs}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
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
    
    # Load and process the dataset
    datasets = load_and_process_data(data_args, tokenizer, processor)
    
    # Initialize data collator
    data_collator = MultimodalDataCollator(tokenizer, processor)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    logger.info("Starting training")
    train_result = trainer.train()
    
    # Save the model
    logger.info("Saving model")
    trainer.save_model()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info("Training completed")


if __name__ == "__main__":
    main()