#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for Gemma-3 Vision-Language Model using Hugging Face Transformers.
This script supports multimodal training with image-text pairs for models like:
- google/gemma-3-4b-it
- google/gemma-3-12b-it
- google/gemma-3-27b-it

Features:
- Support for image-text pairs from datasets like Flickr8k
- Uses native Gemma3ForConditionalGeneration for multimodal processing
- Quantization support (4-bit or 8-bit) to reduce GPU memory requirements
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- DeepSpeed ZeRO-3 integration for distributed training across multiple GPUs
- Gradient checkpointing for memory efficiency

Example usage:
    python train_gemma3_vlm.py \
        --model_name_or_path="google/gemma-3-12b-it" \
        --dataset_path="/path/to/flickr8k.json" \
        --image_dir="/path/to/flickr8k/images" \
        --output_dir="./outputs/gemma3-12b-vlm-finetuned" \
        --num_train_epochs=3 \
        --per_device_train_batch_size=1 \
        --gradient_accumulation_steps=8 \
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
import json
import logging
import torch
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    set_seed,
    HfArgumentParser
)
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import deepspeed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="google/gemma-3-12b-it",
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
        default=None, metadata={"help": "Path to local dataset JSON file"}
    )
    image_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing images"}
    )
    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing"}
    )

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="./outputs/gemma3-vlm-finetuned",
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
        default=8,
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
        default=100,
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
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing to save memory"}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use: adamw_torch, adamw_8bit"}
    )
    report_to: List[str] = field(
        default_factory=lambda: ["none"],
        metadata={"help": "The list of integrations to report the results and logs to"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove columns not used by the model"}
    )

class Gemma3DataCollator:
    """Data collator for Gemma-3 multimodal inputs"""
    
    def __init__(self, processor):
        self.processor = processor
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Prepare messages for each example
        messages_list = []
        images_list = []
        
        for feature in features:
            # Create conversation with user prompt and assistant response
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": feature["image"]},
                        {"type": "text", "text": feature["text"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": feature.get("caption", "")}]
                }
            ]
            
            messages_list.append(messages)
            images_list.append(feature["image"])
        
        # Apply chat template and process
        texts = [self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) 
                 for msgs in messages_list]
        
        # Process text and images together
        batch = self.processor(
            text=texts,
            images=images_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # For training, we need labels
        # Set labels to -100 for pad tokens so they're ignored in loss calculation
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

def load_flickr8k_dataset(dataset_path: str, image_dir: str) -> Dataset:
    """Load Flickr8k dataset from local files"""
    logger.info(f"Loading Flickr8k dataset from {dataset_path}")
    
    # Load JSON file
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Process each item
    processed_data = []
    image_dir_path = Path(image_dir)
    
    for item in data:
        # Handle different possible field names
        image_filename = item.get('image', item.get('image_path', ''))
        
        # Remove 'images/' prefix if it exists since we're already in the images directory
        if image_filename.startswith('images/'):
            image_filename = image_filename.replace('images/', '', 1)
        
        # Construct full image path
        if not image_filename.startswith('/'):
            image_path = image_dir_path / image_filename
        else:
            image_path = Path(image_filename)
        
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue
        
        # Get instruction and output/caption
        instruction = item.get('instruction', 'Describe this image in detail.')
        caption = item.get('output', item.get('caption', item.get('captions', '')))
        
        # Handle multiple captions if provided as list
        if isinstance(caption, list):
            # Create one example per caption
            for cap in caption:
                processed_data.append({
                    'image_path': str(image_path),
                    'text': instruction,
                    'caption': cap
                })
        else:
            # Single caption
            processed_data.append({
                'image_path': str(image_path),
                'text': instruction,
                'caption': caption
            })
    
    logger.info(f"Loaded {len(processed_data)} image-caption pairs")
    
    # Create dataset
    dataset = Dataset.from_list(processed_data)
    
    # Load images
    def load_image(example):
        example['image'] = Image.open(example['image_path']).convert('RGB')
        return example
    
    dataset = dataset.map(load_image, num_proc=1)
    
    return dataset

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    # Load processor (handles both text and images)
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    
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
    
    # Load model
    logger.info(f"Loading model {model_args.model_name_or_path}")
    
    # Check if DeepSpeed is being used
    if training_args.deepspeed:
        device_map = None
    else:
        device_map = "auto"
    
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map
    )
    
    # Enable gradient checkpointing if specified
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    
    # Prepare model for k-bit training if using quantization
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA if specified
    if model_args.use_lora:
        logger.info("Using LoRA for fine-tuning")
        
        # Find target modules for LoRA
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",  # Use CAUSAL_LM for generative models
            target_modules=target_modules
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Load dataset
    if data_args.dataset_name:
        # Load from Hugging Face Hub
        logger.info(f"Loading dataset {data_args.dataset_name} from Hugging Face Hub")
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        train_dataset = dataset["train"]
    elif data_args.dataset_path and data_args.image_dir:
        # Load local Flickr8k dataset
        train_dataset = load_flickr8k_dataset(data_args.dataset_path, data_args.image_dir)
    else:
        raise ValueError("Either dataset_name or (dataset_path and image_dir) must be provided")
    
    # Create data collator
    data_collator = Gemma3DataCollator(processor)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )
    
    # Training
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Save the model
    if model_args.use_lora:
        # Save only the LoRA adapter weights
        model.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model()
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Save processor
    processor.save_pretrained(training_args.output_dir)
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()