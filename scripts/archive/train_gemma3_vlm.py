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
import sys
import logging
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent.parent))

from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    HfArgumentParser,
)

from utils import (
    # Data utilities
    GenericVLMDatasetLoader,
    create_data_collator,
    
    # Configuration classes
    ModelWithLoRAArguments,
    VLMDataArguments,
    TrainingArguments,
    
    # Model utilities
    get_quantization_config,
    load_model_for_training,
    setup_lora,
    save_model_and_processor,
    log_gpu_memory_usage,
    estimate_model_size,
    
    # Training utilities
    setup_training_environment,
    create_trainer,
    save_training_info,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    # Parse arguments
    parser = HfArgumentParser((ModelWithLoRAArguments, VLMDataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup training environment
    setup_training_environment(
        seed=training_args.seed,
        tf32=training_args.tf32,
    )
    
    # Log configuration
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Output dir: {training_args.output_dir}")
    logger.info(f"Training epochs: {training_args.num_train_epochs}")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    
    # Load processor (handles both text and images)
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    # Get quantization config
    quantization_config = get_quantization_config(
        use_4bit=model_args.use_4bit,
        use_8bit=model_args.use_8bit,
        bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=model_args.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
    )
    
    # Check if DeepSpeed is being used
    if training_args.deepspeed:
        device_map = None
        logger.info("Using DeepSpeed, setting device_map to None")
    else:
        device_map = model_args.device_map
    
    # Load model
    logger.info(f"Loading model {model_args.model_name_or_path}")
    model = load_model_for_training(
        model_name_or_path=model_args.model_name_or_path,
        model_class=Gemma3ForConditionalGeneration,
        quantization_config=quantization_config,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        device_map=device_map,
        gradient_checkpointing=training_args.gradient_checkpointing,
    )
    
    # Log model size
    model_info = estimate_model_size(model)
    logger.info(f"Model size: {model_info['total_size_gb']:.2f}GB")
    logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    
    # Apply LoRA if specified
    if model_args.use_lora:
        model = setup_lora(
            model=model,
            lora_r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            lora_target_modules=model_args.lora_target_modules,
            lora_bias=model_args.lora_bias,
        )
        
        # Log updated model info
        model_info = estimate_model_size(model)
        logger.info(f"Trainable parameters: {model_info['trainable_parameters']:,}")
        logger.info(f"Trainable size: {model_info['trainable_size_gb']:.2f}GB")
        logger.info(f"Trainable percentage: {model_info['trainable_percentage']:.2f}%")
    
    # Log GPU memory usage
    log_gpu_memory_usage()
    
    # Load dataset
    logger.info("Loading dataset...")
    train_dataset = GenericVLMDatasetLoader.load(
        dataset_path=data_args.dataset_path,
        image_dir=data_args.image_dir,
        dataset_name=data_args.dataset_name,
        dataset_config=data_args.dataset_config_name,
        num_workers=data_args.preprocessing_num_workers,
    )
    
    logger.info(f"Dataset size: {len(train_dataset)}")
    
    # Create data collator
    data_collator = create_data_collator(
        processor=processor,
        model_type="gemma",
        max_length=data_args.max_seq_length,
    )
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=processor,  # Processor acts as tokenizer
        use_memory_efficient_trainer=True,
    )
    
    # Training
    logger.info("Starting training...")
    logger.info(f"Number of training steps: {trainer.state.max_steps}")
    
    train_result = trainer.train()
    
    # Save the model
    logger.info("Saving model...")
    if model_args.use_lora:
        # Save only the LoRA adapter weights
        save_model_and_processor(
            model=model,
            processor=processor,
            output_dir=training_args.output_dir,
            is_lora=True,
            save_full_model=False,
        )
    else:
        # Save full model
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Save training info
    save_training_info(
        output_dir=training_args.output_dir,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        metrics=metrics,
    )
    
    # Log final GPU memory usage
    logger.info("Final GPU memory usage:")
    log_gpu_memory_usage()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()