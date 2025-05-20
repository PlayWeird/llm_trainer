#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic LoRA fine-tuning script for Gemma that uses a manual training loop,
bypassing the Transformers Trainer to avoid accelerator issues.
"""

import os
import json
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import argparse
from tqdm import tqdm

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

class TextDataset(Dataset):
    """Simple dataset for text samples."""
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return len(self.encodings["input_ids"])
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Basic LoRA fine-tuning for Gemma models")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", 
                        help="Model name or path")
    parser.add_argument("--train_file", type=str, default="/home/user/llm_trainer/datasets/test_dataset/test_data.json", 
                        help="Path to training data file")
    parser.add_argument("--output_dir", type=str, default="/home/user/llm_trainer/outputs/gemma-2-2b-finetuned", 
                        help="Output directory for model and checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, 
                        help="Learning rate")
    parser.add_argument("--use_4bit", action="store_true", 
                        help="Use 4-bit quantization")
    parser.add_argument("--max_seq_length", type=int, default=512, 
                        help="Maximum sequence length for training")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Number of gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=16, 
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout probability")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

def load_dataset_from_file(file_path):
    """Load dataset from JSON file."""
    logger.info(f"Loading dataset from {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples")
    return data

def main():
    """Main function to run training."""
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"Using random seed: {args.seed}")
    
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_safetensors=True)
    
    # Make sure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set padding token to EOS token")
    
    # Prepare model loading arguments
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "use_safetensors": True,  # Use safetensors to avoid PyTorch vulnerability
    }
    
    # Add quantization config if specified
    if args.use_4bit:
        logger.info("Using 4-bit quantization")
        from transformers import BitsAndBytesConfig
        
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model
    logger.info(f"Loading model {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs
    )
    
    # Prepare model for kbit training if using quantization
    if args.use_4bit:
        logger.info("Preparing model for k-bit training")
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    logger.info("Setting up LoRA")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Move model to device
    if not args.use_4bit:  # For 4-bit, device mapping is handled internally
        model = model.to(device)
    
    # Load dataset
    data = load_dataset_from_file(args.train_file)
    
    # Tokenize dataset
    logger.info("Tokenizing dataset")
    texts = [item["text"] for item in data]
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_length,
        return_tensors="pt"
    )
    
    # Create labels for causal language modeling
    encodings["labels"] = encodings["input_ids"].clone()
    
    # Create dataset and dataloader
    dataset = TextDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Set up learning rate scheduler
    num_update_steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warmup_steps = int(0.03 * max_train_steps)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # Training loop
    logger.info("Starting training")
    model.train()
    total_steps = 0
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs}")
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Normalize loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                total_steps += 1
                
                # Log progress
                progress_bar.set_postfix({"loss": loss.item() * args.gradient_accumulation_steps})
                
                # Save checkpoint periodically
                if total_steps % 10 == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{total_steps}")
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    # Save final model
    logger.info("Saving final model")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed")

if __name__ == "__main__":
    main()