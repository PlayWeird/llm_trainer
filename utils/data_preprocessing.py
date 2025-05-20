#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for preprocessing data for Gemma 3 27B training.
Includes functions for formatting instruction data and handling multimodal inputs.
"""

import os
import json
import random
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer


def prepare_alpaca_format_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    include_input: bool = True,
) -> Dataset:
    """
    Prepare a dataset in Alpaca instruction format.
    
    Args:
        data_path: Path to the JSON file containing the data
        tokenizer: Tokenizer to use for encoding the texts
        max_seq_length: Maximum sequence length
        include_input: Whether to include the input field in the prompt
        
    Returns:
        A HuggingFace Dataset
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    
    for item in data:
        if include_input and item.get("input", "").strip():
            prompt = f"<start_of_turn>user\n{item['instruction']}\n{item['input']}<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = f"<start_of_turn>user\n{item['instruction']}<end_of_turn>\n<start_of_turn>model\n"
        
        response = f"{item['output']}<end_of_turn>"
        full_text = prompt + response
        
        formatted_data.append({
            "text": full_text,
            "prompt": prompt,
            "response": response,
        })
    
    # Create the dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Add labels for training (shifted input_ids)
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].clone()
        prompt_mask = []
        
        for i, prompt in enumerate(examples["prompt"]):
            # Tokenize prompt to find its length
            prompt_ids = tokenizer(prompt, truncation=True, padding=False)["input_ids"]
            prompt_length = len(prompt_ids)
            
            # Create mask where prompt tokens are -100 (ignored in loss)
            curr_mask = [-100] * prompt_length + examples["labels"][i][prompt_length:].tolist()
            # Ensure the mask is the same length as the input
            curr_mask = curr_mask[:len(examples["labels"][i])]
            curr_mask = curr_mask + [-100] * (len(examples["labels"][i]) - len(curr_mask))
            prompt_mask.append(curr_mask)
        
        examples["labels"] = torch.tensor(prompt_mask)
        return examples
    
    return tokenized_dataset.map(add_labels, batched=True)


def prepare_multimodal_dataset(
    data_path: str,
    image_dir: str,
    tokenizer: PreTrainedTokenizer,
    processor=None,
    max_seq_length: int = 2048,
) -> Dataset:
    """
    Prepare a multimodal dataset with both text and images.
    
    Args:
        data_path: Path to the JSON file containing the data
        image_dir: Directory containing the image files
        tokenizer: Tokenizer to use for encoding the texts
        processor: Image processor from the model
        max_seq_length: Maximum sequence length
        
    Returns:
        A HuggingFace Dataset
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    
    for item in data:
        # Check if this is a multimodal example
        if "image" in item and item["image"]:
            image_path = os.path.join(image_dir, item["image"])
            if os.path.exists(image_path):
                # Load and process the image if it exists
                image = Image.open(image_path).convert("RGB")
                
                if processor:
                    # Process the image with the model's processor
                    processed_image = processor(images=image, return_tensors="pt")["pixel_values"][0]
                else:
                    # Just include the raw image path
                    processed_image = image_path
                
                prompt = f"<start_of_turn>user\n{item['instruction']}<end_of_turn>\n<start_of_turn>model\n"
                response = f"{item['output']}<end_of_turn>"
                
                formatted_data.append({
                    "text": prompt + response,
                    "prompt": prompt,
                    "response": response,
                    "image": processed_image,
                    "has_image": True
                })
        else:
            # Text-only example
            prompt = f"<start_of_turn>user\n{item['instruction']}<end_of_turn>\n<start_of_turn>model\n"
            response = f"{item['output']}<end_of_turn>"
            
            formatted_data.append({
                "text": prompt + response,
                "prompt": prompt,
                "response": response,
                "has_image": False
            })
    
    # Create the dataset
    return Dataset.from_list(formatted_data)


if __name__ == "__main__":
    # Example usage
    print("This module provides utility functions for data preprocessing.")
    print("Import and use these functions in your training scripts.")