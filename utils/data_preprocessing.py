#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for preprocessing data for Gemma 3 27B training.
Includes functions for formatting instruction data and handling multimodal inputs.
"""

import os
import json
import random
from typing import Dict, List, Optional, Union, Tuple, Any
from PIL import Image
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer


def prepare_instruction_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 2048,
    include_input: bool = True,
) -> Dataset:
    """
    Prepare a dataset in instruction-tuning format (Dolly format).
    
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
    data_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    processor: Any = None,
    max_seq_length: int = 2048,
    raw_data: Optional[List[Dict[str, Any]]] = None,
    prompt_template: str = "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
    response_template: str = "{output}<end_of_turn>",
) -> Dataset:
    """
    Prepare a multimodal dataset with both text and images.
    
    Args:
        data_path: Path to the JSON file containing the data (optional if raw_data is provided)
        image_dir: Directory containing the image files
        tokenizer: Tokenizer to use for encoding the texts
        processor: Image processor from the model
        max_seq_length: Maximum sequence length
        raw_data: List of data examples to process (alternative to data_path)
        prompt_template: Template string for formatting prompts
        response_template: Template string for formatting responses
        
    Returns:
        A HuggingFace Dataset
    """
    # Load data either from file or use provided raw_data
    if raw_data is None:
        if data_path is None:
            raise ValueError("Either data_path or raw_data must be provided")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = raw_data
    
    formatted_data = []
    
    for item in data:
        # Check if this is a multimodal example with an image
        has_image = False
        processed_image = None
        
        if "image" in item and item["image"]:
            has_image = True
            
            # Handle different image formats (path string or PIL Image)
            if isinstance(item["image"], str):
                # Image is a path - resolve against image_dir if provided
                if image_dir is not None:
                    image_path = os.path.join(image_dir, item["image"])
                else:
                    image_path = item["image"]
                
                if os.path.exists(image_path):
                    # Load and process the image if it exists
                    try:
                        image = Image.open(image_path).convert("RGB")
                        
                        if processor:
                            # Process the image with the model's processor
                            processed_image = processor(images=image, return_tensors="pt")["pixel_values"][0]
                        else:
                            # Just include the raw image path
                            processed_image = image_path
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                        has_image = False
                else:
                    print(f"Warning: Image file not found: {image_path}")
                    has_image = False
            elif hasattr(item["image"], "convert"):  # Check if it's a PIL Image
                image = item["image"]
                if processor:
                    # Process the image with the model's processor
                    processed_image = processor(images=image, return_tensors="pt")["pixel_values"][0]
                else:
                    processed_image = image
            else:
                # Assume it's already processed (e.g., tensor from HF dataset)
                processed_image = item["image"]
        
        # Format the text components
        # Support both 'output' and 'caption' fields
        output_text = item.get("output", item.get("caption", ""))
        
        # Apply templates
        instruction = item.get("instruction", "Describe this image in detail.")
        prompt = prompt_template.format(instruction=instruction)
        response = response_template.format(output=output_text)
        
        # Create the formatted example
        formatted_example = {
            "text": prompt + response,
            "prompt": prompt,
            "response": response,
            "has_image": has_image,
        }
        
        # Add image if available
        if has_image and processed_image is not None:
            formatted_example["image"] = processed_image
        
        formatted_data.append(formatted_example)
    
    # Create the dataset
    return Dataset.from_list(formatted_data)


def prepare_flickr8k_for_vlm(
    data_path: str,
    image_dir: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    processor: Any = None,
    split: str = "train",
) -> Dataset:
    """
    Prepare the Flickr8k dataset specifically for VLM training.
    
    Args:
        data_path: Path to the JSON file containing the Flickr8k data
        image_dir: Directory containing the image files
        tokenizer: Tokenizer to use for encoding the texts
        processor: Image processor from the model
        split: Which split to use ("train", "val", or "test")
        
    Returns:
        A HuggingFace Dataset ready for VLM training
    """
    # Determine the correct file path based on the split
    if not data_path.endswith(f"flickr8k_{split}.json"):
        data_path = os.path.join(os.path.dirname(data_path), f"flickr8k_{split}.json")
    
    # Ensure the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Flickr8k {split} split not found at {data_path}")
    
    print(f"Loading Flickr8k {split} split from {data_path}")
    
    # Use the general multimodal dataset preparation function
    return prepare_multimodal_dataset(
        data_path=data_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        processor=processor,
        prompt_template="<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
        response_template="{output}<end_of_turn>"
    )


if __name__ == "__main__":
    # Example usage
    print("This module provides utility functions for data preprocessing.")
    print("Import and use these functions in your training scripts.")