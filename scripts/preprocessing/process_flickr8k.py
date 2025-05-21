#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process the Flickr8k dataset for Vision-Language Model (VLM) training.
This script takes the downloaded Flickr8k dataset and processes it into a format
suitable for training VLM models with the train_gemma3_vlm.py script.

Features:
- Processes raw Flickr8k dataset from Hugging Face into a standardized format
- Saves images to a structured directory
- Creates a training-ready JSON file with image paths and captions
- Supports dataset splitting (train/val/test)
- Configurable sample size for testing with smaller datasets

Example usage:
    python process_flickr8k.py \
        --input_dir="datasets/test_dataset/vlm/flickr8k" \
        --output_dir="datasets/processed/vlm/flickr8k" \
        --sample_size=1000 \
        --val_split=0.1 \
        --test_split=0.1
"""

import os
import json
import argparse
import random
import shutil
from typing import Dict, List, Tuple, Any, Optional
import logging
from datasets import load_dataset
from PIL import Image
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process Flickr8k dataset for VLM training")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="datasets/test_dataset/vlm/flickr8k",
        help="Directory containing raw Flickr8k data"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="datasets/processed/vlm/flickr8k",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--sample_size", 
        type=int, 
        default=0,
        help="Number of examples to include (0 for all)"
    )
    parser.add_argument(
        "--val_split", 
        type=float, 
        default=0.1,
        help="Fraction of data for validation"
    )
    parser.add_argument(
        "--test_split", 
        type=float, 
        default=0.1,
        help="Fraction of data for testing"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--instruction", 
        type=str, 
        default="Describe this image in detail.",
        help="Instruction prompt for the model"
    )
    parser.add_argument(
        "--from_hf",
        action="store_true",
        help="Load dataset directly from Hugging Face instead of local files"
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="ariG23498/flickr8k",
        help="Hugging Face dataset ID to use if --from_hf is set"
    )
    
    return parser.parse_args()

def load_flickr8k_dataset(args) -> List[Dict[str, Any]]:
    """Load the Flickr8k dataset from either local files or Hugging Face."""
    if args.from_hf:
        # Load directly from Hugging Face
        logger.info(f"Loading Flickr8k dataset from Hugging Face: {args.hf_dataset}")
        try:
            dataset = load_dataset(args.hf_dataset)
            
            if "train" in dataset:
                # Get the training split if available
                dataset = dataset["train"]
            
            # Convert to list format
            data = []
            
            # Determine available fields
            sample_item = dataset[0]
            caption_field = None
            image_field = None
            
            # Find caption field
            for field in ["caption", "captions"]:
                if field in sample_item:
                    caption_field = field
                    break
            
            # Find image field
            for field in ["image", "img", "images"]:
                if field in sample_item:
                    image_field = field
                    break
            
            if not caption_field or not image_field:
                # Try to identify fields by inspection
                for field in sample_item:
                    if "caption" in field.lower() or "text" in field.lower():
                        caption_field = field
                    if "image" in field.lower() or "img" in field.lower() or "photo" in field.lower():
                        image_field = field
            
            if not caption_field or not image_field:
                raise ValueError(f"Could not identify caption and image fields in dataset. Available fields: {list(sample_item.keys())}")
            
            logger.info(f"Using '{caption_field}' as caption field and '{image_field}' as image field")
            
            # Process all items
            for item in dataset:
                caption = item[caption_field]
                image = item[image_field]
                
                # Handle case where there are multiple captions per image
                if isinstance(caption, list):
                    # Create an entry for each caption
                    for single_caption in caption:
                        data.append({
                            "image": image,
                            "caption": single_caption
                        })
                else:
                    data.append({
                        "image": image,
                        "caption": caption
                    })
            
            logger.info(f"Loaded {len(data)} image-caption pairs from Hugging Face")
            return data
            
        except Exception as e:
            logger.error(f"Error loading from Hugging Face: {e}")
            logger.info("Falling back to local files")
    
    # Load from local files
    logger.info(f"Loading Flickr8k dataset from local files in {args.input_dir}")
    
    data_file = os.path.join(args.input_dir, "flickr8k_test_data.json")
    images_dir = os.path.join(args.input_dir, "images")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} examples from local files")
    
    # Convert to standard format if needed
    standardized_data = []
    for item in data:
        standardized_data.append({
            "image": item["image"],
            "caption": item["output"],
            "instruction": item.get("instruction", args.instruction)
        })
    
    return standardized_data

def process_and_save_dataset(
    data: List[Dict[str, Any]], 
    args
) -> Dict[str, List[Dict[str, str]]]:
    """Process and save the dataset to the specified output directory."""
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Sample the dataset if required
    if args.sample_size > 0 and args.sample_size < len(data):
        random.seed(args.seed)
        data = random.sample(data, args.sample_size)
        logger.info(f"Sampled {args.sample_size} examples from dataset")
    
    # Split the dataset
    random.shuffle(data)
    total = len(data)
    
    val_size = int(total * args.val_split)
    test_size = int(total * args.test_split)
    train_size = total - val_size - test_size
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    logger.info(f"Split dataset into {len(train_data)} train, {len(val_data)} validation, and {len(test_data)} test examples")
    
    # Process images and create formatted data
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
    
    processed_splits = {}
    
    for split_name, split_data in splits.items():
        processed_data = []
        
        for i, item in enumerate(split_data):
            # Copy image to output directory
            src_path = item["image"]
            
            # Handle case where image is already a PIL Image or other object
            if not isinstance(src_path, str):
                # PIL Image or tensor from HF dataset
                if hasattr(item["image"], "save"):
                    # It's a PIL image
                    image_filename = f"{split_name}_{i:05d}.jpg"
                    dest_path = os.path.join(images_dir, image_filename)
                    item["image"].save(dest_path)
                else:
                    # Skip this item if we can't handle the image type
                    logger.warning(f"Skipping item {i} in {split_name} split due to unsupported image type: {type(item['image'])}")
                    continue
            else:
                # It's a path string
                if src_path.startswith("images/"):
                    # Relative path - resolve against input dir
                    src_path = os.path.join(args.input_dir, src_path)
                
                if not os.path.exists(src_path):
                    logger.warning(f"Image file not found: {src_path}")
                    continue
                
                # Create a new filename for the image
                image_filename = f"{split_name}_{i:05d}.jpg"
                dest_path = os.path.join(images_dir, image_filename)
                
                try:
                    # Copy the image file
                    shutil.copy2(src_path, dest_path)
                except Exception as e:
                    logger.error(f"Error copying image {src_path}: {e}")
                    continue
            
            # Format the example
            formatted_example = {
                "instruction": item.get("instruction", args.instruction),
                "output": item["caption"],
                "image": f"images/{image_filename}"
            }
            
            processed_data.append(formatted_example)
        
        # Save the processed data
        output_file = os.path.join(args.output_dir, f"flickr8k_{split_name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2)
        
        logger.info(f"Saved {len(processed_data)} processed examples to {output_file}")
        
        processed_splits[split_name] = processed_data
    
    return processed_splits

def main():
    args = parse_args()
    
    try:
        # Load the dataset
        data = load_flickr8k_dataset(args)
        
        # Process and save the dataset
        processed_data = process_and_save_dataset(data, args)
        
        logger.info("Dataset processing completed successfully")
        
        # Print summary
        total_examples = sum(len(split) for split in processed_data.values())
        logger.info(f"Total processed examples: {total_examples}")
        logger.info(f"Output directory: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()