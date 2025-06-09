#!/usr/bin/env python3
"""
Vision-Language Model Data Utilities

This module provides common data loading and processing utilities for VLM training,
including dataset loading, data collation, and preprocessing functions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import torch
from datasets import Dataset
from transformers import ProcessorMixin

logger = logging.getLogger(__name__)


class BaseVLMDataCollator:
    """Base data collator for Vision-Language Models"""
    
    def __init__(self, processor: ProcessorMixin, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
    
    def format_conversation(self, text: str, caption: str, image: Any) -> List[Dict[str, Any]]:
        """
        Format a single example into a conversation format.
        Can be overridden by subclasses for model-specific formatting.
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption}]
            }
        ]
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process a batch of features into model inputs"""
        messages_list = []
        images_list = []
        
        for feature in features:
            messages = self.format_conversation(
                feature.get("text", ""),
                feature.get("caption", ""),
                feature.get("image")
            )
            messages_list.append(messages)
            if "image" in feature:
                images_list.append(feature["image"])
        
        # Apply chat template if available
        if hasattr(self.processor, "apply_chat_template"):
            texts = [
                self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in messages_list
            ]
        else:
            # Fallback for processors without chat template
            texts = [self._format_messages_fallback(msgs) for msgs in messages_list]
        
        # Process text and images together
        batch = self.processor(
            text=texts,
            images=images_list if images_list else None,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # For training, we need labels
        if "input_ids" in batch:
            labels = batch["input_ids"].clone()
            if hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "pad_token_id"):
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        
        return batch
    
    def _format_messages_fallback(self, messages: List[Dict[str, Any]]) -> str:
        """Fallback message formatting for processors without chat template"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", [])
            
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            
            if text_parts:
                formatted.append(f"{role}: {' '.join(text_parts)}")
        
        return "\n".join(formatted)


class FlickrDatasetLoader:
    """Loader for Flickr-style datasets"""
    
    @staticmethod
    def load(dataset_path: str, image_dir: str, num_workers: int = 4) -> Dataset:
        """Load Flickr dataset from local files"""
        logger.info(f"Loading Flickr dataset from {dataset_path}")
        
        # Load JSON file
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Process each item
        processed_data = []
        image_dir_path = Path(image_dir)
        
        for item in data:
            # Handle different possible field names
            image_filename = item.get('image', item.get('image_path', ''))
            
            # Remove 'images/' prefix if it exists
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
            try:
                example['image'] = Image.open(example['image_path']).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image {example['image_path']}: {e}")
                # Return None image to filter out later
                example['image'] = None
            return example
        
        dataset = dataset.map(load_image, num_proc=num_workers)
        
        # Filter out examples with failed image loading
        dataset = dataset.filter(lambda x: x['image'] is not None)
        
        return dataset


class COCODatasetLoader:
    """Loader for COCO-style datasets"""
    
    @staticmethod
    def load(dataset_path: str, image_dir: str, num_workers: int = 4) -> Dataset:
        """Load COCO dataset from local files"""
        # Implementation for COCO format
        raise NotImplementedError("COCO dataset loader not yet implemented")


class GenericVLMDatasetLoader:
    """Generic loader that detects dataset format"""
    
    @staticmethod
    def load(
        dataset_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config: Optional[str] = None,
        num_workers: int = 4
    ) -> Dataset:
        """
        Load VLM dataset from various sources
        
        Args:
            dataset_path: Path to local dataset file
            image_dir: Directory containing images
            dataset_name: Name of dataset from HuggingFace Hub
            dataset_config: Configuration name for HF dataset
            num_workers: Number of workers for data processing
            
        Returns:
            Loaded dataset
        """
        if dataset_name:
            # Load from Hugging Face Hub
            from datasets import load_dataset
            logger.info(f"Loading dataset {dataset_name} from Hugging Face Hub")
            dataset = load_dataset(dataset_name, dataset_config)
            return dataset["train"] if "train" in dataset else dataset
        
        elif dataset_path and image_dir:
            # Detect format based on file content
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            # Simple format detection
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                
                # Check for Flickr-style format
                if any(key in first_item for key in ['image', 'image_path', 'caption', 'captions']):
                    return FlickrDatasetLoader.load(dataset_path, image_dir, num_workers)
                
                # Add more format detections here
                
            raise ValueError(f"Unable to detect dataset format for {dataset_path}")
        
        else:
            raise ValueError("Either dataset_name or (dataset_path and image_dir) must be provided")


def create_data_collator(
    processor: ProcessorMixin,
    model_type: str = "auto",
    max_length: int = 2048
) -> BaseVLMDataCollator:
    """
    Create appropriate data collator based on model type
    
    Args:
        processor: Model processor
        model_type: Type of model (auto, gemma, llava, idefics, etc.)
        max_length: Maximum sequence length
        
    Returns:
        Data collator instance
    """
    # For now, return base collator
    # Can be extended with model-specific collators
    return BaseVLMDataCollator(processor, max_length)