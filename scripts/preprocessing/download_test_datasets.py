#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and prepare small test datasets for LLM and VLM training.
Downloads and converts datasets to the format expected by the training scripts.
"""

import os
import json
import random
import argparse
import requests
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def download_llm_dataset(output_dir, dataset_name, num_samples=2000, seed=42):
    """
    Download and prepare a small LLM instruction tuning dataset.
    
    Args:
        output_dir: Directory to save the processed dataset
        dataset_name: Which dataset to download ('dolly' or 'oasst')
        num_samples: Number of examples to include
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_test_data.json")
    
    print(f"Downloading and processing {dataset_name} dataset...")
    
    if dataset_name == "dolly":
        # Load Databricks Dolly dataset
        dataset = load_dataset("databricks/databricks-dolly-15k", trust_remote_code=True)
        data = dataset["train"]
        
        # Sample a subset
        indices = random.sample(range(len(data)), min(num_samples, len(data)))
        sampled_data = [data[i] for i in indices]
        
        # Convert to expected format (instruction, input, output)
        processed_data = []
        for item in sampled_data:
            processed_data.append({
                "instruction": item["instruction"],
                "input": item.get("context", ""),  # Use context as input if available
                "output": item["response"]
            })
            
    elif dataset_name == "oasst":
        # Load OpenAssistant tiny subset
        dataset = load_dataset("ybelkada/oasst1-tiny-subset", trust_remote_code=True)
        data = dataset["train"]
        
        # Sample a subset (though it's already small)
        indices = random.sample(range(len(data)), min(num_samples, len(data)))
        sampled_data = [data[i] for i in indices]
        
        # Convert to expected format
        processed_data = []
        for item in sampled_data:
            # The first message is the instruction, the response is the output
            if len(item["messages"]) >= 2:
                processed_data.append({
                    "instruction": item["messages"][0]["content"],
                    "input": "",
                    "output": item["messages"][1]["content"]
                })
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Save the processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(processed_data)} examples to {output_file}")
    return output_file


def download_vlm_dataset(output_dir, dataset_name, num_samples=500, seed=42):
    """
    Download and prepare a small VLM image-text dataset.
    
    Args:
        output_dir: Directory to save the processed dataset
        dataset_name: Which dataset to download ('flickr8k' or 'vqa-rad')
        num_samples: Number of examples to include
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    data_dir = os.path.join(output_dir, dataset_name)
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    
    output_file = os.path.join(data_dir, f"{dataset_name}_test_data.json")
    
    print(f"Downloading and processing {dataset_name} dataset...")
    
    if dataset_name == "flickr8k":
        # Use the Flickr8k dataset from HuggingFace
        try:
            # First try jontooy's dataset which should have pre-extracted features
            print("Loading Flickr8k dataset from 'jontooy/Flickr8k-Image-Features'...")
            dataset = load_dataset("jontooy/Flickr8k-Image-Features", trust_remote_code=True)
        except Exception as e:
            print(f"Error loading from first source: {e}")
            # Try ariG23498's flickr8k dataset
            try:
                print("Trying alternative source 'ariG23498/flickr8k'...")
                dataset = load_dataset("ariG23498/flickr8k", trust_remote_code=True)
            except Exception as e2:
                print(f"Error loading from second source: {e2}")
                # Try Naveengo's flickr8k dataset
                try:
                    print("Trying alternative source 'Naveengo/flickr8k'...")
                    dataset = load_dataset("Naveengo/flickr8k", trust_remote_code=True)
                except Exception as e3:
                    print(f"Error loading from third source: {e3}")
                    # Try jxie's flickr8k dataset
                    try:
                        print("Trying alternative source 'jxie/flickr8k'...")
                        dataset = load_dataset("jxie/flickr8k", trust_remote_code=True)
                    except Exception as e4:
                        print(f"Error loading from fourth source: {e4}")
                        # Try using flickr30k as a fallback
                        try:
                            print("Trying fallback to 'nlphuji/flickr30k'...")
                            dataset = load_dataset("nlphuji/flickr30k", trust_remote_code=True)
                        except Exception as e5:
                            # Try one more flickr30k source
                            try:
                                print("Trying final source 'sentence-transformers/flickr30k-captions'...")
                                dataset = load_dataset("sentence-transformers/flickr30k-captions", trust_remote_code=True)
                            except Exception as e6:
                                raise ValueError(f"Failed to load any Flickr dataset from multiple sources. Last error: {e6}")
        
        # Get the train split or the available split
        if "train" in dataset:
            data = dataset["train"]
        else:
            # Use the first available split
            first_key = list(dataset.keys())[0]
            data = dataset[first_key]
            print(f"Using '{first_key}' split from dataset")
        
        print(f"Dataset loaded successfully with {len(data)} items")
        print(f"Sample item keys: {list(data[0].keys())}")
        
        # Print dataset info
        print(f"Dataset structure: {dataset}")
        
        # Sample a subset
        indices = random.sample(range(len(data)), min(num_samples, len(data)))
        sampled_data = [data[i] for i in indices]
        
        # Print sample data structure
        print(f"Sample item keys: {list(sampled_data[0].keys())}")
        print(f"Sample item content: {sampled_data[0]}")
        
        # Convert to expected format and process images
        processed_data = []
        for item in tqdm(sampled_data, desc="Processing images"):
            try:
                # Extract image and caption based on dataset structure
                # Different datasets might have different key names
                image_data = None
                caption = None
                
                # Special handling for jontooy/Flickr8k-Image-Features
                if "feature" in item:
                    # This dataset has pre-extracted features, not images
                    # Use the image_id and features directly
                    image_id = item.get("img_id", f"feature_{len(processed_data)}")
                    print(f"Processing feature-based item with id: {image_id}")
                    
                    # Create a placeholder image with the image_id text
                    img_filename = f"flickr8k_{len(processed_data)}.jpg"
                    local_path = os.path.join(image_dir, img_filename)
                    
                    # Extract caption
                    if "caption" in item:
                        caption = item["caption"]
                    elif "sent" in item:
                        caption = item["sent"]
                    else:
                        # Create placeholder caption
                        caption = f"Image {image_id} description"
                    
                    # Create placeholder image with text
                    from PIL import Image, ImageDraw, ImageFont
                    img = Image.new('RGB', (512, 512), color='gray')
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 10), f"ID: {image_id}", fill='white')
                    img.save(local_path)
                    
                    # Add to processed data
                    processed_data.append({
                        "instruction": "Describe this image in detail.",
                        "output": caption,
                        "image": f"images/{img_filename}"
                    })
                    continue
                
                # Handle 'sentence-transformers/flickr30k-captions' special case
                if "sentence1" in item and "sentence2" in item:
                    # This dataset has paired captions, no image
                    # Create a placeholder image
                    img_filename = f"flickr8k_{len(processed_data)}.jpg"
                    local_path = os.path.join(image_dir, img_filename)
                    
                    caption = item["sentence1"]  # Use first sentence as caption
                    
                    # Create placeholder image with text
                    from PIL import Image, ImageDraw, ImageFont
                    img = Image.new('RGB', (512, 512), color='gray')
                    draw = ImageDraw.Draw(img)
                    draw.text((10, 10), caption[:50], fill='white')
                    img.save(local_path)
                    
                    # Add to processed data
                    processed_data.append({
                        "instruction": "Describe this image in detail.",
                        "output": caption,
                        "image": f"images/{img_filename}"
                    })
                    continue
                
                # Standard handling for datasets with images
                if "image" in item:
                    image_data = item["image"]
                elif "img" in item:
                    image_data = item["img"]
                elif "image_path" in item:
                    image_data = item["image_path"]
                elif "filename" in item:
                    image_data = item["filename"]
                else:
                    # Try to find image field by inspecting keys
                    for key in item.keys():
                        if "image" in key.lower() or "img" in key.lower() or "photo" in key.lower() or "file" in key.lower():
                            image_data = item[key]
                            break
                
                if "caption" in item:
                    caption = item["caption"]
                elif "captions" in item:
                    # Get first caption if it's a list
                    captions = item["captions"]
                    if isinstance(captions, list):
                        caption = captions[0]
                    else:
                        caption = captions
                elif "text" in item:
                    caption = item["text"]
                elif "sentence" in item:
                    caption = item["sentence"]
                else:
                    # Try to find caption field by inspecting keys
                    for key in item.keys():
                        if "caption" in key.lower() or "text" in key.lower() or "description" in key.lower() or "sent" in key.lower():
                            caption_data = item[key]
                            if isinstance(caption_data, list):
                                caption = caption_data[0]  # Take first caption if multiple
                            else:
                                caption = caption_data
                            break
                
                if image_data is None and caption is None:
                    print(f"Missing both image data and caption in item: {item.keys()}")
                    continue
                elif image_data is None:
                    print(f"Missing image data in item: {item.keys()}")
                    # Create a placeholder image with caption text
                    img_filename = f"flickr8k_{len(processed_data)}.jpg"
                    local_path = os.path.join(image_dir, img_filename)
                    
                    # Create placeholder image with caption
                    from PIL import Image, ImageDraw, ImageFont
                    img = Image.new('RGB', (512, 512), color='gray')
                    draw = ImageDraw.Draw(img)
                    if caption:
                        draw.text((10, 10), caption[:50], fill='white')
                    img.save(local_path)
                    
                    image_data = local_path
                elif caption is None:
                    print(f"Missing caption in item: {item.keys()}")
                    # Create a placeholder caption
                    caption = f"Image {len(processed_data)} without a description"
                
                # Create a filename for the local image
                if isinstance(image_data, str):  # It's a URL or path
                    img_filename = os.path.basename(image_data)
                    if not img_filename.endswith(('.jpg', '.jpeg', '.png')):
                        img_filename = f"flickr8k_{len(processed_data)}.jpg"
                else:  # It's a PIL image or something else
                    img_filename = f"flickr8k_{len(processed_data)}.jpg"
                
                local_path = os.path.join(image_dir, img_filename)
                
                # Process the image data
                if not os.path.exists(local_path):
                    if isinstance(image_data, str):  # It's a URL or path
                        try:
                            # Try to download it
                            with open(local_path, 'wb') as img_file:
                                img_file.write(requests.get(image_data).content)
                        except Exception as e:
                            print(f"Error downloading image {image_data}: {e}")
                            # Try to load from local filesystem if it's a path
                            try:
                                if os.path.exists(image_data):
                                    import shutil
                                    shutil.copy(image_data, local_path)
                                else:
                                    # Create a placeholder image
                                    Image.new('RGB', (512, 512), color='gray').save(local_path)
                            except Exception as e2:
                                print(f"Error handling local path: {e2}")
                                # Fallback to placeholder
                                Image.new('RGB', (512, 512), color='gray').save(local_path)
                    else:
                        # Save the PIL Image directly
                        try:
                            image_data.save(local_path)
                        except Exception as e:
                            print(f"Error saving PIL image: {e}")
                            # Fallback to placeholder
                            Image.new('RGB', (512, 512), color='gray').save(local_path)
                
                processed_data.append({
                    "instruction": "Describe this image in detail.",
                    "output": caption,
                    "image": f"images/{img_filename}"
                })
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
    
    elif dataset_name == "vqa-rad":
        # Load VQA-RAD dataset
        dataset = load_dataset("flaviagiammarino/vqa-rad", trust_remote_code=True)
        data = dataset["train"]
        
        # Sample a subset
        indices = random.sample(range(len(data)), min(num_samples, len(data)))
        sampled_data = [data[i] for i in indices]
        
        # Convert to expected format and download images
        processed_data = []
        for item in tqdm(sampled_data, desc="Processing images"):
            try:
                # Extract image and QA
                image_data = item["image"]  # This might be a PIL Image or path
                question = item["question"]
                answer = item["answer"]
                
                # Create a filename for the local image
                img_filename = f"{hash(question)}_{hash(answer)}.jpg"
                local_path = os.path.join(image_dir, img_filename)
                
                # Save the image if it doesn't exist
                if not os.path.exists(local_path):
                    if isinstance(image_data, (str, Path)):
                        # If it's a path, copy the image
                        try:
                            with open(local_path, 'wb') as img_file:
                                img_file.write(requests.get(image_data).content)
                        except:
                            # Fallback: create a placeholder image
                            Image.new('RGB', (512, 512), color='gray').save(local_path)
                    else:
                        # If it's a PIL Image or similar, save it
                        try:
                            image_data.save(local_path)
                        except:
                            # Fallback: create a placeholder image
                            Image.new('RGB', (512, 512), color='gray').save(local_path)
                
                processed_data.append({
                    "instruction": question,
                    "output": answer,
                    "image": f"images/{img_filename}"
                })
            except Exception as e:
                print(f"Error processing item: {e}")
                continue
    
    # Other datasets can be added here as needed
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Save the processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(processed_data)} examples to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Download and prepare test datasets for LLM and VLM training")
    parser.add_argument("--output_dir", type=str, default="datasets/test_dataset", help="Directory to save the processed datasets")
    parser.add_argument("--llm_dataset", type=str, default="dolly", choices=["dolly", "oasst"], help="LLM dataset to download")
    parser.add_argument("--vlm_dataset", type=str, default="flickr8k", choices=["flickr8k", "vqa-rad"], help="VLM dataset to download")
    parser.add_argument("--llm_samples", type=int, default=2000, help="Number of LLM examples to include")
    parser.add_argument("--vlm_samples", type=int, default=500, help="Number of VLM examples to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--llm_only", action="store_true", help="Download only LLM dataset")
    parser.add_argument("--vlm_only", action="store_true", help="Download only VLM dataset")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.vlm_only:
        llm_dir = os.path.join(args.output_dir, "llm")
        os.makedirs(llm_dir, exist_ok=True)
        llm_file = download_llm_dataset(llm_dir, args.llm_dataset, args.llm_samples, args.seed)
        print(f"LLM dataset saved to: {llm_file}")
    
    if not args.llm_only:
        vlm_dir = os.path.join(args.output_dir, "vlm")
        os.makedirs(vlm_dir, exist_ok=True)
        vlm_file = download_vlm_dataset(vlm_dir, args.vlm_dataset, args.vlm_samples, args.seed)
        print(f"VLM dataset saved to: {vlm_file}")
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()