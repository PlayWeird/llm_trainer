#!/usr/bin/env python3
"""
Integration test for the refactored training script
Tests with a small model and minimal data to verify everything works together
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def create_test_dataset(output_dir):
    """Create a minimal test dataset"""
    # Create dataset directory
    dataset_dir = Path(output_dir) / "test_dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    # Create images directory
    images_dir = dataset_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Create dummy image files (just text files for testing)
    for i in range(3):
        img_path = images_dir / f"test_{i}.jpg"
        img_path.write_text(f"dummy image {i}")
    
    # Create dataset JSON
    dataset = [
        {
            "image": f"test_{i}.jpg",
            "instruction": "Describe this image.",
            "output": f"This is test image number {i}."
        }
        for i in range(3)
    ]
    
    dataset_json = dataset_dir / "dataset.json"
    with open(dataset_json, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    return str(dataset_json), str(images_dir)


def test_training_script():
    """Test the training script with minimal configuration"""
    print("Testing refactored training script...")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test dataset
        dataset_path, image_dir = create_test_dataset(temp_dir)
        output_dir = Path(temp_dir) / "output"
        
        # Prepare command
        cmd = [
            sys.executable,
            "scripts/training/train_gemma3_vlm.py",
            "--model_name_or_path", "google/gemma-2-2b",  # Use smallest model
            "--dataset_path", dataset_path,
            "--image_dir", image_dir,
            "--output_dir", str(output_dir),
            "--num_train_epochs", "1",
            "--max_steps", "2",  # Just 2 steps for testing
            "--per_device_train_batch_size", "1",
            "--gradient_accumulation_steps", "1",
            "--learning_rate", "2e-5",
            "--use_lora", "True",
            "--use_4bit", "False",  # Disable quantization for test
            "--logging_steps", "1",
            "--save_steps", "10",  # Don't save during test
            "--report_to", "none",
        ]
        
        # Run training
        import subprocess
        
        # Activate conda environment
        activate_cmd = f"source /home/user/miniforge3/etc/profile.d/conda.sh && conda activate llm_trainer_env"
        full_cmd = f"{activate_cmd} && {' '.join(cmd)}"
        
        print(f"Running: {full_cmd}")
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
        
        # Check if output files were created
        if output_dir.exists():
            print("✓ Output directory created")
            
            # Check for expected files
            expected_files = ["training_info.json", "adapter_config.json"]
            for file in expected_files:
                if (output_dir / file).exists():
                    print(f"✓ {file} created")
                else:
                    print(f"✗ {file} not found")
                    return False
        else:
            print("✗ Output directory not created")
            return False
        
        print("✓ Training script integration test passed!")
        return True


def main():
    """Run integration test"""
    try:
        success = test_training_script()
        return 0 if success else 1
    except Exception as e:
        print(f"Error running integration test: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())