#!/usr/bin/env python3
"""
Test script for refactored training utilities
"""

import sys
import os
from pathlib import Path
import logging
import torch
from dataclasses import dataclass

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    logger.info("Testing imports...")
    try:
        from utils import (
            BaseVLMDataCollator,
            FlickrDatasetLoader,
            GenericVLMDatasetLoader,
            create_data_collator,
            BaseModelArguments,
            QuantizationArguments,
            LoRAArguments,
            DataArguments,
            VLMDataArguments,
            TrainingArguments,
            ModelWithLoRAArguments,
            get_model_config_for_architecture,
            get_quantization_config,
            get_torch_dtype,
            load_model_for_training,
            setup_lora,
            save_model_and_processor,
            estimate_model_size,
            SavePeftModelCallback,
            LoggingCallback,
            MemoryEfficientTrainer,
            setup_training_environment,
            compute_metrics_for_generation,
            create_trainer,
            save_training_info,
            log_gpu_memory_usage,
            get_compute_metrics_fn,
            GradientAccumulationManager,
        )
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import error: {e}")
        return False


def test_configuration_classes():
    """Test configuration dataclasses"""
    logger.info("\nTesting configuration classes...")
    try:
        from utils import ModelWithLoRAArguments, VLMDataArguments, TrainingArguments
        
        # Test instantiation with defaults
        model_args = ModelWithLoRAArguments(
            model_name_or_path="google/gemma-2-2b-it"
        )
        assert model_args.use_lora == True
        assert model_args.lora_r == 16
        assert model_args.use_4bit == False
        
        data_args = VLMDataArguments(
            dataset_path="test.json",
            image_dir="./images"
        )
        assert data_args.max_seq_length == 2048
        assert data_args.preprocessing_num_workers == 4
        
        training_args = TrainingArguments(
            output_dir="./test_output"
        )
        assert training_args.gradient_checkpointing == True
        assert training_args.bf16 == True
        
        logger.info("✓ Configuration classes work correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test error: {e}")
        return False


def test_quantization_config():
    """Test quantization configuration"""
    logger.info("\nTesting quantization configuration...")
    try:
        from utils import get_quantization_config
        
        # Test no quantization
        config = get_quantization_config(use_4bit=False, use_8bit=False)
        assert config is None
        
        # Test 4-bit quantization
        config = get_quantization_config(
            use_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16"
        )
        assert config.load_in_4bit == True
        assert config.load_in_8bit == False
        assert config.bnb_4bit_quant_type == "nf4"
        
        # Test 8-bit quantization
        config = get_quantization_config(use_8bit=True)
        assert config.load_in_8bit == True
        assert config.load_in_4bit == False
        
        logger.info("✓ Quantization configuration works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Quantization config test error: {e}")
        return False


def test_torch_dtype_conversion():
    """Test torch dtype conversion"""
    logger.info("\nTesting torch dtype conversion...")
    try:
        from utils import get_torch_dtype
        
        assert get_torch_dtype("float32") == torch.float32
        assert get_torch_dtype("float16") == torch.float16
        assert get_torch_dtype("bfloat16") == torch.bfloat16
        assert get_torch_dtype("fp16") == torch.float16
        assert get_torch_dtype("bf16") == torch.bfloat16
        
        logger.info("✓ Torch dtype conversion works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Dtype conversion test error: {e}")
        return False


def test_model_config_detection():
    """Test model-specific configuration detection"""
    logger.info("\nTesting model configuration detection...")
    try:
        from utils import get_model_config_for_architecture
        
        # Test Gemma config
        config = get_model_config_for_architecture("google/gemma-2-2b-it")
        assert config["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # Test LLaVA config
        config = get_model_config_for_architecture("llava-hf/llava-1.5-7b-hf")
        assert "gate_proj" in config["target_modules"]
        
        # Test Qwen config
        config = get_model_config_for_architecture("Qwen/Qwen2.5-7B")
        assert "gate_proj" in config["target_modules"]
        
        logger.info("✓ Model configuration detection works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Model config detection test error: {e}")
        return False


def test_data_collator():
    """Test data collator functionality"""
    logger.info("\nTesting data collator...")
    try:
        from utils import BaseVLMDataCollator
        from transformers import AutoTokenizer
        
        # Create a mock processor with tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        
        class MockProcessor:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                
            def __call__(self, text=None, images=None, **kwargs):
                # Simple mock implementation
                if text:
                    batch = self.tokenizer(text, **kwargs)
                    batch["pixel_values"] = torch.zeros(len(text), 3, 224, 224) if images else None
                    return batch
                return {}
        
        processor = MockProcessor(tokenizer)
        collator = BaseVLMDataCollator(processor)
        
        # Test formatting
        messages = collator.format_conversation(
            text="What is in this image?",
            caption="A dog playing in the park",
            image=None
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        
        logger.info("✓ Data collator works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Data collator test error: {e}")
        return False


def test_training_environment_setup():
    """Test training environment setup"""
    logger.info("\nTesting training environment setup...")
    try:
        from utils import setup_training_environment
        
        setup_training_environment(seed=42)
        
        # Check if environment variables are set
        assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"
        
        # Check if CUDA settings are applied (if CUDA available)
        if torch.cuda.is_available():
            assert torch.backends.cuda.matmul.allow_tf32 == True
        
        logger.info("✓ Training environment setup works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Training environment test error: {e}")
        return False


def test_gradient_accumulation_manager():
    """Test gradient accumulation manager"""
    logger.info("\nTesting gradient accumulation manager...")
    try:
        from utils import GradientAccumulationManager
        
        manager = GradientAccumulationManager(accumulation_steps=4)
        
        # Test step counting - should_step increments counter
        assert manager.should_step() == False  # Step 1
        assert manager.should_step() == False  # Step 2
        assert manager.should_step() == False  # Step 3
        assert manager.should_step() == True   # Step 4 - should step
        
        # After 4 steps, step_count is 4, so 4 % 4 == 0
        # should_zero_grad checks if step_count % accumulation_steps == 1
        assert manager.should_zero_grad() == False  # step_count is 4
        
        # Continue another cycle
        assert manager.should_step() == False  # Step 5 (step_count becomes 5)
        assert manager.should_zero_grad() == True  # 5 % 4 == 1, so True
        
        assert manager.get_scale_factor() == 0.25
        
        logger.info("✓ Gradient accumulation manager works correctly")
        return True
    except Exception as e:
        import traceback
        logger.error(f"✗ Gradient accumulation test error: {e}")
        logger.error(traceback.format_exc())
        return False


def test_model_size_estimation():
    """Test model size estimation"""
    logger.info("\nTesting model size estimation...")
    try:
        from utils import estimate_model_size
        import torch.nn as nn
        
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 50)
                self.linear2 = nn.Linear(50, 10)
                
        model = SimpleModel()
        info = estimate_model_size(model)
        
        assert info["total_parameters"] == 100*50 + 50 + 50*10 + 10  # weights + biases
        assert info["trainable_parameters"] == info["total_parameters"]
        assert info["trainable_percentage"] == 100.0
        
        logger.info("✓ Model size estimation works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Model size estimation test error: {e}")
        return False


def test_training_script_syntax():
    """Test that the refactored training script has valid syntax"""
    logger.info("\nTesting refactored training script syntax...")
    try:
        import ast
        script_path = Path(__file__).parent / "training" / "train_gemma3_vlm.py"
        
        with open(script_path, 'r') as f:
            code = f.read()
        
        # Try to parse the script
        ast.parse(code)
        
        logger.info("✓ Refactored training script has valid syntax")
        return True
    except SyntaxError as e:
        logger.error(f"✗ Syntax error in training script: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Error testing training script: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("Running tests for refactored training utilities...\n")
    
    tests = [
        test_imports,
        test_configuration_classes,
        test_quantization_config,
        test_torch_dtype_conversion,
        test_model_config_detection,
        test_data_collator,
        test_training_environment_setup,
        test_gradient_accumulation_manager,
        test_model_size_estimation,
        test_training_script_syntax,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Unexpected error in {test.__name__}: {e}")
            failed += 1
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info(f"{'='*50}")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)