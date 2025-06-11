#!/usr/bin/env python3
"""
Test inference with the fine-tuned Qwen2.5-14B model
"""

import os
import sys
import torch
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_finetuned_model(checkpoint_path):
    """Test the fine-tuned model"""
    logger.info(f"Loading fine-tuned model from: {checkpoint_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        logger.info("✓ Tokenizer loaded")
        
        # Load base model with 4-bit quantization to save memory
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-14B-Instruct",
            quantization_config=bnb_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        logger.info("✓ Base model loaded")
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logger.info("✓ LoRA adapter loaded")
        
        # Set to evaluation mode
        model.eval()
        
        # Test prompts from the Dolly dataset style
        test_prompts = [
            "What is the capital of France?",
            "Explain photosynthesis in simple terms.",
            "Write a Python function to reverse a string.",
            "What are the benefits of exercise?",
            "How does a computer work?"
        ]
        
        logger.info("\n" + "="*60)
        logger.info("Testing fine-tuned model responses:")
        logger.info("="*60)
        
        for i, prompt in enumerate(test_prompts, 1):
            # Format as instruction
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "assistant" in response:
                response = response.split("assistant")[-1].strip()
            else:
                response = response[len(text):].strip()
            
            logger.info(f"\n{i}. Prompt: {prompt}")
            logger.info(f"   Response: {response[:200]}{'...' if len(response) > 200 else ''}")
        
        # Check memory usage
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"\nGPU Memory used: {mem_allocated:.2f}GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return False

def main():
    """Main test function"""
    # Find the latest checkpoint
    output_dir = project_root / "outputs"
    
    # Look for the most recent qwen25-14b output
    checkpoints = list(output_dir.glob("qwen25-14b-full-test-*/"))
    if not checkpoints:
        logger.error("No checkpoints found!")
        return
    
    latest_checkpoint = sorted(checkpoints)[-1]
    logger.info(f"Using checkpoint: {latest_checkpoint}")
    
    # Test the model
    success = test_finetuned_model(str(latest_checkpoint))
    
    if success:
        logger.info("\n✅ Inference test completed successfully!")
    else:
        logger.error("\n❌ Inference test failed!")

if __name__ == "__main__":
    main()