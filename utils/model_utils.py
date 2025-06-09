#!/usr/bin/env python3
"""
Model Utilities

This module provides utilities for model initialization, quantization configuration,
LoRA setup, and other model-related operations.
"""

import logging
from typing import Optional, Dict, Any, Union, List
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    PreTrainedModel
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

logger = logging.getLogger(__name__)


def get_quantization_config(
    use_4bit: bool = False,
    use_8bit: bool = False,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_use_double_quant: bool = True
) -> Optional[BitsAndBytesConfig]:
    """
    Create quantization configuration for BitsAndBytes
    
    Args:
        use_4bit: Whether to use 4-bit quantization
        use_8bit: Whether to use 8-bit quantization
        bnb_4bit_quant_type: Quantization type for 4-bit
        bnb_4bit_compute_dtype: Compute dtype for 4-bit
        bnb_4bit_use_double_quant: Whether to use double quantization
        
    Returns:
        BitsAndBytesConfig or None if no quantization
    """
    if not (use_4bit or use_8bit):
        return None
    
    if use_4bit and use_8bit:
        logger.warning("Both 4-bit and 8-bit quantization requested. Using 4-bit.")
        use_8bit = False
    
    # Convert compute dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    compute_dtype = dtype_map.get(bnb_4bit_compute_dtype, torch.bfloat16)
    
    return BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        load_in_8bit=use_8bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float": torch.float32,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), torch.bfloat16)


def load_model_for_training(
    model_name_or_path: str,
    model_class: Optional[type] = None,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    torch_dtype: Union[str, torch.dtype] = "bfloat16",
    trust_remote_code: bool = True,
    device_map: Optional[Union[str, Dict[str, Any]]] = "auto",
    use_flash_attention: bool = True,
    gradient_checkpointing: bool = True,
    **kwargs
) -> PreTrainedModel:
    """
    Load a model for training with proper configuration
    
    Args:
        model_name_or_path: Model identifier or path
        model_class: Specific model class to use (if None, will auto-detect)
        quantization_config: Quantization configuration
        torch_dtype: Data type for model weights
        trust_remote_code: Whether to trust remote code
        device_map: Device placement strategy
        use_flash_attention: Whether to use flash attention
        gradient_checkpointing: Whether to enable gradient checkpointing
        **kwargs: Additional model loading arguments
        
    Returns:
        Loaded model
    """
    # Convert string dtype to torch dtype if necessary
    if isinstance(torch_dtype, str):
        torch_dtype = get_torch_dtype(torch_dtype)
    
    # Determine model class if not provided
    if model_class is None:
        # Try to detect VLM models
        vlm_keywords = ["llava", "idefics", "gemma3", "qwen2-vl", "blip", "clip"]
        is_vlm = any(keyword in model_name_or_path.lower() for keyword in vlm_keywords)
        
        if is_vlm:
            # Try to import model-specific classes
            if "gemma3" in model_name_or_path.lower():
                try:
                    from transformers import Gemma3ForConditionalGeneration
                    model_class = Gemma3ForConditionalGeneration
                except ImportError:
                    model_class = AutoModelForVision2Seq
            elif "llava" in model_name_or_path.lower():
                try:
                    from transformers import LlavaForConditionalGeneration
                    model_class = LlavaForConditionalGeneration
                except ImportError:
                    model_class = AutoModelForVision2Seq
            elif "idefics" in model_name_or_path.lower():
                try:
                    from transformers import Idefics3ForConditionalGeneration
                    model_class = Idefics3ForConditionalGeneration
                except ImportError:
                    model_class = AutoModelForVision2Seq
            else:
                model_class = AutoModelForVision2Seq
        else:
            model_class = AutoModelForCausalLM
    
    # Set up attention implementation
    attn_implementation = "flash_attention_2" if use_flash_attention else "eager"
    
    # Load model
    logger.info(f"Loading model {model_name_or_path} with {model_class.__name__}")
    
    try:
        model = model_class.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            attn_implementation=attn_implementation,
            **kwargs
        )
    except Exception as e:
        if "flash_attention_2" in str(e):
            logger.warning("Flash Attention 2 not available, falling back to eager attention")
            model = model_class.from_pretrained(
                model_name_or_path,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                attn_implementation="eager",
                **kwargs
            )
        else:
            raise
    
    # Enable gradient checkpointing if specified
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    
    # Prepare model for k-bit training if using quantization
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    return model


def setup_lora(
    model: PreTrainedModel,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[List[str]] = None,
    lora_bias: str = "none",
    task_type: str = "CAUSAL_LM",
    **kwargs
) -> PeftModel:
    """
    Set up LoRA for a model
    
    Args:
        model: The model to apply LoRA to
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        lora_bias: Bias configuration
        task_type: Task type for LoRA
        **kwargs: Additional LoRA configuration
        
    Returns:
        Model with LoRA applied
    """
    logger.info("Setting up LoRA for fine-tuning")
    
    # Auto-detect target modules if not provided
    if lora_target_modules is None:
        # Get model type
        model_type = model.config.model_type.lower()
        
        # Define target modules based on model architecture
        target_modules_map = {
            "gemma": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "gemma2": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "gemma3": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "llava": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "idefics": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "idefics2": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "idefics3": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        
        lora_target_modules = target_modules_map.get(
            model_type,
            ["q_proj", "k_proj", "v_proj", "o_proj"]  # Default fallback
        )
        
        logger.info(f"Auto-detected target modules for {model_type}: {lora_target_modules}")
    
    # Convert task type string to TaskType enum
    task_type_map = {
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "SEQ_CLS": TaskType.SEQ_CLS,
        "TOKEN_CLS": TaskType.TOKEN_CLS,
        "QUESTION_ANS": TaskType.QUESTION_ANS,
        "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
    }
    task_type_enum = task_type_map.get(task_type.upper(), TaskType.CAUSAL_LM)
    
    # Create LoRA configuration
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type=task_type_enum,
        target_modules=lora_target_modules,
        **kwargs
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def save_model_and_processor(
    model: Union[PreTrainedModel, PeftModel],
    processor: Any,
    output_dir: str,
    is_lora: bool = False,
    save_full_model: bool = False
):
    """
    Save model and processor to disk
    
    Args:
        model: Model to save
        processor: Processor/Tokenizer to save
        output_dir: Directory to save to
        is_lora: Whether the model uses LoRA
        save_full_model: Whether to save the full model (for LoRA models)
    """
    logger.info(f"Saving model to {output_dir}")
    
    if is_lora and not save_full_model:
        # Save only the LoRA adapter weights
        model.save_pretrained(output_dir)
    else:
        # Save the full model
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(output_dir)
        else:
            # For Trainer-wrapped models
            model.save_model(output_dir)
    
    # Save processor
    if processor is not None:
        processor.save_pretrained(output_dir)
    
    logger.info("Model and processor saved successfully")


def load_model_for_inference(
    model_path: str,
    base_model_name: Optional[str] = None,
    device_map: Union[str, Dict[str, Any]] = "auto",
    torch_dtype: Union[str, torch.dtype] = "auto",
    trust_remote_code: bool = True,
    is_lora: bool = False
) -> tuple:
    """
    Load a model for inference
    
    Args:
        model_path: Path to saved model
        base_model_name: Base model name (for LoRA models)
        device_map: Device placement
        torch_dtype: Model dtype
        trust_remote_code: Whether to trust remote code
        is_lora: Whether the model is a LoRA adapter
        
    Returns:
        Tuple of (model, processor)
    """
    from transformers import AutoProcessor, AutoTokenizer
    
    if is_lora and base_model_name:
        # Load base model first
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code
        )
        # Load LoRA weights
        model = PeftModel.from_pretrained(model, model_path)
        # Merge LoRA weights for faster inference
        model = model.merge_and_unload()
    else:
        # Load full model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code
        )
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except:
        processor = AutoTokenizer.from_pretrained(model_path)
    
    return model, processor