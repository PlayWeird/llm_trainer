#!/usr/bin/env python3
"""
Training Configuration Utilities

This module provides base configuration classes and utilities for model training,
including argument parsing, configuration management, and common training parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class BaseModelArguments:
    """Base arguments for model configuration"""
    
    model_name_or_path: str = field(
        default="google/gemma-2-2b-it",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading models"}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Torch dtype for model weights (float32, float16, bfloat16)"}
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "Device map for model placement"}
    )


@dataclass
class QuantizationArguments:
    """Arguments for model quantization"""
    
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4-bit quantization"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8-bit quantization"}
    )
    bnb_4bit_quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization type for 4-bit (fp4 or nf4)"}
    )
    bnb_4bit_compute_dtype: str = field(
        default="bfloat16",
        metadata={"help": "Compute dtype for 4-bit quantization"}
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Whether to use double quantization for 4-bit"}
    )


@dataclass
class LoRAArguments:
    """Arguments for LoRA configuration"""
    
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA for fine-tuning"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )
    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Target modules for LoRA. If None, will use model defaults"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias type (none, all, lora_only)"}
    )


@dataclass
class DataArguments:
    """Base arguments for data configuration"""
    
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (from the HF hub)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use"}
    )
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local dataset file"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum total input sequence length after tokenization"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "The number of processes to use for preprocessing"}
    )
    validation_split_percentage: Optional[int] = field(
        default=None,
        metadata={"help": "Percentage of training data to use for validation"}
    )


@dataclass
class VLMDataArguments(DataArguments):
    """Arguments specific to Vision-Language Model data"""
    
    image_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing images"}
    )
    image_processor_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of image processor to use (if different from model)"}
    )
    max_image_size: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum size for image preprocessing"}
    )


@dataclass
class TrainingArguments(HFTrainingArguments):
    """Extended training arguments with common defaults"""
    
    output_dir: str = field(
        default="./outputs/model-finetuned",
        metadata={"help": "The output directory where model predictions and checkpoints will be written"}
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Overwrite the content of the output directory"}
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "Whether to run training"}
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for training"}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU for evaluation"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass"}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing to save memory"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for AdamW optimizer"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Linear warmup over warmup_steps"}
    )
    num_train_epochs: float = field(
        default=3.0,
        metadata={"help": "Total number of training epochs"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps"}
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={"help": "Limit the total amount of checkpoints"}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy (no, steps, epoch)"}
    )
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "The evaluation strategy (no, steps, epoch)"}
    )
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 (mixed) precision instead of 32-bit"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use fp16 (mixed) precision instead of 32-bit"}
    )
    tf32: bool = field(
        default=True,
        metadata={"help": "Whether to enable tf32 mode for Ampere GPUs"}
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of subprocesses to use for data loading"}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Remove columns not used by the model"}
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to the Hub"}
    )
    report_to: Union[str, List[str]] = field(
        default="none",
        metadata={"help": "The list of integrations to report results to"}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed config file"}
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use"}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={"help": "Whether to find unused parameters in DDP"}
    )
    
    def __post_init__(self):
        super().__post_init__()
        
        # Set tf32 for better performance on Ampere GPUs
        if self.tf32:
            import torch
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True


@dataclass
class ModelWithLoRAArguments(BaseModelArguments, LoRAArguments, QuantizationArguments):
    """Combined model arguments including LoRA and quantization"""
    pass


def get_model_config_for_architecture(model_name: str) -> dict:
    """
    Get model-specific configuration based on architecture
    
    Args:
        model_name: Model name or path
        
    Returns:
        Dictionary with model-specific settings
    """
    config = {
        "target_modules": None,
        "task_type": "CAUSAL_LM",
        "modules_to_save": None,
    }
    
    # Gemma models
    if "gemma" in model_name.lower():
        config["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # LLaVA models
    elif "llava" in model_name.lower():
        config["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Idefics models
    elif "idefics" in model_name.lower():
        config["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # Qwen models
    elif "qwen" in model_name.lower():
        config["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Default fallback
    else:
        config["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    return config