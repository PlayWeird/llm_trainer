#!/usr/bin/env python3
"""
Training Utilities

This module provides common utilities for training workflows, including
trainer initialization, callbacks, metric computation, and training helpers.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union
import torch
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    EvalPrediction,
    set_seed
)
from torch.utils.data import Dataset
import numpy as np

logger = logging.getLogger(__name__)


class SavePeftModelCallback(TrainerCallback):
    """Callback to save PEFT model properly during training"""
    
    def on_save(self, args, state, control, model=None, **kwargs):
        if model is not None:
            # Check if this is a PEFT model
            if hasattr(model, "save_pretrained"):
                checkpoint_path = os.path.join(
                    args.output_dir,
                    f"checkpoint-{state.global_step}"
                )
                model.save_pretrained(checkpoint_path)


class LoggingCallback(TrainerCallback):
    """Enhanced logging callback for training progress"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Format and log metrics
            metrics_str = " - ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                     for k, v in logs.items()])
            logger.info(f"Step {state.global_step}: {metrics_str}")


class MemoryEfficientTrainer(Trainer):
    """Extended trainer with memory efficiency features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add custom callbacks
        self.add_callback(LoggingCallback())
        
        # Add PEFT callback if using LoRA
        if hasattr(self.model, "peft_config"):
            self.add_callback(SavePeftModelCallback())
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss for better memory efficiency
        """
        # Enable autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=self.args.fp16 or self.args.bf16):
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        
        return (loss, outputs) if return_outputs else loss


def setup_training_environment(
    seed: int = 42,
    tf32: bool = True,
    allow_tf32: bool = True,
    cudnn_deterministic: bool = False
):
    """
    Set up the training environment with reproducibility and performance settings
    
    Args:
        seed: Random seed for reproducibility
        tf32: Whether to enable TF32 on Ampere GPUs
        allow_tf32: Whether to allow TF32 for matmul operations
        cudnn_deterministic: Whether to use deterministic CUDNN algorithms
    """
    # Set random seed
    set_seed(seed)
    
    # Configure TF32 for better performance on Ampere GPUs
    if torch.cuda.is_available() and tf32:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    
    # Set CUDNN deterministic mode if requested
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    
    # Set environment variables for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    logger.info(f"Training environment set up with seed={seed}, tf32={tf32}")


def compute_metrics_for_generation(eval_preds: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for generation tasks
    
    Args:
        eval_preds: Predictions from evaluation
        
    Returns:
        Dictionary of metrics
    """
    predictions = eval_preds.predictions
    labels = eval_preds.label_ids
    
    # Handle different prediction formats
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Compute perplexity
    loss = np.mean(predictions) if predictions.ndim == 1 else np.mean(predictions[:, 0])
    perplexity = np.exp(loss)
    
    metrics = {
        "eval_loss": float(loss),
        "eval_perplexity": float(perplexity),
    }
    
    return metrics


def create_trainer(
    model: PreTrainedModel,
    args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    data_collator: Optional[Any] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    compute_metrics: Optional[Any] = None,
    callbacks: Optional[List[TrainerCallback]] = None,
    optimizers: tuple = (None, None),
    preprocess_logits_for_metrics: Optional[Any] = None,
    use_memory_efficient_trainer: bool = True,
    **kwargs
) -> Trainer:
    """
    Create a trainer instance with appropriate configuration
    
    Args:
        model: Model to train
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator
        tokenizer: Tokenizer (or processor)
        compute_metrics: Metrics computation function
        callbacks: Additional callbacks
        optimizers: Custom optimizers
        preprocess_logits_for_metrics: Function to preprocess logits
        use_memory_efficient_trainer: Whether to use memory efficient trainer
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured trainer instance
    """
    # Choose trainer class
    trainer_class = MemoryEfficientTrainer if use_memory_efficient_trainer else Trainer
    
    # Create trainer
    trainer = trainer_class(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        optimizers=optimizers,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        **kwargs
    )
    
    return trainer


def save_training_info(
    output_dir: str,
    model_args: Any,
    data_args: Any,
    training_args: Any,
    metrics: Optional[Dict[str, float]] = None
):
    """
    Save training configuration and metrics
    
    Args:
        output_dir: Output directory
        model_args: Model arguments
        data_args: Data arguments
        training_args: Training arguments
        metrics: Training metrics
    """
    import json
    from dataclasses import asdict, is_dataclass
    
    # Convert dataclasses to dict
    def to_dict(obj):
        if is_dataclass(obj):
            return asdict(obj)
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)
    
    # Prepare info dictionary
    info = {
        "model_args": to_dict(model_args),
        "data_args": to_dict(data_args),
        "training_args": to_dict(training_args),
    }
    
    if metrics:
        info["final_metrics"] = metrics
    
    # Save to JSON
    info_path = os.path.join(output_dir, "training_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    
    logger.info(f"Training info saved to {info_path}")


def log_gpu_memory_usage():
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"GPU {i}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def estimate_model_size(model: PreTrainedModel) -> Dict[str, float]:
    """
    Estimate model size and parameter count
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with size information
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size in GB (assuming float32)
    size_gb = param_count * 4 / (1024**3)
    trainable_size_gb = trainable_count * 4 / (1024**3)
    
    return {
        "total_parameters": param_count,
        "trainable_parameters": trainable_count,
        "total_size_gb": size_gb,
        "trainable_size_gb": trainable_size_gb,
        "trainable_percentage": (trainable_count / param_count * 100) if param_count > 0 else 0
    }


def get_compute_metrics_fn(task_type: str = "generation"):
    """
    Get appropriate metrics computation function based on task type
    
    Args:
        task_type: Type of task (generation, classification, etc.)
        
    Returns:
        Metrics computation function
    """
    if task_type == "generation":
        return compute_metrics_for_generation
    else:
        # Add more task types as needed
        return None


class GradientAccumulationManager:
    """Helper class to manage gradient accumulation"""
    
    def __init__(self, accumulation_steps: int):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def should_step(self) -> bool:
        """Check if optimizer should step"""
        self.step_count += 1
        return self.step_count % self.accumulation_steps == 0
    
    def should_zero_grad(self) -> bool:
        """Check if gradients should be zeroed"""
        return self.step_count % self.accumulation_steps == 1
    
    def get_scale_factor(self) -> float:
        """Get gradient scaling factor"""
        return 1.0 / self.accumulation_steps