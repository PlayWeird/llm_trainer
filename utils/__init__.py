"""
LLM Trainer Utilities Package

This package provides common utilities for training large language models
and vision-language models.
"""

from .vlm_data_utils import (
    BaseVLMDataCollator,
    FlickrDatasetLoader,
    GenericVLMDatasetLoader,
    create_data_collator
)

from .training_config import (
    BaseModelArguments,
    QuantizationArguments,
    LoRAArguments,
    DataArguments,
    VLMDataArguments,
    TrainingArguments,
    ModelWithLoRAArguments,
    get_model_config_for_architecture
)

from .model_utils import (
    get_quantization_config,
    get_torch_dtype,
    load_model_for_training,
    setup_lora,
    save_model_and_processor,
    load_model_for_inference
)

from .training_utils import (
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
    estimate_model_size
)

__all__ = [
    # Data utilities
    "BaseVLMDataCollator",
    "FlickrDatasetLoader",
    "GenericVLMDatasetLoader",
    "create_data_collator",
    
    # Training configuration
    "BaseModelArguments",
    "QuantizationArguments",
    "LoRAArguments",
    "DataArguments",
    "VLMDataArguments",
    "TrainingArguments",
    "ModelWithLoRAArguments",
    "get_model_config_for_architecture",
    
    # Model utilities
    "get_quantization_config",
    "get_torch_dtype",
    "load_model_for_training",
    "setup_lora",
    "save_model_and_processor",
    "load_model_for_inference",
    
    # Training utilities
    "SavePeftModelCallback",
    "LoggingCallback",
    "MemoryEfficientTrainer",
    "setup_training_environment",
    "compute_metrics_for_generation",
    "create_trainer",
    "save_training_info",
    "log_gpu_memory_usage",
    "get_compute_metrics_fn",
    "GradientAccumulationManager",
]