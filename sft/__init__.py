"""
SFT (Supervised Fine-Tuning) Module for ChartQA

This module provides a complete implementation of supervised fine-tuning for vision-language 
models on the ChartQA dataset. It includes configuration, data loading, model setup, custom 
data collators, and training utilities.

Main Components:
    - Configuration classes: ModelConfig, LoRAConfig, DataConfig, TrainingConfig, SystemPromptConfig
    - Model utilities: load_model_and_processor, print_trainable_parameters
    - Data utilities: load_json_data, format_response, build_conversation, load_and_process_dataset
    - Custom collator: QwenCompletionCollator
    - Trainer utilities: create_trainer
    - Callbacks: TrainingMonitorCallback

"""

__version__ = "1.0.0"
__author__ = "ChartQA SFT Team"

# Configuration
from .config import (
    ModelConfig,
    LoRAConfig,
    DataConfig,
    TrainingConfig,
    SystemPromptConfig,
    Config,
    get_config,
)

# Model utilities
from .model import (
    load_model_and_processor,
    print_trainable_parameters,
)

# Data utilities
from .data_loader import (
    load_json_data,
    format_response,
    build_conversation,
    process_single_item,
    load_and_process_dataset,
)

# Custom collator
from .collator import QwenCompletionCollator

# Trainer utilities
from .trainer import create_trainer

# Callbacks
from .callbacks import TrainingMonitorCallback

# Define public API
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    
    # Configuration
    "ModelConfig",
    "LoRAConfig",
    "DataConfig",
    "TrainingConfig",
    "SystemPromptConfig",
    "Config",
    "get_config",
    
    # Model
    "load_model_and_processor",
    "print_trainable_parameters",
    
    # Data
    "load_json_data",
    "format_response",
    "build_conversation",
    "process_single_item",
    "load_and_process_dataset",
    
    # Collator
    "QwenCompletionCollator",
    
    # Training
    "create_trainer",
    
    # Callbacks
    "TrainingMonitorCallback",
]
