"""
GRPO (Group Relative Policy Optimization) Module for ChartQA

This module provides a complete implementation of GRPO training for vision-language models
on the ChartQA dataset. It includes configuration, data loading, model setup, reward functions,
and training utilities.

Main Components:
    - GRPOConfig: Configuration dataclass for GRPO training parameters
    - Model loading utilities: load_model, load_processor
    - Data utilities: ChartQAGRPODataset, load_datasets
    - Reward functions: compute_format_reward, compute_accuracy_reward, compute_length_reward
    - Training: create_grpo_trainer, create_reward_function
    - Callbacks: RewardLoggingCallback

"""

__version__ = "1.0.0"
__author__ = "ChartQA GRPO Team"

# Configuration
from .config import GRPOConfig

# Model utilities
from .model import (
    load_model,
    load_processor,
)

# Data utilities
from .data_loader import (
    ChartQAGRPODataset,
    load_datasets,
    SYSTEM_PROMPT,
)

# Reward functions
from .rewards import (
    normalize_text,
    extract_answer,
    compute_format_reward,
    compute_accuracy_reward,
    compute_length_reward,
    create_reward_function,
)

# Trainer utilities
from .trainer import create_grpo_trainer

# Callbacks
from .callbacks import RewardLoggingCallback

# Define public API
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    
    # Configuration
    "GRPOConfig",
    
    # Model
    "load_model",
    "load_processor",
    
    # Data
    "ChartQAGRPODataset",
    "load_datasets",
    "SYSTEM_PROMPT",
    
    # Rewards
    "normalize_text",
    "extract_answer",
    "compute_format_reward",
    "compute_accuracy_reward",
    "compute_length_reward",
    "create_reward_function",
    
    # Training
    "create_grpo_trainer",
    
    # Callbacks
    "RewardLoggingCallback",
]
