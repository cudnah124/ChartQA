from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen3-VL-2B-Thinking"
    max_seq_length: int = 1024
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    
@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

@dataclass
class DataConfig:
    train_file: str = "ChartQADataset/train/train_converted.json"
    val_file: str = "ChartQADataset/val/val_converted.json"
    train_images: str = "ChartQADataset/train/png"
    val_images: str = "ChartQADataset/val/png"
    batch_size: int = 4
    
    def __post_init__(self):
        """Convert relative paths to absolute"""
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.train_file = os.path.join(base_path, self.train_file)
        self.val_file = os.path.join(base_path, self.val_file)
        self.train_images = os.path.join(base_path, self.train_images)
        self.val_images = os.path.join(base_path, self.val_images)

@dataclass
class TrainingConfig:
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.05
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 0
    logging_steps: int = 5
    eval_steps: int = 25
    save_steps: int = 25
    save_total_limit: int = 2
    fp16: bool = True
    bf16: bool = False
    optim: str = "adamw_8bit"
    seed: int = 3407
    neftune_noise_alpha: float = 5.0
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    report_to: str = "none"
    remove_unused_columns: bool = False
    early_stopping_patience: int = 3

@dataclass
class SystemPromptConfig:
    prompt: str = (
        "You are a helpful assistant capable of visual reasoning. "
        "Provide step-by-step reasoning about the chart, then output your final answer "
        "in JSON format: {answer: 'your_answer'}"
    )

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system_prompt: SystemPromptConfig = field(default_factory=SystemPromptConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            lora=LoRAConfig(**config_dict.get('lora', {})),
            data=DataConfig(**config_dict.get('data', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            system_prompt=SystemPromptConfig(**config_dict.get('system_prompt', {}))
        )
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'model': self.model.__dict__,
            'lora': self.lora.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'system_prompt': self.system_prompt.__dict__
        }

def get_config():
    """Get default configuration"""
    return Config()
