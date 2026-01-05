from dataclasses import dataclass

@dataclass
class GRPOConfig:
    model_name: str = "Nhaass/Qwen3-VL-2B-ChartQA"
    
    num_epochs: int = 1
    batch_size: int = 1
    num_samples_per_prompt: int = 4
    learning_rate: float = 1e-6
    
    max_new_tokens: int = 512
    temperature: float = 0.9
    
    lambda_format: float = 1.0
    lambda_accuracy: float = 1.0
    lambda_length: float = 0.5
    
    target_length: int = 200
    max_length: int = 256
    
    kl_coef: float = 0.1
    clip_range: float = 0.2
    
    train_file: str = "ChartQADataset/train/train_augmented.json"
    train_images: str = "ChartQADataset/train/png"
    val_file: str = "ChartQADataset/val/val_augmented.json"
    val_images: str = "ChartQADataset/val/png"
    output_dir: str = "./grpo_checkpoints"
