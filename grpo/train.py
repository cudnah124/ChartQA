import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from grpo import (
    GRPOConfig,
    load_model,
    load_processor,
    load_datasets,
    create_reward_function,
    RewardLoggingCallback,
    create_grpo_trainer
)

def main():
    config = GRPOConfig()
    
    print("Loading model...")
    model = load_model(config.model_name)
    processor = load_processor(config.model_name)
    model.print_trainable_parameters()
    
    print("\nLoading datasets...")
    train_dataset, val_dataset = load_datasets(
        config.train_file,
        config.train_images,
        config.val_file,
        config.val_images
    )
    print(f"Train: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Val: {len(val_dataset)} samples")
    
    print("\nSetting up trainer...")
    reward_func = create_reward_function(processor, config)
    callbacks = [RewardLoggingCallback()]
    
    trainer = create_grpo_trainer(
        model, processor, train_dataset, val_dataset,
        reward_func, config, callbacks
    )
    
    print("\n" + "="*80)
    print("Starting GRPO Training")
    print("="*80)
    
    trainer.train()
    
    print("\n" + "="*80)
    print("Training Completed")
    print("="*80)

if __name__ == "__main__":
    main()
