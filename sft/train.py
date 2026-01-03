import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from model import load_model_and_processor, print_trainable_parameters
from data_loader import load_and_process_dataset
from collator import QwenCompletionCollator
from trainer import create_trainer
from callbacks import TrainingMonitorCallback

def main():
    print("\n" + "=" * 80)
    print("ChartQA SFT Training")
    print("=" * 80 + "\n")
    

    config = get_config()
    print("✅ Configuration loaded\n")
    

    model, processor = load_model_and_processor(config)
    print_trainable_parameters(model)
    print()
    

    print("=" * 80)
    print("Loading Training Data")
    print("=" * 80)
    train_dataset = load_and_process_dataset(
        data_file=config.data.train_file,
        image_folder=config.data.train_images,
        system_prompt=config.system_prompt.prompt,
        processor=processor,
        batch_size=config.data.batch_size
    )
    print(f"✅ Train dataset: {len(train_dataset)} samples\n")
    
    print("=" * 80)
    print("Loading Validation Data")
    print("=" * 80)
    eval_dataset = load_and_process_dataset(
        data_file=config.data.val_file,
        image_folder=config.data.val_images,
        system_prompt=config.system_prompt.prompt,
        processor=processor,
        batch_size=config.data.batch_size
    )
    print(f"✅ Eval dataset: {len(eval_dataset)} samples\n")
    

    data_collator = QwenCompletionCollator(processor=processor)
    print()
    

    print("=" * 80)
    print("Setting up Trainer")
    print("=" * 80)
    trainer = create_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        config=config,
        callbacks=[TrainingMonitorCallback()]
    )
    print("✅ Trainer ready\n")
    

    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    trainer.train()
    

    print("\n" + "=" * 80)
    print("Saving Model")
    print("=" * 80)
    final_output_dir = os.path.join(config.training.output_dir, "final_model")
    trainer.save_model(final_output_dir)
    processor.save_pretrained(final_output_dir)
    print(f"✅ Model saved to: {final_output_dir}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
