from trl import GRPOConfig as TRLGRPOConfig, GRPOTrainer

def create_grpo_trainer(model, processor, train_dataset, val_dataset, reward_func, config, callbacks):
    grpo_config = TRLGRPOConfig(
        output_dir=config.output_dir,
        
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        
        num_generations=config.num_samples_per_prompt,
        max_completion_length=config.max_new_tokens,
        max_prompt_length=2048,
        temperature=config.temperature,
        top_p=0.9,
        top_k=50,
        
        beta=config.kl_coef,
        
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        
        logging_steps=1,
        logging_first_step=True,
        disable_tqdm=False,
        
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )
    
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        reward_funcs=[reward_func],
        processing_class=processor,
        callbacks=callbacks,
    )
    
    return trainer
