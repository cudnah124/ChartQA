from transformers import TrainingArguments, EarlyStoppingCallback, Trainer

def create_trainer(
    model,
    processor,
    train_dataset,
    eval_dataset,
    data_collator,
    config,
    callbacks=None
):

    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_steps=config.training.warmup_steps,
        logging_steps=config.training.logging_steps,
        eval_strategy="steps",
        eval_steps=config.training.eval_steps,
        save_strategy="steps",
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        optim=config.training.optim,
        seed=config.training.seed,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        report_to=config.training.report_to,
        remove_unused_columns=config.training.remove_unused_columns,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )
    

    if callbacks is None:
        callbacks = []
    
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=config.training.early_stopping_patience
        )
    )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        callbacks=callbacks,
    )
    
    return trainer
