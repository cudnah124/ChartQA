from transformers import TrainerCallback

class TrainingMonitorCallback(TrainerCallback):
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            
            # Training loss
            if "loss" in logs:
                print(f"📊 Step {step}: Loss = {logs['loss']:.4f}")
            
            # Evaluation loss
            if "eval_loss" in logs:
                print(f"✅ Step {step}: Eval Loss = {logs['eval_loss']:.4f}")
            
            # Learning rate
            if "learning_rate" in logs:
                print(f"📈 Step {step}: LR = {logs['learning_rate']:.2e}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n🎯 Epoch {state.epoch} completed!\n")
    
    def on_train_begin(self, args, state, control, **kwargs):
        print("\n🚀 Training started!\n")
    
    def on_train_end(self, args, state, control, **kwargs):
        print("\n🎉 Training completed!\n")
