import torch
from transformers import TrainerCallback

class RewardLoggingCallback(TrainerCallback):
    def __init__(self):
        self.step_metrics = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            step = state.global_step
            
            metrics = {
                'step': step,
                'loss': logs.get('loss', 0),
                'reward_mean': logs.get('reward', logs.get('rewards/combined_reward_func/mean', 0)),
                'reward_std': logs.get('reward_std', logs.get('rewards/combined_reward_func/std', 0)),
                'frac_reward_zero_std': logs.get('frac_reward_zero_std', 0),
                'kl': logs.get('kl', logs.get('objective/kl', 0)),
                'entropy': logs.get('entropy', logs.get('objective/entropy', 0)),
                'completions_mean_length': logs.get('completions/mean_length', 0),
                'completions_clipped_ratio': logs.get('completions/clipped_ratio', 0),
                'clip_ratio_region': logs.get('clip_ratio/region_mean', 0),
                'lr': logs.get('learning_rate', 0),
                'step_time': logs.get('step_time', 0),
            }
            
            self.step_metrics.append(metrics)
            
            print(f"\n{'='*80}")
            print(f"Step {step}")
            print(f"{'='*80}")
            print(f"  Loss:           {metrics['loss']:.4f}")
            print(f"  Reward:         {metrics['reward_mean']:.4f} ± {metrics['reward_std']:.4f}")
            print(f"  Zero Std Frac:  {metrics['frac_reward_zero_std']:.2%}")
            print(f"  KL Divergence:  {metrics['kl']:.6f}")
            print(f"  Entropy:        {metrics['entropy']:.4f}")
            print(f"  Completion Len: {metrics['completions_mean_length']:.1f} tokens")
            print(f"  LR:             {metrics['lr']:.2e}")
            
            if torch.cuda.is_available() and step % 10 == 0:
                gpu_mem = torch.cuda.memory_allocated(0) / 1e9
                print(f"  GPU Memory:     {gpu_mem:.2f} GB")
            
            print(f"{'='*80}\n")
