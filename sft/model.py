from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import torch

def load_model_and_processor(config):
    print("=" * 60)
    print("Loading Model and Processor")
    print("=" * 60)
    

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True
    )
    
    print(f"✅ Loaded model: Qwen/Qwen2-VL-2B-Instruct")
    print(f"   Max sequence length: {config.model.max_seq_length}")
    

    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    
    print(f"✅ Applied LoRA:")
    print(f"   r: {config.lora.r}")
    print(f"   alpha: {config.lora.lora_alpha}")
    print(f"   dropout: {config.lora.lora_dropout}")
    print("=" * 60)
    
    return model, processor

def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model
    
    Args:
        model: The model to analyze
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_param:,} || "
          f"Trainable%: {100 * trainable_params / all_param:.2f}%")
