import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

def load_model(model_name, lora_r=8, lora_alpha=16):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    return model

def load_processor(model_name):
    return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
