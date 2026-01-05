import re
import json

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_answer(text):
    try:
        if '{answer:' in text or '{"answer":' in text:
            start = text.rfind('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                answer_json = text[start:end].replace('answer:', '"answer":').replace("'", '"')
                parsed = json.loads(answer_json)
                return parsed.get("answer", "").strip()
    except:
        pass
    return ""

def compute_format_reward(response):
    if '{answer:' in response or '{"answer":' in response:
        return 1.0
    return 0.0

def compute_accuracy_reward(response, ground_truth):
    pred = extract_answer(response)
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(str(ground_truth))
    
    if pred_norm == gt_norm:
        return 1.0
    
    if gt_norm in pred_norm or pred_norm in gt_norm:
        return 1.0
    
    return 0.0

def compute_length_reward(num_tokens, target=256, max_len=512):
    if num_tokens <= target:
        return 1.0
    elif num_tokens <= max_len:
        return 1.0 - (num_tokens - target) / (max_len - target)
    else:
        return -0.5

def create_reward_function(processor, config):
    def combined_reward_func(prompts, completions, ground_truth, **kwargs):
        rewards = []
        
        for completion, gt in zip(completions, ground_truth):
            response = str(completion)
            
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            
            response_tokens = processor.tokenizer.encode(response, add_special_tokens=False)
            num_tokens = len(response_tokens)
            
            r_format = compute_format_reward(response)
            r_accuracy = compute_accuracy_reward(response, gt)
            r_length = compute_length_reward(num_tokens, config.target_length, config.max_length)
            
            r_format_w = config.lambda_format * r_format
            r_accuracy_w = config.lambda_accuracy * r_accuracy
            r_length_w = config.lambda_length * r_length
            
            total_reward = r_format_w * (r_accuracy_w + r_length_w) + r_format_w - 1.0
            
            rewards.append(float(total_reward))
        
        return rewards
    
    return combined_reward_func
