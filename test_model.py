import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

def normalize_text(text):
    import re
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_answer(text):
    import re
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

def test_model(
    model_name="Nhaass/Qwen3-VL-2B-ChartQA-GRPO",
    test_file="ChartQADataset/test/test_augmented.json",
    test_images="ChartQADataset/test/png",
    num_samples=100,
    output_file="test_results.json"
):
    print(f"Loading model: {model_name}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    print(f"Loading test data: {test_file}")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    system_prompt = """You are a helpful assistant capable of visual reasoning.
Provide step-by-step reasoning about the chart, then output your final answer
in JSON format: {answer: 'your_answer'}"""
    
    results = {
        "model": model_name,
        "total": 0,
        "correct": 0,
        "format_correct": 0,
        "responses": []
    }
    
    total_tokens = []
    
    print(f"\nEvaluating on {min(num_samples, len(test_data))} samples...")
    
    for item in tqdm(test_data[:num_samples]):
        try:
            image_path = os.path.join(test_images, item['image'])
            image = Image.open(image_path).convert("RGB")
            
            if max(image.size) > 800:
                image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": item['question']}
                ]}
            ]
            
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False)
            
            response = processor.decode(outputs[0], skip_special_tokens=True)
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1].strip()
            
            response_tokens = processor.tokenizer.encode(response, add_special_tokens=False)
            total_tokens.append(len(response_tokens))
            
            has_format = '{answer:' in response or '{"answer":' in response
            if has_format:
                results["format_correct"] += 1
            
            pred = extract_answer(response)
            is_correct = normalize_text(pred) == normalize_text(item['label'])
            if is_correct:
                results["correct"] += 1
            
            results["responses"].append({
                "question": item['question'],
                "ground_truth": item['label'],
                "prediction": pred,
                "correct": is_correct,
                "has_format": has_format,
                "response_length": len(response_tokens)
            })
            
            results["total"] += 1
            
            del image, inputs, outputs
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
    results["format_compliance"] = results["format_correct"] / results["total"] if results["total"] > 0 else 0
    results["avg_response_length"] = sum(total_tokens) / len(total_tokens) if total_tokens else 0
    
    print(f"\n{'='*80}")
    print("Test Results")
    print(f"{'='*80}")
    print(f"Accuracy: {results['correct']}/{results['total']} = {results['accuracy']*100:.2f}%")
    print(f"Format Compliance: {results['format_correct']}/{results['total']} = {results['format_compliance']*100:.2f}%")
    print(f"Avg Response Length: {results['avg_response_length']:.1f} tokens")
    print(f"{'='*80}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    test_model()
