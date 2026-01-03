"""
Inference script for ChartQA SFT
"""

import os
import sys
import json
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from unsloth import FastVisionModel

def load_trained_model(model_path):
    """Load trained model"""
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    return model, processor

def inference(model, processor, image_path, question, system_prompt):
    """
    Run inference on a single image-question pair
    
    Args:
        model: Trained model
        processor: Model processor
        image_path: Path to image
        question: Question text
        system_prompt: System prompt
        
    Returns:
        Model response
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Build conversation
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process inputs
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True
    )
    
    # Decode
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    
    return response

def extract_answer_from_response(response):
    """
    Extract answer from response format: think_text {answer: "value"}
    
    Args:
        response: Model response
        
    Returns:
        Extracted answer or None
    """
    try:
        # Find {answer: "..."} pattern
        if "{answer:" in response:
            start = response.index("{answer:")
            end = response.index("}", start) + 1
            answer_json = response[start:end]
            
            # Parse JSON (handle both "answer" and answer formats)
            answer_json = answer_json.replace("answer:", '"answer":')
            parsed = json.loads(answer_json)
            return parsed.get("answer", "")
        else:
            return None
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None

def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on ChartQA model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--question", type=str, required=True, help="Question text")
    args = parser.parse_args()
    
    # Load config for system prompt
    config = get_config()
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model, processor = load_trained_model(args.model_path)
    print("✅ Model loaded\n")
    
    # Run inference
    print(f"Image: {args.image}")
    print(f"Question: {args.question}\n")
    
    response = inference(
        model=model,
        processor=processor,
        image_path=args.image,
        question=args.question,
        system_prompt=config.system_prompt.prompt
    )
    
    print("=" * 60)
    print("Full Response:")
    print("=" * 60)
    print(response)
    print()
    
    # Extract answer
    answer = extract_answer_from_response(response)
    if answer:
        print("=" * 60)
        print("Extracted Answer:")
        print("=" * 60)
        print(answer)
    
if __name__ == "__main__":
    main()
