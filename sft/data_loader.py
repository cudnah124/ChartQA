import json
import os
from PIL import Image
from typing import Dict, List, Tuple
from functools import partial

def load_json_data(file_path: str) -> List[Dict]:
    """
    Load data from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of data items
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_response(think: str, answer: str) -> str:
    # Clean up think text
    think_text = think.strip()
    
    # Format answer as JSON
    answer_json = f'{{answer: "{answer}"}}'
    
    # Combine: think text + JSON answer
    if think_text:
        return f"{think_text} {answer_json}"
    else:
        return answer_json

def build_conversation(
    system_prompt: str,
    question: str,
    think: str,
    label: str
) -> List[Dict]:
    # Format assistant response: think + {answer: "label"}
    assistant_response = format_response(think, label)
    
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_response}
    ]
    
    return conversation

def process_single_item(
    item: Dict,
    image_folder: str,
    system_prompt: str,
    processor
) -> Dict:
    # Load image
    image_path = os.path.join(image_folder, item['image'])
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    

    conversation = build_conversation(
        system_prompt=system_prompt,
        question=item['question'],
        think=item['think'],
        label=item['label']
    )
    

    conversation[1]["content"] = [
        {"type": "image"},
        {"type": "text", "text": item['question']}
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Process with processor (Qwen2-VL format: image first, text second)
    inputs = processor(
        text=[text],
        images=[image],
        padding=False,
        return_tensors="pt"
    )
    
    return {
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0],
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": inputs["image_grid_thw"],
        "labels": inputs["input_ids"][0].clone()
    }

def format_and_transform(
    examples: Dict,
    image_folder: str,
    system_prompt: str,
    processor
) -> Dict:
        
    Returns:
        Batch of processed examples
    """
    batch_input_ids = []
    batch_attention_masks = []
    batch_pixel_values = []
    batch_image_grid_thw = []
    batch_labels = []
    
    # Process each item in batch
    for i in range(len(examples["image"])):
        item = {
            "image": examples["image"][i],
            "question": examples["question"][i],
            "think": examples["think"][i],
            "answer": examples["answer"][i],
            "label": examples["label"][i]
        }
        
        processed = process_single_item(item, image_folder, system_prompt, processor)
        
        if processed is not None:
            batch_input_ids.append(processed["input_ids"])
            batch_attention_masks.append(processed["attention_mask"])
            batch_pixel_values.append(processed["pixel_values"])
            batch_image_grid_thw.append(processed["image_grid_thw"])
            batch_labels.append(processed["labels"])
    
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_masks,
        "pixel_values": batch_pixel_values,
        "image_grid_thw": batch_image_grid_thw,
        "labels": batch_labels
    }

def load_and_process_dataset(
    data_file: str,
    image_folder: str,
    system_prompt: str,
    processor,
    batch_size: int = 4
):
    """
    Load and process dataset
    
    Args:
        data_file: Path to JSON data file
        image_folder: Path to image folder
        system_prompt: System prompt
        processor: Model processor
        batch_size: Batch size for processing
        
    Returns:
        Processed dataset
    """
    from datasets import load_dataset
    
    # Load dataset
    dataset = load_dataset("json", data_files=data_file, split="train")
    
    # Process dataset
    processed_dataset = dataset.map(
        partial(
            format_and_transform,
            image_folder=image_folder,
            system_prompt=system_prompt,
            processor=processor
        ),
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names
    )
    
    return processed_dataset
