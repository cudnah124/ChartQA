import os
import json
from PIL import Image
from datasets import Dataset
from tqdm import tqdm

SYSTEM_PROMPT = """You are a helpful assistant capable of visual reasoning.
Provide step-by-step reasoning about the chart, then output your final answer
in JSON format: {answer: 'your_answer'}"""

class ChartQAGRPODataset:
    def __init__(self, data_file, image_dir):
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
    
    def to_hf_dataset(self):
        dataset_dict = {
            "prompt": [],
            "images": [],
            "ground_truth": []
        }
        
        for item in tqdm(self.data, desc="Processing images"):
            image_path = os.path.join(self.image_dir, item['image'])
            if not os.path.exists(image_path):
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
                
                if max(image.size) > 800:
                    image.thumbnail((800, 800), Image.Resampling.LANCZOS)
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item['question']}
                ]
                
                dataset_dict["prompt"].append(messages)
                dataset_dict["images"].append([image])
                dataset_dict["ground_truth"].append(item['label'])
            except Exception:
                continue
        
        return Dataset.from_dict(dataset_dict)

def load_datasets(train_file, train_images, val_file=None, val_images=None):
    train_ds = ChartQAGRPODataset(train_file, train_images).to_hf_dataset()
    
    val_ds = None
    if val_file and val_images:
        val_ds = ChartQAGRPODataset(val_file, val_images).to_hf_dataset()
    
    return train_ds, val_ds
