import torch
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class QwenCompletionCollator:
    
    processor: Any
    
    def __post_init__(self):

        self.response_template = "<|im_start|>assistant\n"
        self.response_token_ids = self.processor.tokenizer.encode(
            self.response_template,
            add_special_tokens=False
        )
        
        print(f"📋 Response template: {repr(self.response_template)}")
        print(f"📋 Response token IDs: {self.response_token_ids}")
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        

        from torch.nn.utils.rnn import pad_sequence
        
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id
        )
        
        attention_masks = pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0
        )
        
        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )
        
        pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)
        image_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0)
        
        labels = self._mask_labels(input_ids, labels)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw
        }
    
    def _mask_labels(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n = len(self.response_token_ids)
        masked_count = 0
        not_found_count = 0
        
        for i in range(len(input_ids)):
            start_idx = -1
            

            for j in range(len(input_ids[i]) - n + 1):
                if input_ids[i][j : j + n].tolist() == self.response_token_ids:
                    start_idx = j + n
                    break
            
            if start_idx != -1:

                labels[i, :start_idx] = -100
                masked_count += 1
            else:

                labels[i, :] = -100
                not_found_count += 1
        

        if not_found_count > 0:
            print(f"⚠️  Response template not found in {not_found_count}/{len(input_ids)} samples")
        
        return labels
