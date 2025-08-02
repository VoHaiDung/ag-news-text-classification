import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional

class AGNewsDataset(Dataset):
    """AG News dataset for PyTorch"""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer_name: str = 'bert-base-uncased',
        max_length: int = 512,
        label_map: Optional[Dict[int, str]] = None
    ):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.label_map = label_map or {
            0: "World",
            1: "Sports", 
            2: "Business",
            3: "Sci/Tech"
        }
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
