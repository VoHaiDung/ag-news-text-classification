import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Constants
MODEL_PATH = "./results/final_model"
MAX_LENGTH = 512
STRIDE = 128
labels = ["World", "Sports", "Business", "Sci/Tech"]

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def sliding_window_tokenize(text, tokenizer, max_len=MAX_LENGTH, stride=STRIDE):
    enc = tokenizer(text, return_attention_mask=True, return_tensors=None, truncation=False)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    total_len = len(input_ids)
    start = 0
    chunks = []

    while start < total_len:
        end = min(start + max_len, total_len)
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]

        # pad to max_len
        pad_len = max_len - len(chunk_ids)
        if pad_len > 0:
            chunk_ids = chunk_ids + [tokenizer.pad_token_id] * pad_len
            chunk_mask = chunk_mask + [0] * pad_len

        chunks.append({
            "input_ids": torch.tensor([chunk_ids]),
            "attention_mask": torch.tensor([chunk_mask])
        })

        if end == total_len:
            break
        start += (max_len - stride)

    return chunks


def classify(text):
    # Split text into overlapping windows
    chunks = sliding_window_tokenize(text, tokenizer)
    logits_list = []

    # Forward pass on each chunk
    with torch.no_grad():
        for chunk in chunks:
            outputs = model(
                input_ids=chunk["input_ids"],
                attention_mask=chunk["attention_mask"]
            )
            logits_list.append(outputs.logits.numpy())

    # Aggregate logits
    all_logits = np.vstack(logits_list).squeeze(1)
    avg_logits = np.mean(all_logits, axis=0)
    pred_idx = int(np.argmax(avg_logits))
    return labels[pred_idx]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your input text here\"")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    pred = classify(input_text)
    print(f"Predicted category: {pred}")
