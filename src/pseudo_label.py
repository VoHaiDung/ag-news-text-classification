import os
import argparse
import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def tokenize_sliding(examples, tokenizer, max_len: int = 512, stride: int = 256):
    # Apply sliding window tokenization to a batch of texts
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_len,
        stride=stride,
        return_overflowing_tokens=True,
        padding="max_length",
    )

def predict_logits(
    model, tokenizer, texts: List[str], device: torch.device
) -> np.ndarray:
    # Predict mean logits per document using sliding window
    all_logits = []
    for text in tqdm(texts, desc="Infer"):
        # Tokenize with overflow to handle long sequences
        tokenized = tokenizer(
            text,
            max_length=512,
            stride=256,
            truncation=True,
            padding="max_length",
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits_chunks = []
            for i in range(tokenized["input_ids"].shape[0]):
                input_ids = tokenized["input_ids"][i].unsqueeze(0).to(device)
                attention = tokenized["attention_mask"][i].unsqueeze(0).to(device)
                out = model(input_ids=input_ids, attention_mask=attention)
                logits_chunks.append(out.logits.cpu())
        # Average logits across all segments
        stacked = torch.stack(logits_chunks).mean(dim=0)
        all_logits.append(stacked.squeeze(0).numpy())
    return np.vstack(all_logits)

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo labels using ensemble models")
    parser.add_argument("--unlabeled_csv", type=str, required=True)  # Input CSV with 'title' and 'description'
    parser.add_argument("--deberta_dir", type=str, required=True)
    parser.add_argument("--longformer_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="data/processed/pseudo_labeled.csv")
    parser.add_argument("--prob_threshold", type=float, default=0.90)
    parser.add_argument("--batch_size", type=int, default=8)  # Not used (windowed tokenization is per sample)
    args = parser.parse_args()

    # Load unlabeled corpus
    raw_df = pd.read_csv(args.unlabeled_csv)
    texts = (raw_df["title"] + " " + raw_df["description"]).tolist()
    logger.info("Loaded %d unlabeled samples", len(texts))

    # Load tokenizers and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_deb = AutoTokenizer.from_pretrained(args.deberta_dir)
    tok_lon = AutoTokenizer.from_pretrained(args.longformer_dir)
    model_deb = AutoModelForSequenceClassification.from_pretrained(args.deberta_dir).to(device).eval()
    model_lon = AutoModelForSequenceClassification.from_pretrained(args.longformer_dir).to(device).eval()

    # Predict logits for each model
    logits_deb = predict_logits(model_deb, tok_deb, texts, device)
    logits_lon = predict_logits(model_lon, tok_lon, texts, device)

    # Ensemble by soft-voting
    logits_ensemble = (logits_deb + logits_lon) / 2
    probs = softmax(torch.from_numpy(logits_ensemble), dim=-1).numpy()
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    # Filter by probability threshold
    keep_mask = max_probs >= args.prob_threshold
    kept = sum(keep_mask)
    logger.info("Selected %d / %d samples (≥ %.2f prob)", kept, len(texts), args.prob_threshold)

    pseudo_df = raw_df.loc[keep_mask].copy()
    pseudo_df["label"] = preds[keep_mask]
    pseudo_df["confidence"] = max_probs[keep_mask]

    # Save pseudo‑labeled CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pseudo_df.to_csv(args.output_csv, index=False)
    logger.info("Pseudo-labeled data saved to %s", args.output_csv)


if __name__ == "__main__":
    main()
