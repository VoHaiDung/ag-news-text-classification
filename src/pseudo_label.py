import os
import argparse
import pandas as pd
import numpy as np
import torch
from typing import List
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils import configure_logger

# Set up logger
logger = configure_logger("outputs/logs/pseudo_label.log")

# Predict mean logits per document using sliding-window inference
# texts: list of documents; max_len, stride control window size
# batch_size: number of documents per batch for inference

def predict_logits(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_len: int,
    stride: int,
    batch_size: int
) -> np.ndarray:
    all_logits = []
    use_amp = device.type == "cuda"

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        for text in batch_texts:
            tokenized = tokenizer(
                text,
                max_length=max_len,
                stride=stride,
                truncation=True,
                padding="max_length",
                return_overflowing_tokens=True,
                return_tensors="pt",
            )

            logits_chunks = []
            for i in range(tokenized["input_ids"].shape[0]):
                input_ids = tokenized["input_ids"][i].unsqueeze(0).to(device)
                attention_mask = tokenized["attention_mask"][i].unsqueeze(0).to(device)

                with torch.no_grad():
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            out = model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits_chunks.append(out.logits.cpu())

            mean_logits = torch.stack(logits_chunks).mean(dim=0)
            all_logits.append(mean_logits.squeeze(0).numpy())

    return np.vstack(all_logits)

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels using ensemble models")
    parser.add_argument("--unlabeled_csv", type=str, required=True,
                        help="Input CSV with 'title' and 'description' columns")
    parser.add_argument("--deberta_dir", type=str, required=True,
                        help="Path to DeBERTa-LoRA model")
    parser.add_argument("--longformer_dir", type=str, required=True,
                        help="Path to Longformer-LoRA model")
    parser.add_argument("--output_csv", type=str, default="data/processed/pseudo_labeled.csv",
                        help="Output pseudo-labeled CSV path")
    parser.add_argument("--prob_threshold", type=float, default=0.90,
                        help="Probability threshold for pseudo-labeling")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Max tokens per window")
    parser.add_argument("--stride", type=int, default=256,
                        help="Stride size for sliding window")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of documents per inference batch")
    args = parser.parse_args()

    # Load unlabeled data with error handling
    try:
        raw_df = pd.read_csv(args.unlabeled_csv)
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return
    if 'title' not in raw_df.columns or 'description' not in raw_df.columns:
        logger.error("CSV must contain 'title' and 'description' columns")
        return

    texts = (raw_df['title'] + ' ' + raw_df['description']).tolist()
    logger.info(f"Loaded {len(texts)} unlabeled samples")

    # Load models and tokenizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    tok_deb = AutoTokenizer.from_pretrained(args.deberta_dir)
    tok_lon = AutoTokenizer.from_pretrained(args.longformer_dir)
    model_deb = AutoModelForSequenceClassification.from_pretrained(args.deberta_dir).to(device).eval()
    model_lon = AutoModelForSequenceClassification.from_pretrained(args.longformer_dir).to(device).eval()

    # Predict logits in batches
    logger.info("Predicting logits with DeBERTa...")
    logits_deb = predict_logits(
        model_deb, tok_deb, texts, device,
        args.max_len, args.stride, args.batch_size
    )
    logger.info("Predicting logits with Longformer...")
    logits_lon = predict_logits(
        model_lon, tok_lon, texts, device,
        args.max_len, args.stride, args.batch_size
    )

    # Ensemble logits and compute probabilities
    logits_ensemble = (logits_deb + logits_lon) / 2
    probs = torch.softmax(torch.from_numpy(logits_ensemble), dim=-1).numpy()
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    # Filter by threshold
    mask = max_probs >= args.prob_threshold
    selected = mask.sum()
    logger.info(f"Selected {selected} / {len(texts)} samples (>= {args.prob_threshold:.2f})")

    # Create and save pseudo-labeled DataFrame
    pseudo_df = raw_df.loc[mask].copy()
    pseudo_df['label'] = preds[mask]
    pseudo_df['confidence'] = max_probs[mask]

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    pseudo_df.to_csv(args.output_csv, index=False)
    logger.info(f"Pseudo-labeled data saved to {args.output_csv}")

if __name__ == "__main__":
    main()
