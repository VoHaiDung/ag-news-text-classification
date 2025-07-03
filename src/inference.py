import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple

from src.utils import configure_logger

# Initialize logger
logger = configure_logger("results/logs/inference.log")

def predict_logits(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    max_len: int,
    stride: int,
    batch_size: int
) -> np.ndarray:
    # Perform sliding-window inference to return logits for each document
    model.eval()
    use_amp = device.type == "cuda"
    all_logits = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Batch inference"):
        batch_texts = texts[start:start + batch_size]
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
    parser = argparse.ArgumentParser(description="Inference pipeline: DeBERTa, Longformer, ensemble or stacking")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV with 'title' and 'description'")
    parser.add_argument("--output_csv", type=str, default="outputs/predictions.csv", help="Output CSV with predictions")
    parser.add_argument("--deberta_dir", type=str, required=True, help="Path to DeBERTa-LoRA model")
    parser.add_argument("--longformer_dir", type=str, required=True, help="Path to Longformer-LoRA model")
    parser.add_argument("--use_stacking", action="store_true", help="Use stacking model instead of soft-voting")
    parser.add_argument("--stacking_model", type=str, help="Path to stacking joblib model")
    parser.add_argument("--weight_deberta", type=float, default=0.5, help="Weight for DeBERTa logits in soft-voting")
    parser.add_argument("--weight_longformer", type=float, default=0.5, help="Weight for Longformer logits in soft-voting")
    parser.add_argument("--max_len", type=int, default=512, help="Max tokens per window for DeBERTa inference")
    parser.add_argument("--stride", type=int, default=256, help="Stride size for sliding window")
    parser.add_argument("--batch_size", type=int, default=8, help="Documents per inference batch")
    parser.add_argument("--class_names", nargs="+", default=["World","Sports","Business","Sci/Tech"], help="List of class names")
    args = parser.parse_args()

    # Load input data
    df = pd.read_csv(args.input_csv)
    texts = (df['title'] + ' ' + df['description']).tolist()
    logger.info(f"Loaded {len(texts)} samples for inference")

    # Device and models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # DeBERTa inference
    tok_deb = AutoTokenizer.from_pretrained(args.deberta_dir)
    model_deb = AutoModelForSequenceClassification.from_pretrained(args.deberta_dir).to(device)
    logger.info("Running DeBERTa inference...")
    logits_deb = predict_logits(model_deb, tok_deb, texts, device, args.max_len, args.stride, args.batch_size)

    # Longformer inference
    tok_lon = AutoTokenizer.from_pretrained(args.longformer_dir)
    model_lon = AutoModelForSequenceClassification.from_pretrained(args.longformer_dir).to(device)
    logger.info("Running Longformer inference...")
    logits_lon = predict_logits(model_lon, tok_lon, texts, device, args.max_len, args.stride, args.batch_size)

    # Determine final predictions
    if args.use_stacking:
        import joblib
        stk = joblib.load(args.stacking_model)
        features = np.concatenate([logits_deb, logits_lon], axis=1)
        probs = stk.predict_proba(features)
        logger.info("Using stacking model for final predictions")
    else:
        # Soft-voting ensemble
        w_sum = args.weight_deberta + args.weight_longformer
        w_deb = args.weight_deberta / w_sum
        w_lon = args.weight_longformer / w_sum
        probs = (w_deb * logits_deb + w_lon * logits_lon)
        probs = torch.softmax(torch.from_numpy(probs), dim=-1).numpy()
        logger.info("Using soft-voting ensemble for final predictions")

    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    # Save output CSV
    df_out = df.copy()
    df_out['predicted_label'] = preds
    df_out['confidence'] = confidences
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    logger.info(f"Saved predictions to {args.output_csv}")

if __name__ == '__main__':
    main()
