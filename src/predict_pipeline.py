import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

from src.utils import configure_logger

# Initialize logger
logger = configure_logger("results/logs/predict_pipeline.log")

def sliding_window_logits(model, tokenizer, texts, device, max_len, stride, batch_size):
    # Generate logits via sliding-window for a list of texts
    model.eval()
    use_amp = device.type == "cuda"
    all_logits = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Inference batches"):
        batch_texts = texts[start:start + batch_size]
        for text in batch_texts:
            tok = tokenizer(
                text,
                max_length=max_len,
                stride=stride,
                truncation=True,
                padding="max_length",
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            logits_chunks = []
            for i in range(tok['input_ids'].size(0)):
                inputs = tok['input_ids'][i].unsqueeze(0).to(device)
                masks = tok['attention_mask'][i].unsqueeze(0).to(device)
                with torch.no_grad():
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            out = model(inputs, attention_mask=masks)
                    else:
                        out = model(inputs, attention_mask=masks)
                logits_chunks.append(out.logits.cpu())
            mean_logit = torch.stack(logits_chunks).mean(dim=0)
            all_logits.append(mean_logit.squeeze(0).numpy())
    return np.vstack(all_logits)


def main():
    parser = argparse.ArgumentParser(description="Full predict pipeline: ensemble or stacking inference")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV with 'title' and 'description'")
    parser.add_argument("--output_csv", type=str, default="outputs/predictions_pipeline.csv", help="Output CSV path")
    parser.add_argument("--deberta_dir", type=str, required=True, help="DeBERTa-LoRA model path")
    parser.add_argument("--longformer_dir", type=str, required=True, help="Longformer-LoRA model path")
    parser.add_argument("--max_len", type=int, default=512, help="Max seq length for DeBERTa inference")
    parser.add_argument("--stride", type=int, default=256, help="Sliding window stride for DeBERTa")
    parser.add_argument("--batch_size", type=int, default=8, help="Documents per batch")
    parser.add_argument("--use_stacking", action="store_true", help="Use stacking model for final predictions")
    parser.add_argument("--stacking_model", type=str, help="Path to stacking joblib model")
    parser.add_argument("--weight_deberta", type=float, default=0.5, help="Weight for DeBERTa in soft-voting")
    parser.add_argument("--weight_longformer", type=float, default=0.5, help="Weight for Longformer in soft-voting")
    parser.add_argument("--class_names", nargs="+", default=["World","Sports","Business","Sci/Tech"], help="List of class names")
    args = parser.parse_args()

    # Read input data
    df = pd.read_csv(args.input_csv)
    texts = (df['title'] + ' ' + df['description']).tolist()
    logger.info(f"Loaded {len(texts)} texts from {args.input_csv}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # DeBERTa inference
    deb_tok = AutoTokenizer.from_pretrained(args.deberta_dir)
    deb_model = AutoModelForSequenceClassification.from_pretrained(args.deberta_dir).to(device)
    logger.info("Running DeBERTa inference...")
    deb_logits = sliding_window_logits(deb_model, deb_tok, texts, device, args.max_len, args.stride, args.batch_size)

    # Longformer inference
    lon_tok = AutoTokenizer.from_pretrained(args.longformer_dir)
    lon_model = AutoModelForSequenceClassification.from_pretrained(args.longformer_dir).to(device)
    logger.info("Running Longformer inference...")
    # Use larger window for Longformer if desired
    long_max = args.max_len * 8
    long_stride = args.stride * 4
    lon_logits = sliding_window_logits(lon_model, lon_tok, texts, device, long_max, long_stride, args.batch_size)

    # Final ensemble or stacking
    if args.use_stacking and args.stacking_model:
        logger.info("Applying stacking model...")
        stk = joblib.load(args.stacking_model)
        features = np.concatenate([deb_logits, lon_logits], axis=1)
        probs = stk.predict_proba(features)
    else:
        logger.info("Applying soft-voting ensemble...")
        w_sum = args.weight_deberta + args.weight_longformer
        w_deb = args.weight_deberta / w_sum
        w_lon = args.weight_longformer / w_sum
        ensemble_logits = w_deb * deb_logits + w_lon * lon_logits
        probs = torch.softmax(torch.from_numpy(ensemble_logits), dim=-1).numpy()

    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    # Display detailed results for each text
    class_names = args.class_names
    for text, prob in zip(texts, probs):
        pred_idx = prob.argmax()
        pred_label = class_names[pred_idx]
        print(f"Input Text:\n“{text}”\n")
        print(f"Predict: {pred_label}\n")
        print("Probabilities:")
        for i, name in enumerate(class_names):
            print(f"- {name:<10} {prob[i]:.2f}")
        print("\n" + "-" * 50 + "\n")
    
    # Save to CSV
    df["predicted_label"] = preds
    df["confidence"] = conf
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()
