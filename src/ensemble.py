import os
import argparse
import numpy as np
import torch
from datasets import load_from_disk
from typing import Tuple

from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm

from src.utils import (
    configure_logger,
    prepare_dataloader,
    compute_metrics,
    print_classification_report,
)

# Set up logger
logger = configure_logger("outputs/logs/ensemble.log")


def load_model(model_dir: str, device: torch.device) -> torch.nn.Module:
    # Load fine-tuned model for inference
    logger.info(f"Loading model from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model


def inference(
    model: torch.nn.Module,
    dataloader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    # Run inference (optionally with AMP) returning logits and labels arrays
    all_logits, all_labels = [], []
    use_amp = device.type == "cuda"

    for batch in tqdm(dataloader, desc="Inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].cpu().numpy()

        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits.cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels)

    return np.vstack(all_logits), np.concatenate(all_labels)


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble DeBERTa & Longformer via weighted soft-voting"
    )
    parser.add_argument("--deberta_dir", type=str, required=True)
    parser.add_argument("--longformer_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/interim")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_deberta", type=float, default=0.5)
    parser.add_argument("--weight_longformer", type=float, default=0.5)
    parser.add_argument("--save_logits", type=str, default="")
    parser.add_argument("--class_names", nargs="+", default=["World", "Sports", "Business", "Sci/Tech"])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    # Load test dataset
    ds_test = load_from_disk(os.path.join(args.data_dir, "test"))
    logger.info("Loaded test dataset with %d samples", len(ds_test))
    dataloader = prepare_dataloader(ds_test, args.batch_size)

    # Load models
    model_deberta = load_model(args.deberta_dir, device)
    model_longformer = load_model(args.longformer_dir, device)

    # Inference
    logger.info("Running inference with DeBERTa...")
    logits_deberta, labels = inference(model_deberta, dataloader, device)

    logger.info("Running inference with Longformer...")
    logits_longformer, _ = inference(model_longformer, dataloader, device)

    # Normalize and ensemble logits
    total_weight = args.weight_deberta + args.weight_longformer
    w_deb = args.weight_deberta / total_weight
    w_lon = args.weight_longformer / total_weight
    logger.info("Ensembling outputs (weights: DeBERTa=%.2f, Longformer=%.2f)", w_deb, w_lon)
    logits_ensemble = w_deb * logits_deberta + w_lon * logits_longformer

    # Compute and log metrics
    metrics = compute_metrics(logits_ensemble, labels)
    logger.info("Ensemble Metrics: %s", metrics)

    # Print classification report
    print_classification_report(logits_ensemble, labels, tuple(args.class_names))

    # Save logits if requested
    if args.save_logits:
        os.makedirs(args.save_logits, exist_ok=True)
        np.save(os.path.join(args.save_logits, "logits_deberta.npy"), logits_deberta)
        np.save(os.path.join(args.save_logits, "logits_longformer.npy"), logits_longformer)
        np.save(os.path.join(args.save_logits, "logits_ensemble.npy"), logits_ensemble)
        np.save(os.path.join(args.save_logits, "labels.npy"), labels)
        logger.info("Saved logits and labels to %s", args.save_logits)


if __name__ == "__main__":
    main()
