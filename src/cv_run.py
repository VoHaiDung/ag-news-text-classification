import os
import argparse
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer

from src.utils import configure_logger, set_global_seed, compute_metrics
from src.deberta_lora import get_deberta_lora_model, DebertaLoraConfig
from src.longformer_lora import get_longformer_lora_model, LongformerLoraConfig

# Initialize logger
logger = configure_logger("results/logs/cv_run.log")

def preprocess_split(dataset, tokenizer, max_length, stride):
    # Sliding-window tokenization combining title + description
    def _fn(examples):
        texts = [t + " " + d for t, d in zip(examples['title'], examples['description'])]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False
        )
        labels = [examples['label'][i] for i in tokenized['overflow_to_sample_mapping']]
        tokenized['labels'] = labels
        return tokenized

    ds = dataset.map(_fn, batched=True, remove_columns=['title','description','label'])
    ds.set_format('torch', columns=['input_ids','attention_mask','labels'])
    return ds


def run_cv(args):
    set_global_seed(args.seed)
    # Load full dataset
    ds_all = load_dataset("ag_news")
    texts = np.array(ds_all['train']['title'], dtype=object)  # placeholder for index length
    labels = np.array(ds_all['train']['label'])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        logger.info(f"Starting fold {fold}/{args.folds}")
        # Prepare train/val splits
        ds_train = ds_all['train'].select(train_idx)
        ds_val = ds_all['train'].select(val_idx)
        ds_split = DatasetDict({'train': ds_train, 'test': ds_val})

        # Tokenization
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        tokenized = DatasetDict({
            split: preprocess_split(ds_split[split], tokenizer, args.max_length, args.stride)
            for split in ds_split
        })

        # Initialize models
        deb_cfg = DebertaLoraConfig(model_name=args.model_name, num_labels=4, r=args.r,
                                   lora_alpha=args.alpha, lora_dropout=args.dropout)
        model_deberta = get_deberta_lora_model(deb_cfg)
        lon_cfg = LongformerLoraConfig(model_name=args.longformer_name, num_labels=4,
                                       r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout)
        model_longformer = get_longformer_lora_model(lon_cfg)

        # Training arguments
        output_dir = os.path.join(args.output_dir, f"fold_{fold}")
        os.makedirs(output_dir, exist_ok=True)
        total_steps = (len(ds_train) // args.batch_size + 1) * args.epochs
        warmup_steps = int(0.1 * total_steps)
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy='steps',
            save_strategy='no',
            logging_steps=warmup_steps,
            warmup_steps=warmup_steps,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            gradient_accumulation_steps=2,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            fp16=True,
            disable_tqdm=True
        )

        # Trainer for DeBERTa
        trainer_deb = Trainer(
            model=model_deberta,
            args=training_args,
            train_dataset=tokenized['train'],
            eval_dataset=tokenized['test'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer_deb.train()
        metrics_deb = trainer_deb.evaluate()

        # Trainer for Longformer
        trainer_lon = Trainer(
            model=model_longformer,
            args=training_args,
            train_dataset=tokenized['train'],
            eval_dataset=tokenized['test'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer_lon.train()
        metrics_lon = trainer_lon.evaluate()

        # Ensemble metrics: average accuracy
        acc = (metrics_deb['eval_accuracy'] + metrics_lon['eval_accuracy']) / 2
        logger.info(f"Fold {fold} metrics: DeBERTa={metrics_deb['eval_accuracy']:.4f}, Longformer={metrics_lon['eval_accuracy']:.4f}, Ensemble avg={acc:.4f}")
        fold_metrics.append(acc)

    # Summarize
    mean_acc = np.mean(fold_metrics)
    std_acc = np.std(fold_metrics)
    logger.info(f"Cross-validation accuracy: mean={mean_acc:.4f}, std={std_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="K-fold cross-validation for AG News classification")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--longformer_name", type=str, default="allenai/longformer-large-4096")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="results/cv")
    args = parser.parse_args()
    run_cv(args)

if __name__ == "__main__":
    main()
