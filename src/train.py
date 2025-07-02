import os
import argparse
import logging
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer

from src.utils import configure_logger, set_global_seed, compute_metrics
from src.deberta_lora import get_deberta_lora_model, DebertaLoraConfig
from src.longformer_lora import get_longformer_lora_model, LongformerLoraConfig

# Set up logger and fix seed
logger = configure_logger("results/logs/train.log")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa & Longformer with LoRA on AG News")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--longformer_name", type=str, default="allenai/longformer-large-4096")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length for tokenization")
    parser.add_argument("--stride", type=int, default=256, help="Stride size for sliding window tokenization")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    args = parser.parse_args()

    # Set seeds
    set_global_seed(args.seed)
    logger.info(f"Seed set to {args.seed}")

    # Load dataset and tokenizer
    logger.info("Loading AG News dataset...")
    dataset = load_dataset("ag_news")  # returns DatasetDict
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Tokenization with sliding window for long texts
    def preprocess_fn(examples):
        texts = [t + " " + d for t, d in zip(examples['title'], examples['description'])]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=args.max_length,
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=False
        )
        labels = []
        for i in range(len(tokenized['input_ids'])):
            sample_index = tokenized['overflow_to_sample_mapping'][i]
            labels.append(examples['label'][sample_index])
        tokenized['labels'] = labels
        return tokenized

    logger.info("Tokenizing dataset...")
    tokenized = DatasetDict({
        split: dataset[split].map(
            preprocess_fn,
            batched=True,
            remove_columns=['title', 'description', 'label'],
        )
        for split in ['train', 'test']
    })
    tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Prepare models with LoRA
    deb_cfg = DebertaLoraConfig(model_name=args.model_name, num_labels=4)
    model_deberta = get_deberta_lora_model(deb_cfg)

    lon_cfg = LongformerLoraConfig(model_name=args.longformer_name, num_labels=4)
    model_longformer = get_longformer_lora_model(lon_cfg)

    # Common training arguments
    os.makedirs('results', exist_ok=True)
    total_steps = (len(tokenized['train']) // args.batch_size + 1) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    training_args = TrainingArguments(
        output_dir='results',
        evaluation_strategy='steps',  # evaluate by steps for finer control
        save_strategy='steps',
        save_steps=warmup_steps,      # save at end of warmup
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,  # accumulate gradients for effective larger batch
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,    # linear warmup
        logging_dir='results/logs',
        logging_steps=warmup_steps//2,
        fp16=True,                     # mixed precision
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True
    )

    # Train DeBERTa-LoRA
    logger.info("[DeBERTa-LoRA] Training started with warmup_steps=%d, fp16=True", warmup_steps)
    trainer_deberta = Trainer(
        model=model_deberta,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logger.info("[DeBERTa-LoRA] Training started")
    trainer_deberta = Trainer(
        model=model_deberta,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer_deberta.train()
    trainer_deberta.save_model('results/deberta_lora')
    logger.info("[DeBERTa-LoRA] Model saved to results/deberta_lora")

    # Train Longformer-LoRA
    logger.info("[Longformer-LoRA] Training started")
    trainer_longformer = Trainer(
        model=model_longformer,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer_longformer.train()
    trainer_longformer.save_model('results/longformer_lora')
    logger.info("[Longformer-LoRA] Model saved to results/longformer_lora")

    logger.info("Training pipeline completed.")

if __name__ == '__main__':
    main()
