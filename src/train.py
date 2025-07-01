import os
import logging
import numpy as np
import torch
from datasets import DatasetDict
from transformers import AutoTokenizer, TrainingArguments, Trainer

from src.data_utils import load_agnews_dataset, get_tokenizer, tokenize_dataset, DataConfig
from src.deberta_lora import get_deberta_lora_model, DebertaLoraConfig
from src.longformer_lora import get_longformer_lora_model, LongformerLoraConfig
import evaluate

# Set up logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": evaluate.load("accuracy").compute(predictions=preds, references=labels)["accuracy"],
        "precision": evaluate.load("precision").compute(predictions=preds, references=labels, average="macro")["precision"],
        "recall": evaluate.load("recall").compute(predictions=preds, references=labels, average="macro")["recall"],
        "f1": evaluate.load("f1").compute(predictions=preds, references=labels, average="macro")["f1"],
    }


def main():
    # Load and preprocess dataset
    cfg = DataConfig(
        model_name="microsoft/deberta-v3-large",
        max_length=512,
        stride=256
    )
    dataset = load_agnews_dataset()            # theo Hugging Face
    tokenizer = get_tokenizer(cfg.model_name)
    tokenized = tokenize_dataset(dataset, tokenizer, cfg.max_length, cfg.stride)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Initialize LoRA models
    deb_cfg = DebertaLoraConfig(
        model_name=cfg.model_name,
        num_labels=4
    )
    model_deberta = get_deberta_lora_model(deb_cfg)

    lon_cfg = LongformerLoraConfig(
        model_name="allenai/longformer-large-4096",
        num_labels=4
    )
    model_longformer = get_longformer_lora_model(lon_cfg)

    # Shared training arguments
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Train DeBERTa-LoRA
    logger.info("[DeBERTa-LoRA] Training started")
    trainer_deberta = Trainer(
        model=model_deberta,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer_deberta.train()
    trainer_deberta.save_model(os.path.join(output_dir, "deberta_lora"))
    logger.info("[DeBERTa-LoRA] Model saved to results/deberta_lora")

    # Train Longformer-LoRA
    logger.info("[Longformer-LoRA] Training started")
    trainer_longformer = Trainer(
        model=model_longformer,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer_longformer.train()
    trainer_longformer.save_model(os.path.join(output_dir, "longformer_lora"))
    logger.info("[Longformer-LoRA] Model saved to results/longformer_lora")

    logger.info("Training pipeline completed.")


if __name__ == "__main__":
    main()
