import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np
import os

# Constants
MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 4
MAX_LENGTH = 512
STRIDE = 128
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./results"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "final_model")


def main():
    # 0. Make sure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)
    
    # 1. Load AG News dataset
    dataset = load_dataset("ag_news")

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Preprocess with sliding-window
    def preprocess_function(examples):
        texts = examples["text"]
        labels = examples["label"]
        out_input_ids = []
        out_attention_mask = []
        out_labels = []

        for text, label in zip(texts, labels):
            # full tokenization without truncation
            enc = tokenizer(text, return_attention_mask=True, return_offsets_mapping=False, truncation=False)
            ids = enc["input_ids"]
            mask = enc["attention_mask"]
            total_len = len(ids)
            start = 0

            while start < total_len:
                end = min(start + MAX_LENGTH, total_len)
                chunk_ids = ids[start:end]
                chunk_mask = mask[start:end]

                # pad if needed
                pad_len = MAX_LENGTH - len(chunk_ids)
                if pad_len > 0:
                    chunk_ids = chunk_ids + [tokenizer.pad_token_id] * pad_len
                    chunk_mask = chunk_mask + [0] * pad_len

                out_input_ids.append(chunk_ids)
                out_attention_mask.append(chunk_mask)
                out_labels.append(label)

                if end == total_len:
                    break
                start += (MAX_LENGTH - STRIDE)

        return {
            "input_ids": out_input_ids,
            "attention_mask": out_attention_mask,
            "labels": out_labels,
        }

    # apply mapping to both train and test splits
    tokenized = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text", "label"],
    )
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 4. Load DeBERTa-v3 model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    # 5. Define evaluation metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
            "precision": precision_metric.compute(predictions=predictions, references=labels, average="macro")["precision"],
            "recall": recall_metric.compute(predictions=predictions, references=labels, average="macro")["recall"],
            "f1": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"],
        }

    # 6. Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # 7. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 8. Train
    trainer.train()

    # 9. Evaluate
    eval_result = trainer.evaluate()
    print("Final Evaluation:")
    for k, v in eval_result.items():
        print(f"{k}: {v:.4f}")

    # 10. Save model and tokenizer
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"Model and tokenizer saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
