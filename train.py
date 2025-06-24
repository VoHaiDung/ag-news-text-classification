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
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 4
OUTPUT_DIR = "./results"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "final_model")
MAX_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

def main():
    # 0. Make sure directories exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)
    
    # 1. Load AG News dataset
    dataset = load_dataset("ag_news")

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. Preprocess function
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 4. Load pre-trained BERT model
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
            "f1": f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
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

    # 7. Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 8. Train the model
    trainer.train()

    # 9. Evaluate the model
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
