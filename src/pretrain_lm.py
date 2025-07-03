import os
import argparse
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Initialize logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def load_unlabeled_text_data(file_path):
    if file_path.endswith(".txt"):
        return load_dataset("text", data_files={"train": file_path})
    elif file_path.endswith(".csv"):
        return load_dataset("csv", data_files={"train": file_path})
    else:
        raise ValueError("Unsupported file format. Use .txt or .csv")

def tokenize_function(examples, tokenizer, block_size=512):
    return tokenizer(
        examples["text"],
        return_special_tokens_mask=True,
        truncation=True,
        max_length=block_size
    )

def group_texts(examples, block_size=512):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
        for k, t in concatenated.items()
    }
    return result

def run_dapt(
    model_name,
    data_file,
    output_dir,
    block_size,
    mlm_prob,
    num_train_epochs,
    batch_size,
    learning_rate,
    logging_steps,
    save_steps,
    save_total_limit,
    use_fp16,
):
    # Load tokenizer and model
    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Load and preprocess data
    logger.info("Loading unlabeled data from: %s", data_file)
    dataset = load_unlabeled_text_data(data_file)
    tokenized = dataset["train"].map(
        lambda x: tokenize_function(x, tokenizer, block_size),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    lm_dataset = tokenized.map(
        lambda x: group_texts(x, block_size),
        batched=True,
        desc="Grouping into blocks"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_strategy="steps",
        logging_steps=logging_steps,
        fp16=use_fp16,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train and save
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("DAPT model saved to %s", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain-Adaptive Pre-Training for Masked LM")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/dapt_checkpoints/")
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--use_fp16", action="store_true")
    args = parser.parse_args()

    run_dapt(
        model_name=args.model_name,
        data_file=args.data_file,
        output_dir=args.output_dir,
        block_size=args.block_size,
        mlm_prob=args.mlm_prob,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        use_fp16=args.use_fp16,
    )
