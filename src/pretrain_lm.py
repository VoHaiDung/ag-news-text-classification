import os
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset


def load_unlabeled_text_data(file_path):
    """
    Load unlabeled text data for domain-adaptive pretraining.

    Args:
        file_path (str): Path to text file (.txt or .csv with 'text' column).

    Returns:
        Dataset: Hugging Face Dataset object.
    """
    if file_path.endswith(".txt"):
        return load_dataset("text", data_files={"train": file_path})
    elif file_path.endswith(".csv"):
        return load_dataset("csv", data_files={"train": file_path})
    else:
        raise ValueError("Unsupported file format. Use .txt or .csv")


def tokenize_function(examples, tokenizer, block_size=512):
    """
    Tokenize text and chunk into block_size segments.

    Args:
        examples (dict): Dictionary with 'text'.
        tokenizer: Hugging Face tokenizer.
        block_size (int): Length of token blocks.

    Returns:
        dict: Tokenized inputs.
    """
    return tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, max_length=block_size)


def group_texts(examples, block_size=512):
    """
    Concatenate texts and split into chunks of block_size for MLM.

    Args:
        examples (dict): Tokenized inputs.
        block_size (int): Token length per segment.

    Returns:
        dict: Chunked input_ids and attention_mask.
    """
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = (len(concatenated["input_ids"]) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
        for k, t in concatenated.items()
    }
    return result


def run_dapt(
    model_name="microsoft/deberta-v3-large",
    data_file="data/external/unlabeled.txt",
    output_dir="outputs/dapt_checkpoints/",
    block_size=512,
    mlm_prob=0.15,
    num_train_epochs=5,
    batch_size=8,
    learning_rate=5e-5,
):
    """
    Run Domain-Adaptive Pretraining (DAPT) on unlabeled news corpus.

    Args:
        model_name (str): Pretrained model name.
        data_file (str): Path to unlabeled text file (.txt or .csv).
        output_dir (str): Where to save the adapted checkpoint.
        block_size (int): Max tokens per segment.
        mlm_prob (float): Probability of masking tokens.
        num_train_epochs (int): Number of training epochs.
        batch_size (int): Per-device batch size.
        learning_rate (float): Learning rate.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

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

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_total_limit=1,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"DAPT model saved to {output_dir}")


if __name__ == "__main__":
    run_dapt()
