import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def load_agnews_dataset():
    """Load AG News dataset from Hugging Face"""
    return load_dataset("ag_news")


def get_tokenizer(model_name):
    """Load tokenizer given a model name"""
    return AutoTokenizer.from_pretrained(model_name)


def preprocess_function(example, tokenizer, max_length=512, stride=256):
    """
    Tokenize a single example with sliding window
    Applies truncation and overlapping segments if needed
    """
    text = example["title"] + " " + example["description"]
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )
    tokenized["labels"] = [example["label"]] * len(tokenized["input_ids"])
    return tokenized


def tokenize_dataset(dataset, tokenizer, max_length=512, stride=256):
    """Tokenize all splits in the dataset"""
    tokenized_datasets = DatasetDict()
    for split in dataset:
        print(f"Tokenizing: {split}")
        tokenized = dataset[split].map(
            lambda x: preprocess_function(x, tokenizer, max_length, stride),
            batched=False,
            remove_columns=dataset[split].column_names,
        )
        tokenized_datasets[split] = tokenized
    return tokenized_datasets


def save_tokenized_dataset(tokenized_datasets, output_dir="data/interim/"):
    """Save tokenized datasets to disk"""
    os.makedirs(output_dir, exist_ok=True)
    for split in tokenized_datasets:
        path = os.path.join(output_dir, split)
        print(f"Saving to: {path}")
        tokenized_datasets[split].save_to_disk(path)


def prepare_data_pipeline(
    model_name="microsoft/deberta-v3-large",
    max_length=512,
    stride=256,
    output_dir="data/interim/"
):
    """Run full pipeline: load → tokenize → save"""
    dataset = load_agnews_dataset()
    tokenizer = get_tokenizer(model_name)
    tokenized = tokenize_dataset(dataset, tokenizer, max_length, stride)
    save_tokenized_dataset(tokenized, output_dir)
    print("Data preprocessing complete")


if __name__ == "__main__":
    prepare_data_pipeline()