import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def load_agnews_dataset():
    """
    Load the AG News dataset using the Hugging Face Datasets library.

    Returns:
        DatasetDict: A dictionary containing 'train' and 'test' splits.
    """
    return load_dataset("ag_news")


def get_tokenizer(model_name):
    """
    Load a pre-trained tokenizer for a specified Transformer model.

    Args:
        model_name (str): The Hugging Face model identifier.

    Returns:
        PreTrainedTokenizer: The corresponding tokenizer object.
    """
    return AutoTokenizer.from_pretrained(model_name)


def preprocess_function(examples, tokenizer, max_length=512, stride=256):
    """
    Tokenize examples using sliding window strategy for long inputs.

    Args:
        examples (dict): A batch of input examples with 'title', 'description', and 'label'.
        tokenizer (PreTrainedTokenizer): Tokenizer to apply.
        max_length (int): Max token length per segment.
        stride (int): Overlap between segments.

    Returns:
        dict: Tokenized batch including overflow segments and labels.
    """
    texts = [t + " " + d for t, d in zip(examples["title"], examples["description"])]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=False,
        padding="max_length"
    )

    # Repeat labels to match number of overflowed inputs
    labels = []
    for i in range(len(tokenized["input_ids"])):
        sample_idx = tokenized["overflow_to_sample_mapping"][i]
        labels.append(examples["label"][sample_idx])
    tokenized["labels"] = labels

    return tokenized


def tokenize_dataset(dataset, tokenizer, max_length=512, stride=256):
    """
    Apply preprocessing to all splits using batched mapping for speed.

    Args:
        dataset (DatasetDict): Dataset with 'train' and 'test' splits.
        tokenizer (PreTrainedTokenizer): Tokenizer object.
        max_length (int): Max sequence length.
        stride (int): Overlap size for sliding window.

    Returns:
        DatasetDict: Tokenized dataset.
    """
    tokenized_datasets = DatasetDict()
    for split in dataset:
        print(f"Tokenizing split: {split}")
        tokenized = dataset[split].map(
            lambda x: preprocess_function(x, tokenizer, max_length, stride),
            batched=True,
            remove_columns=dataset[split].column_names,
            desc=f"Tokenizing {split}"
        )
        tokenized_datasets[split] = tokenized
    return tokenized_datasets


def save_tokenized_dataset(tokenized_datasets, output_dir="data/interim/"):
    """
    Save the tokenized datasets to disk in Arrow format.

    Args:
        tokenized_datasets (DatasetDict): Tokenized dataset.
        output_dir (str): Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for split in tokenized_datasets:
        path = os.path.join(output_dir, split)
        print(f"Saving {split} split to {path}")
        tokenized_datasets[split].save_to_disk(path)


def prepare_data_pipeline(
    model_name="microsoft/deberta-v3-large",
    max_length=512,
    stride=256,
    output_dir="data/interim/"
):
    """
    Full pipeline to load, tokenize, and save AG News dataset.

    Args:
        model_name (str): Model name for tokenizer.
        max_length (int): Max sequence length.
        stride (int): Sliding window stride.
        output_dir (str): Output path for processed data.
    """
    dataset = load_agnews_dataset()
    tokenizer = get_tokenizer(model_name)
    tokenized = tokenize_dataset(dataset, tokenizer, max_length, stride)
    save_tokenized_dataset(tokenized, output_dir)
    print("Data preprocessing complete.")


if __name__ == "__main__":
    prepare_data_pipeline()
