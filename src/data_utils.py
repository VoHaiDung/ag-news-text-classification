import os
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm


def load_agnews_dataset():
    """
    Load the AG News dataset using the Hugging Face Datasets library.
    Returns:
        DatasetDict: A dictionary containing 'train' and 'test' splits.
    """
    dataset = load_dataset("ag_news")
    return dataset


def get_tokenizer(model_name):
    """
    Load a pre-trained tokenizer for a specified Transformer model.

    Args:
        model_name (str): The Hugging Face model identifier (e.g., 'microsoft/deberta-v3-large').

    Returns:
        PreTrainedTokenizer: The corresponding tokenizer object.
    """
    return AutoTokenizer.from_pretrained(model_name)


def preprocess_function(example, tokenizer, max_length=512, stride=256):
    """
    Tokenize a single AG News example using a sliding window strategy.
    This approach ensures that long sequences exceeding max_length are
    divided into overlapping segments.

    Args:
        example (dict): A single input example with 'title', 'description', and 'label'.
        tokenizer (PreTrainedTokenizer): Tokenizer to apply.
        max_length (int): Maximum number of tokens per segment.
        stride (int): Overlap between consecutive segments.

    Returns:
        dict: Tokenized output including overflow segments and repeated labels.
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

    # Duplicate the label for each overflow segment
    tokenized["labels"] = [example["label"]] * len(tokenized["input_ids"])
    return tokenized


def tokenize_dataset(dataset, tokenizer, max_length=512, stride=256):
    """
    Apply the preprocessing function to all splits of the dataset.

    Args:
        dataset (DatasetDict): Dataset with train/test splits.
        tokenizer (PreTrainedTokenizer): Tokenizer object.
        max_length (int): Maximum length for each tokenized input.
        stride (int): Overlap between segments when splitting long sequences.

    Returns:
        DatasetDict: Tokenized version of the input dataset.
    """
    tokenized_datasets = DatasetDict()
    for split in dataset:
        print(f"Tokenizing split: {split}")
        tokenized = dataset[split].map(
            lambda x: preprocess_function(x, tokenizer, max_length, stride),
            batched=False,
            remove_columns=dataset[split].column_names,
        )
        tokenized_datasets[split] = tokenized
    return tokenized_datasets


def save_tokenized_dataset(tokenized_datasets, output_dir="data/interim/"):
    """
    Save the tokenized dataset to disk in Arrow format.

    Args:
        tokenized_datasets (DatasetDict): Tokenized datasets.
        output_dir (str): Directory where the processed datasets will be stored.
    """
    os.makedirs(output_dir, exist_ok=True)
    for split in tokenized_datasets:
        save_path = os.path.join(output_dir, split)
        print(f"Saving {split} split to {save_path}")
        tokenized_datasets[split].save_to_disk(save_path)


def prepare_data_pipeline(
    model_name="microsoft/deberta-v3-large",
    max_length=512,
    stride=256,
    output_dir="data/interim/"
):
    """
    End-to-end data pipeline that loads, tokenizes, and saves the AG News dataset.

    Args:
        model_name (str): Name of the pre-trained model for tokenizer loading.
        max_length (int): Maximum token length for input sequences.
        stride (int): Overlap used for the sliding window during tokenization.
        output_dir (str): Path to store the processed dataset.
    """
    dataset = load_agnews_dataset()
    tokenizer = get_tokenizer(model_name)
    tokenized = tokenize_dataset(dataset, tokenizer, max_length, stride)
    save_tokenized_dataset(tokenized, output_dir)
    print("Data preprocessing complete.")


if __name__ == "__main__":
    prepare_data_pipeline()
