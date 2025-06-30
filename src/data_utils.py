import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Configure logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def combine_title_description(title: str, description: str) -> str:
    return f"{title.strip()} {description.strip()}"


@dataclass(frozen=True)
class DataConfig:
    model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 512
    stride: int = 256
    output_dir: str = "data/interim/"


def load_agnews_dataset() -> DatasetDict:
    logger.info("Loading AG News dataset from Hugging Face Datasets...")
    dataset = load_dataset("ag_news")
    logger.info("Dataset loaded: %d train examples, %d test examples.",
                len(dataset["train"]), len(dataset["test"]))
    return dataset


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    logger.info("Loading tokenizer for model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


def preprocess_function(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int
) -> Dict[str, List]:
    # Combine title + description
    texts = [combine_title_description(t, d) for t, d in zip(examples["title"], examples["description"])]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=False,
        padding="max_length",
    )

    # Align labels with overflow
    labels: List[int] = []
    for idx in range(len(tokenized["input_ids"])):
        sample_index = tokenized["overflow_to_sample_mapping"][idx]
        labels.append(examples["label"][sample_index])
    tokenized["labels"] = labels

    # Optional sanity check
    max_len = max(len(ids) for ids in tokenized["input_ids"])
    if max_len < max_length:
        logger.debug(
            "Max token length after tokenization (%d) < configured max_length (%d)",
            max_len, max_length
        )

    return tokenized


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int
) -> DatasetDict:
    tokenized_datasets = DatasetDict()
    for split, ds in dataset.items():
        logger.info("Tokenizing split: %s (max_length=%d, stride=%d)", split, max_length, stride)
        tokenized = ds.map(
            lambda ex: preprocess_function(ex, tokenizer, max_length, stride),
            batched=True,
            remove_columns=ds.column_names,
            desc=f"Tokenizing {split}"
        )
        tokenized_datasets[split] = tokenized
    return tokenized_datasets


def save_tokenized_dataset(
    tokenized_datasets: DatasetDict,
    output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for split, ds in tokenized_datasets.items():
        path = os.path.join(output_dir, split)
        logger.info("Saving %s split to %s", split, path)
        ds.save_to_disk(path)


def prepare_data_pipeline(
    config: Optional[DataConfig] = None
) -> None:
    cfg = config or DataConfig()
    # Load and preprocess
    dataset = load_agnews_dataset()
    tokenizer = get_tokenizer(cfg.model_name)
    tokenized = tokenize_dataset(dataset, tokenizer, cfg.max_length, cfg.stride)
    save_tokenized_dataset(tokenized, cfg.output_dir)
    logger.info("Data preprocessing pipeline complete.")


if __name__ == "__main__":
    prepare_data_pipeline()
