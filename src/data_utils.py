import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset, DownloadMode, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.utils import configure_logger

# Initialize logger
logger = configure_logger("results/logs/data_utils.log")

@dataclass(frozen=True)
class DataConfig:
    model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 512
    stride: int = 256
    output_dir: str = "data/interim/"

# Combine title and description into single text
def combine_title_description(title: str, description: str) -> str:
    return f"{title.strip()} {description.strip()}"

# Load AG News dataset
def load_agnews_dataset() -> DatasetDict:
    logger.info("Loading AG News dataset...")
    ds = load_dataset(
        "ag_news",
        cache_dir="/content/cache",
        download_mode="force_redownload"
    )
    logger.info(f"Loaded AG News: train={len(ds['train'])}, test={len(ds['test'])}")
    return ds

# Get tokenizer for given model
def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    logger.info(f"Loading tokenizer: {model_name}")
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Preprocess with sliding window and align labels
def preprocess_function(
    examples: Dict[str, List],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int
) -> Dict[str, List]:
    texts = [combine_title_description(t, d) for t, d in zip(examples['title'], examples['description'])]
    tok = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=False
    )
    labels = [examples['label'][i] for i in tok['overflow_to_sample_mapping']]
    tok['labels'] = labels
    return tok

# Tokenize entire dataset
def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    stride: int
) -> DatasetDict:
    td = DatasetDict()
    for split in dataset:
        logger.info(f"Tokenizing {split} (max_len={max_length}, stride={stride})")
        td[split] = dataset[split].map(
            lambda ex: preprocess_function(ex, tokenizer, max_length, stride),
            batched=True,
            remove_columns=dataset[split].column_names
        )
    return td

# Save tokenized datasets to disk
def save_tokenized_dataset(
    tokenized: DatasetDict,
    output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for split, ds in tokenized.items():
        path = os.path.join(output_dir, split)
        logger.info(f"Saving {split} to {path}")
        ds.save_to_disk(path)

# CLI entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare tokenized AG News dataset")
    parser.add_argument("--model_name", type=str, default=DataConfig.model_name)
    parser.add_argument("--max_length", type=int, default=DataConfig.max_length)
    parser.add_argument("--stride", type=int, default=DataConfig.stride)
    parser.add_argument("--output_dir", type=str, default=DataConfig.output_dir)
    args = parser.parse_args()

    cfg = DataConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        stride=args.stride,
        output_dir=args.output_dir
    )
    ds = load_agnews_dataset()
    tokenizer = get_tokenizer(cfg.model_name)
    tokenized = tokenize_dataset(ds, tokenizer, cfg.max_length, cfg.stride)
    save_tokenized_dataset(tokenized, cfg.output_dir)
    logger.info("Data preprocessing complete.")
