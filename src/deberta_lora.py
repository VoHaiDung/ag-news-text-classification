import argparse
from dataclasses import dataclass
from typing import Optional

import logging
from transformers import AutoModelForSequenceClassification, PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType

from src.utils import configure_logger

# Initialize logger
logger = configure_logger("results/logs/deberta_lora.log")

@dataclass(frozen=True)
class DebertaLoraConfig:
    model_name: str = "microsoft/deberta-v3-large"
    num_labels: int = 4
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: Optional[list] = None


def get_deberta_lora_model(config: DebertaLoraConfig) -> PreTrainedModel:
    # Load base DeBERTa model and apply LoRA adapters
    logger.info("Loading base model: %s with %d labels", config.model_name, config.num_labels)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels
    )

    target_modules = config.target_modules or ["query_proj", "value_proj"]
    logger.info(
        "Applying LoRA: r=%d, alpha=%d, dropout=%.2f, target_modules=%s",
        config.r, config.lora_alpha, config.lora_dropout, target_modules
    )

    lora_cfg = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        task_type=TaskType.SEQ_CLS
    )

    model = get_peft_model(base_model, lora_cfg)
    logger.info("LoRA adapters successfully applied to DeBERTa-v3-large.")
    return model


def main():
    parser = argparse.ArgumentParser(description="Initialize DeBERTa-v3-large with LoRA adapters")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--model_name", type=str, default=DebertaLoraConfig.model_name, help="Pretrained model name or path")
    parser.add_argument("--num_labels", type=int, default=4, help="Number of classification labels")
    args = parser.parse_args()

    config = DebertaLoraConfig(
        model_name=args.model_name,
        num_labels=args.num_labels,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout
    )

    model = get_deberta_lora_model(config)
    logger.info("Model ready with trainable parameters:")
    print(model.print_trainable_parameters())


if __name__ == "__main__":
    main()
