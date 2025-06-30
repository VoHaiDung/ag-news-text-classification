import logging
from dataclasses import dataclass
from typing import Optional

from transformers import AutoModelForSequenceClassification, PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LongformerLoraConfig:
    model_name: str = "allenai/longformer-large-4096"
    num_labels: int = 4
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: Optional[list] = None


def get_longformer_lora_model(config: Optional[LongformerLoraConfig] = None) -> PreTrainedModel:
    """
    Load the Longformer-large model and apply LoRA for parameter-efficient fine-tuning.

    Args:
        config (LongformerLoraConfig, optional): Configuration for model and LoRA settings.

    Returns:
        PreTrainedModel: A Longformer model wrapped with LoRA adapters for sequence classification.
    """
    cfg = config or LongformerLoraConfig()

    logger.info("Loading base model: %s with %d labels", cfg.model_name, cfg.num_labels)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels
    )

    target_modules = cfg.target_modules or ["query", "value"]

    logger.info("Applying LoRA: r=%d, alpha=%d, dropout=%.2f, target_modules=%s",
                cfg.r, cfg.lora_alpha, cfg.lora_dropout, target_modules)

    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        task_type=TaskType.SEQ_CLS
    )

    model = get_peft_model(base_model, lora_cfg)
    logger.info("LoRA adapters successfully applied to Longformer.")
    return model


if __name__ == "__main__":
    model = get_longformer_lora_model()
    print(model.print_trainable_parameters())
