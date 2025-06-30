import logging
from dataclasses import dataclass
from typing import Optional

from transformers import AutoModelForSequenceClassification, PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DebertaLoraConfig:
    model_name: str = "microsoft/deberta-v3-large"
    num_labels: int = 4
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    target_modules: Optional[list] = None


def get_deberta_lora_model(config: Optional[DebertaLoraConfig] = None) -> PreTrainedModel:
    cfg = config or DebertaLoraConfig()

    logger.info("Loading base model: %s with %d labels", cfg.model_name, cfg.num_labels)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels
    )

    target_modules = cfg.target_modules or ["query_proj", "value_proj"]

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
    logger.info("LoRA adapters successfully applied.")
    return model


if __name__ == "__main__":
    model = get_deberta_lora_model()
    print(model.print_trainable_parameters())
