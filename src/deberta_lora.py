from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType


def get_deberta_lora_model():
    """
    Load the DeBERTa-v3-large model and apply LoRA for parameter-efficient fine-tuning.

    Returns:
        A DeBERTa model wrapped with LoRA adapters for sequence classification.
    """
    # Load pretrained DeBERTa-v3 model with 4 output classes
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large",
        num_labels=4
    )

    # Configure LoRA: rank, alpha, dropout, and target attention modules
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_proj", "value_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    # Apply LoRA to the base model
    lora_model = get_peft_model(model, config)
    return lora_model
