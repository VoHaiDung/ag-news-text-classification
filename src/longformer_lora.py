from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType


def get_longformer_lora_model():
    """
    Load the Longformer-large model and apply LoRA for parameter-efficient fine-tuning.

    Returns:
        A Longformer model wrapped with LoRA adapters for sequence classification.
    """
    # Load pretrained Longformer with 4 output classes
    model = AutoModelForSequenceClassification.from_pretrained(
        "allenai/longformer-large-4096",
        num_labels=4
    )

    # Configure LoRA: rank, alpha, dropout, and target attention modules
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    # Apply LoRA to the base Longformer model
    lora_model = get_peft_model(model, config)
    return lora_model
