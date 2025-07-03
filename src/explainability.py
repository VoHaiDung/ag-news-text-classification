import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

from src.utils import configure_logger, prepare_dataloader

# Initialize logger
logger = configure_logger("results/logs/explainability.log")

# Plot attention heatmap for a single text
def plot_attention_heatmap(model, tokenizer, text: str, layer: int = -1, head: int = 0, save_path: str = None):
     # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v for k, v in inputs.items()}

    # Forward pass with attentions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # tuple[layer][batch, head, seq, seq]

    # Extract specified layer and head
    attn = attentions[layer][0, head].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f"Attention Heatmap - Layer {layer}, Head {head}")
    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Attention heatmap saved to {save_path}")
    else:
        plt.show()
    plt.close()

# Explain predictions with SHAP for a batch of texts
def shap_explain(model, tokenizer, texts: list, save_dir: str = None):
    # Try import shap
    try:
        import shap
    except ImportError:
        logger.error("SHAP library not installed. Install with 'pip install shap'.")
        return

    # Tokenize batch and prepare background
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    background = {k: v[:50] for k, v in inputs.items()}  # first 50 for background
    explainer = shap.DeepExplainer((model, model.get_input_embeddings()), background)

    # Compute SHAP values
    shap_values = explainer.shap_values(inputs)

    # Generate force plots
    for i, text in enumerate(texts):
        shap.force_plot(
            explainer.expected_value[0], shap_values[0][i], tokenizer.convert_ids_to_tokens(inputs['input_ids'][i].cpu().numpy()), matplotlib=True
        )
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt_path = os.path.join(save_dir, f"shap_force_{i}.png")
            plt.savefig(plt_path, bbox_inches='tight')
            logger.info(f"SHAP force plot saved to {plt_path}")
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Explain model predictions via attention heatmaps or SHAP")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--tokenizer_dir", type=str, help="Path to tokenizer (defaults to model_dir)")
    parser.add_argument("--text", type=str, help="Single input text for attention visualization")
    parser.add_argument("--texts_file", type=str, help="Path to newline-delimited texts for SHAP explanations")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index for attention heatmap")
    parser.add_argument("--head", type=int, default=0, help="Head index for attention heatmap")
    parser.add_argument("--save_dir", type=str, default="results/explainability", help="Directory to save plots")
    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir or args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    # Run attention heatmap if text provided
    if args.text:
        plot_attention_heatmap(
            model, tokenizer, args.text,
            layer=args.layer, head=args.head,
            save_path=os.path.join(args.save_dir, "attn_heatmap.png")
        )
      
    # Run SHAP if file of texts provided
    if args.texts_file:
        with open(args.texts_file) as f:
            texts = [line.strip() for line in f if line.strip()]
        shap_explain(model, tokenizer, texts, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
