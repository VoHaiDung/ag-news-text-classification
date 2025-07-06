import os
import torch
import numpy as np
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

# Paths to fine-tuned models and stacking classifier
DEBERTA_DIR    = "results/deberta_lora"
LONGFORMER_DIR = "results/longformer_lora"
STACKING_MODEL = "outputs/checkpoints/stacking_model.joblib"

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load DeBERTa-LoRA model and tokenizer
tokenizer_deb = AutoTokenizer.from_pretrained(DEBERTA_DIR, use_fast=True)
model_deb     = AutoModelForSequenceClassification.from_pretrained(DEBERTA_DIR).to(device).eval()

# Load Longformer-LoRA model and tokenizer
tokenizer_lon = AutoTokenizer.from_pretrained(LONGFORMER_DIR, use_fast=True)
model_lon     = AutoModelForSequenceClassification.from_pretrained(LONGFORMER_DIR).to(device).eval()

# Load stacking classifier if available
use_stacking = os.path.exists(STACKING_MODEL)
stacking_clf = joblib.load(STACKING_MODEL) if use_stacking else None

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# Perform sliding-window inference on long text segments
def sliding_window_logits(model, tokenizer, text, max_len, stride):
    enc = tokenizer(
        text,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_len,
        stride=stride,
        padding='max_length',
        return_tensors='pt'
    )
    logits_chunks = []
    for i in range(enc['input_ids'].size(0)):
        input_ids = enc['input_ids'][i].unsqueeze(0).to(device)
        mask      = enc['attention_mask'][i].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=mask)
        logits_chunks.append(out.logits.cpu().numpy())
    # Average logits across all windows
    return np.vstack(logits_chunks).mean(axis=0, keepdims=True)

# Main classification function for Gradio UI
def classify(text, weight_deberta, weight_longformer):
    # Get logits from DeBERTa and Longformer
    deb_logits = sliding_window_logits(model_deb, tokenizer_deb, text, max_len=512, stride=256)
    lon_logits = sliding_window_logits(model_lon, tokenizer_lon, text, max_len=4096, stride=1024)

    # Combine via ensemble or stacking
    if use_stacking and stacking_clf:
        features = np.concatenate([deb_logits, lon_logits], axis=1)
        probs = stacking_clf.predict_proba(features)[0]
    else:
        total = weight_deberta + weight_longformer
        wdeb = weight_deberta / total
        wlon = weight_longformer / total
        combined = wdeb * deb_logits + wlon * lon_logits
        probs = torch.softmax(torch.from_numpy(combined), dim=-1).numpy()[0]

    # Prepare output: probabilities and predicted label
    label = CLASS_NAMES[int(probs.argmax())]
    prob_dict = {cls: float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}
    return prob_dict, label

# Build Gradio interface
demo = gr.Interface(
    fn=classify,
    inputs=[
        gr.Textbox(lines=4, label="Input Text", placeholder="Enter title + description..."),
        gr.Slider(0, 1, value=0.5, step=0.05, label="Weight: DeBERTa"),
        gr.Slider(0, 1, value=0.5, step=0.05, label="Weight: Longformer"),
    ],
    outputs=[
        gr.Label(num_top_classes=4, label="Probabilities"),
        gr.Textbox(label="Predicted Label")
    ],
    title="AG News Classification Demo",
    description="Ensemble DeBERTa‑LoRA & Longformer‑LoRA (or Stacking) for AG News multi-class text classification."
)

if __name__ == "__main__":
    demo.launch()
