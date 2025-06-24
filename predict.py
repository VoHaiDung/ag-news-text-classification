import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./results/final_model"  # Change if you saved elsewhere
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# AG News labels
labels = ["World", "Sports", "Business", "Sci/Tech"]

def classify(text):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return labels[predicted_class]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your input text here\"")
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    prediction = classify(input_text)
    print(f"Predicted category: {prediction}")
