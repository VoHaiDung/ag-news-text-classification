# AG News Text Classification

## Introduction

This project addresses the task of **multi-class text classification** on the widely used **AG News** dataset, which consists of news articles labeled into four distinct categories: **World**, **Sports**, **Business**, and **Science/Technology**. The objective is to construct a **highly accurate and scalable classification pipeline** that leverages **state-of-the-art Transformer-based language models**, with a particular focus on **Microsoft’s DeBERTa-v3** and **AllenAI’s Longformer** architectures.

A central challenge in real-world document classification lies in the processing of **long-form textual inputs** that exceed the maximum token length allowed by standard Transformer models (typically 512 tokens). To overcome this limitation, the pipeline integrates a **Sliding Window** strategy for DeBERTa-v3, which systematically segments input documents into overlapping windows, ensuring full coverage of long documents without loss of contextual information. In parallel, the use of **Longformer**, with its extended context window (up to 4096 tokens), enables direct modeling of longer sequences without fragmentation.

To further improve classification performance and model robustness, this project incorporates an **ensemble mechanism** that combines the output logits of both DeBERTa-v3 and Longformer. This **soft-voting ensemble approach** balances the strengths of both architectures: DeBERTa’s powerful contextual representations in shorter segments and Longformer’s capacity for modeling global context in longer texts.

The pipeline includes:

- **Data preprocessing**: tokenization, normalization, and sliding-window segmentation.
- **Modeling**: fine-tuning of `microsoft/deberta-v3-large` and `allenai/longformer-large-4096` using Hugging Face Transformers.
- **Evaluation**: accuracy, precision, recall, and F1-score across all categories, with full classification reports.
- **Inference**: ensemble prediction using averaged logits across multiple models.

This implementation significantly outperforms traditional classification approaches (e.g., Naive Bayes, SVM, Logistic Regression), highlighting the advantages of **transfer learning**, **multi-model ensembling**, and **context-aware tokenization strategies** in modern NLP pipelines.

All components are built with **modular**, **extensible code** using Hugging Face’s `transformers`, `datasets`, `evaluate`, and `accelerate` libraries, ensuring that the project can be easily adapted to other classification tasks or datasets involving long-form text.

## Dataset

The AG News dataset is a widely used benchmark for topic classification in natural language processing. It consists of news articles collected from over 2,000 news sources by ComeToMyHead during more than one year of activity. The dataset is organized into four topic categories:

- **World**
- **Sports**
- **Business**
- **Science/Technology**

Each sample includes a short title and description of a news article, serving as input text for classification. The dataset is balanced across classes and is pre-split into:

- **Training set**: 120,000 samples (30,000 per class)
- **Test set**: 7,600 samples (1,900 per class)

This dataset presents several challenges common to real-world NLP tasks, including class ambiguity, domain overlap, and varied writing styles. It is available via the [Hugging Face Datasets library](https://huggingface.co/datasets/ag_news), the [TorchText loader](https://pytorch.org/text/stable/datasets.html#AG_NEWS), and the [original CSV source](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html), and supports immediate use with PyTorch and TensorFlow pipelines.

## Installation

This project requires Python 3.8+ and the following libraries:

- `transformers` – for loading and fine-tuning the DeBERTa-v3 model using Hugging Face
- `datasets` – for accessing and handling the AG News dataset efficiently
- `torch` – for model training and inference with PyTorch 
- `evaluate` – for computing evaluation metrics like accuracy and F1
- `scikit-learn` – for classification reports and utility functions
- `tqdm` –  for progress bars during training and evaluation
- `scipy` – for compatibility with evaluation functions
- `tensorboard` – for monitoring training logs visually
- `huggingface_hub` – for sharing, syncing, or downloading models from Hugging Face
- `accelerate` – for managing hardware configuration (CPU/GPU) and efficient training

Install dependencies via pip:

```bash
pip install transformers datasets torch evaluate scikit-learn tqdm scipy tensorboard huggingface_hub accelerate
```

Or install them all at once with:

```bash
pip install -r requirements.txt
```

## Project Structure

The repository is organized as follows:

```plaintext
ag-news-text-classification/
│
├── data/                      # (Optional) Custom dataset storage or logs
├── models/                    # Saved models after training (e.g. final_model/)
├── results/                   # Output directory for logs and checkpoints
│   └── logs/                  # Training logs (TensorBoard, metrics, etc.)
├── train.py                   # Main training script (BERT + Transformers)
├── predict.py                 # (Optional) Script for inference on new text
├── requirements.txt           # Python dependencies for easy setup
├── README.md                  # Project documentation
├── LICENSE                    # License information
└── .gitignore                 # Git ignore rules (cache, outputs, environments, etc.)
```

## Evaluation Metrics

To assess the model’s performance on the AG News classification task, we evaluate it using standard classification metrics:

- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual positives that were correctly identified.
- **F1-Score**: Harmonic mean of precision and recall.

### Evaluation Code

Use the following code snippet to compute metrics during training or evaluation:

```python
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=label_names, digits=4))
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
        'f1': f1_score(labels, preds, average='weighted')
    }
```

**Note**: Replace label_names with your actual class labels, for example:

```python
label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
```

### Sample Results (BERT base uncased fine-tuned on AG News)

| Class        | Precision | Recall | F1-Score   |
| ------------ | --------- | ------ | ---------- |
| World        | 0.94      | 0.93   | 0.94       |
| Sports       | 0.97      | 0.96   | 0.96       |
| Business     | 0.93      | 0.93   | 0.93       |
| Sci/Tech     | 0.93      | 0.94   | 0.93       |
| **Macro**    | 0.94      | 0.94   | 0.94       |
| **Weighted** | 0.94      | 0.94   | 0.94       |
| **Accuracy** |           |        | **0.9402** |

These scores indicate that the BERT-based model performs consistently well across all four categories in AG News.

