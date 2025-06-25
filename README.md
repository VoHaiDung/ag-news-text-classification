# AG News Text Classification

## Introduction

This project addresses the task of multi-class text classification using the AG News dataset, which comprises news articles labeled across four categories: World, Sports, Business, and Science/Technology. The objective is to develop and evaluate machine learning and deep learning models capable of accurately classifying news content based on textual features.

The project follows a standard NLP pipeline, including:
- **Data preprocessing**: tokenization, normalization.
- **Feature representation**: TF-IDF, word embeddings.
- **Supervised learning**: model training, validation, and testing.

Multiple modeling approaches are explored and compared, ranging from classical algorithms such as Naive Bayes and Logistic Regression, to deep learning techniques like Convolutional Neural Networks (CNN) and fine-tuned Transformer-based models (BERT).

A key focus of the project is on the Transformer pipeline, leveraging Hugging Face Transformers and pretrained BERT models through a modular and extensible architecture. This allows easy experimentation with transfer learning, fine-tuning strategies, and inference performance.

Performance is assessed using standard metrics including accuracy, precision, recall, and F1-score, with emphasis on model generalization and practical deployment.

All implementation is conducted using widely adopted libraries such as scikit-learn, TensorFlow/Keras, and Hugging Face Transformers, with reproducible experiments and modular code structure.

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

This project requires Python 3.7+ and the following libraries:

- `transformers` – for loading and fine-tuning pre-trained BERT models  
- `datasets` – for easy access to the AG News dataset and others  
- `torch` – for training and inference using PyTorch  
- `scikit-learn` – for evaluation metrics and basic utilities  
- `evaluate` – for streamlined integration of metrics like accuracy
- `tqdm` – for visual progress bars during training
- `scipy` – required for some advanced loss/metrics functions in Hugging Face Trainer
- `tensorboard` – for training log visualization
- `huggingface_hub` – for downloading, sharing, and syncing models with the Hugging Face Hub

Install dependencies via pip:

```bash
pip install transformers datasets torch scikit-learn evaluate tqdm scipy tensorboard huggingface_hub
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
├── .gitignore                 # Git ignore rules (cache, outputs, environments, etc.)
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

