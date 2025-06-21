# AG News Text Classification

## Project Overview

This project addresses the task of multi-class text classification using the AG News dataset, which comprises news articles labeled across four categories: World, Sports, Business, and Science/Technology. The objective is to develop and evaluate machine learning and deep learning models capable of accurately classifying news content based on textual features.

The project follows a standard NLP pipeline, including data preprocessing (tokenization, normalization), feature representation (TF-IDF, word embeddings), and supervised model training. Multiple approaches are explored and compared, ranging from classical algorithms such as Naive Bayes and Logistic Regression to advanced architectures like Convolutional Neural Networks (CNN) and fine-tuned Transformer-based models (BERT).

Performance is assessed using standard metrics including accuracy, precision, recall, and F1-score, with emphasis on model generalization and practical deployment. The project serves as both a technical exercise and a foundational case study in text classification, providing insight into the challenges and design considerations of real-world NLP applications.

All implementation is conducted using widely adopted libraries such as scikit-learn, TensorFlow/Keras, and Hugging Face Transformers, with reproducible experiments and modular code structure. This makes the project an ideal entry-level portfolio item for showcasing skills in applied natural language processing.

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

Install dependencies via pip:

```bash
pip install transformers datasets torch scikit-learn evaluate
