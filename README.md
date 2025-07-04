# AG News Text Classification

## Introduction

This project investigates the problem of **multi-class text classification** using the **AG News dataset**, a well-established benchmark dataset comprising English-language news articles categorized into four thematic classes: *World*, *Sports*, *Business*, and *Science/Technology*. The central objective is to design and evaluate a high-performance classification framework that leverages **state-of-the-art Transformer architectures**, incorporating strategies to address both the **limitations of input length** and the **efficiency of fine-tuning** in large-scale language models.

A fundamental limitation in Transformer-based models, such as BERT and its variants, lies in their constrained **maximum input sequence length** (typically 512 tokens), which poses significant challenges in classifying **long-form text** - a common characteristic in real-world documents. To circumvent this issue, the proposed architecture integrates a **Sliding Window mechanism** with **DeBERTa-v3**, enabling the model to process extended sequences through overlapping textual segments while maintaining global contextual coherence.

Concurrently, the use of **Longformer** - an architecture specifically engineered for extended attention spans (up to 4096 tokens) - facilitates direct encoding of long-range dependencies without segmentation. This dual-model approach enables robust contextual representation across both short and long textual inputs.

To optimize both computational efficiency and generalization capability, this work adopts **LoRA (Low-Rank Adaptation)**, a paradigm within **Parameter-Efficient Fine-Tuning (PEFT)**. By introducing trainable low-rank matrices into attention layers while freezing the bulk of pretrained parameters, LoRA significantly reduces the number of trainable parameters during fine-tuning, enabling **efficient adaptation on limited hardware resources** without compromising predictive performance.

Moreover, the framework integrates a **logit-level ensemble strategy**, aggregating the outputs of DeBERTa-v3 and Longformer via soft-voting. This ensemble approach seeks to synergize the localized precision of DeBERTa with the global modeling capacity of Longformer, resulting in a more robust and generalizable classifier.

In pursuit of greater transparency and accountability in model behavior, the project further incorporates **Error Analysis** and **Explainable AI (XAI)** methodologies. Post-hoc interpretability tools such as attention heatmaps and logit attribution are employed to analyze model predictions, diagnose failure cases, and guide iterative improvements through targeted data and architecture refinement.

In addition to the core architecture, further performance gains may be achieved through advanced extensions such as **domain-adaptive pretraining (DAPT)**, **confidence-based pseudo-labeling**, **stacking ensembles with meta-learners**, **k-fold cross-validation**, and **targeted data augmentation**. These strategies aim to **enhance robustness**, **reduce variance**, and **align model priors** more closely with the target domain.

**The pipeline encompasses the following components:**

- **Preprocessing**: Advanced tokenization, normalization, and window-based input segmentation.
- **Modeling**: Fine-tuning of `microsoft/deberta-v3-large` and `allenai/longformer-large-4096` with LoRA via the Hugging Face PEFT framework.
- **Ensembling**: Logit-level aggregation across models to enhance robustness and reduce variance.
- **Evaluation**: Comprehensive reporting of Accuracy, Precision, Recall, and F1-Score across all classes.
- **Analysis**: Qualitative and quantitative error investigation, along with model interpretability via XAI techniques.

By integrating recent advances in **transformer modeling**, **efficient fine-tuning**, and **model interpretability**, this project sets forth a replicable and scalable NLP pipeline. The framework not only surpasses classical baselines such as Naive Bayes and Support Vector Machines, but also provides a blueprint for future work in **long-form document** classification under constrained computational environments.

All components are developed using the Hugging Face `transformers`, `datasets`, `evaluate`, and `peft` libraries, ensuring **modularity**, **reproducibility**, and **applicability to a wide range of real-world classification tasks**.

## Dataset

The **AG News dataset**, introduced by **Xiang Zhang, Junbo Zhao, and Yann LeCun in 2015**, is a well-established benchmark corpus for topic classification in natural language processing (NLP). It was curated as part of the **ComeToMyHead** academic project and consists of news articles collected from over 2,000 news sources over a period exceeding one year.

The dataset is organized into four high-level topical categories:

- **World**
- **Sports**
- **Business**
- **Science/Technology**

Each instance comprises a concise **title** and **description** of a news article, together forming the input text for classification. This design supports both short-form and long-form input handling, making it particularly well-suited for evaluating models such as **DeBERTa-v3**, **Longformer**, and those fine-tuned via **LoRA (Low-Rank Adaptation)** for efficient long-sequence modeling.

The dataset is **balanced across categories** and comes with a predefined split:

- **Training set**: 120,000 samples (30,000 per class)
- **Test set**: 7,600 samples (1,900 per class)

AG News captures various real-world challenges in text classification, including:

- **Semantic ambiguity across topic boundaries**
- **Domain overlap and concept drift**
- **Stylistic variation and differences in textual length**

The dataset is publicly available via the [Hugging Face Datasets library](https://huggingface.co/datasets/ag_news), [TorchText loader](https://pytorch.org/text/stable/datasets.html#AG_NEWS), and the [original CSV source](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html), making it readily integrable with **PyTorch** and **TensorFlow** pipelines.

## Installation

This project requires Python 3.9+ and the following libraries:

- `torch` – for model training and inference with PyTorch
- `transformers` – for loading and fine‑tuning DeBERTa‑v3 and Longformer via Hugging Face
- `datasets` – for accessing and handling the AG News dataset efficiently
- `evaluate` – for computing evaluation metrics like accuracy, precision, recall, and F1
- `peft` – for Low‑Rank Adaptation (LoRA) fine‑tuning
- `numpy` – for numerical operations
- `pandas` – for data manipulation and I/O
- `scikit‑learn` – for classification reports, cross‑validation, and utility functions
- `tqdm` – for progress bars during training, evaluation, and inference
- `matplotlib` – for plotting (attention heatmaps, training curves)
- `shap` – for SHAP‑based explainability (force plots, summary plots)
- `joblib` – for serializing stacking models and fast I/O
- `gradio` – for launching an interactive web demo
- `jupyterlab` – for interactive notebook exploration

Install dependencies via pip:

```bash
pip install transformers datasets torch evaluate peft numpy pandas scikit‑learn tqdm matplotlib shap joblib gradio jupyterlab
```

Or install them all at once with:

```bash
pip install -r requirements.txt
```

Install dependencies via Conda:

```bash
conda env create -f environment.yml
conda activate agnews-classification
```

## Project Structure

The repository is organized as follows:

```plaintext
ag-news-text-classification/
├── README.md
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── configs/
│   ├── deberta_config.yaml
│   ├── longformer_config.yaml
│   ├── lora_config.yaml
│   ├── ensemble_config.yaml
│   └── pretrain_config.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── explore_dataset.ipynb
│   ├── train_visualization.ipynb
│   ├── error_analysis_xai.ipynb
│   ├── domain_adaptive_pretrain.ipynb
│   └── colab_demo.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── pretrain_lm.py
│   ├── pseudo_label.py
│   ├── train.py
│   ├── train_longformer.py
│   ├── evaluate.py
│   ├── ensemble.py
│   ├── stacking.py
│   ├── cv_run.py
│   ├── deberta_lora.py
│   ├── longformer_lora.py
│   ├── explainability.py
│   ├── inference.py
│   ├── predict_pipeline.py
│   └── utils.py
│
├── scripts/
│   ├── run_preprocess.sh
│   ├── run_pretrain_dapt.sh
│   ├── run_train_deberta.sh
│   ├── run_train_longformer.sh
│   ├── run_pseudo_label.sh
│   ├── run_ensemble.sh
│   ├── run_stack.sh
│   ├── run_cv.sh
│   └── run_full_pipeline.sh
│
├── outputs/
│   ├── checkpoints/
│   ├── dapt_checkpoints/
│   ├── pseudo_labels/
│   ├── logs/
│   ├── metrics/
│   └── figures/
│
└── ci/
    ├── lint.yml
    └── test.yml
```

## Usage



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

### Sample Results

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

