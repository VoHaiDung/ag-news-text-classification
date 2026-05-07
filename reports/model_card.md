---
language: [en, vi]
license: mit
tags:
  - text-classification
  - news
  - ag-news
  - deberta-v3
  - modernbert
  - setfit
  - explainable-ai
datasets:
  - ag_news
metrics:
  - accuracy
  - f1
pipeline_tag: text-classification
---

# AG News Topic Classifier (SIC AI Capstone)

This model classifies short English (and Vietnamese) news articles into one
of four topics: **World**, **Sports**, **Business**, **Sci/Tech**. It was
produced as the final artefact of the Samsung Innovation Campus (SIC) AI
Course capstone project.

## Intended use

- Educational demonstration of modern NLP techniques (transformers, few-shot
  learning, calibration, explainability).
- Reproducible reference for the AG News benchmark.

This model is **not** intended for production-grade content moderation,
fact-checking, or any high-stakes decision making.

## Training data

- **Primary**: AG News (Zhang et al., 2015) - 120,000 train / 7,600 test
  examples in four classes, retrieved via the Hugging Face ``datasets``
  library.
- **Vietnamese extension**: machine translation of AG News produced with
  ``Helsinki-NLP/opus-mt-en-vi`` and back-translation augmentation
  (VI -> EN -> VI).

## Training procedure

- Backbone: ``microsoft/deberta-v3-small`` for English,
  ``microsoft/mdeberta-v3-base`` for Vietnamese.
- Optimiser: AdamW (epsilon = 1e-8).
- Schedule: linear warm-up (10%) followed by linear decay.
- Hyper-parameters selected via Optuna sweeps; see
  ``configs/sweeps/optuna_deberta.yaml``.

## Evaluation

| Model                         | Accuracy | F1-macro | ECE   | Latency (CPU, ms/sample) |
|-------------------------------|----------|----------|-------|--------------------------|
| TF-IDF + Logistic Regression  |          |          |       |                          |
| TF-IDF + Linear SVM           |          |          |       |                          |
| FastText                      |          |          |       |                          |
| DeBERTa-v3-small (fine-tune)  |          |          |       |                          |
| ModernBERT-base (fine-tune)   |          |          |       |                          |
| SetFit (64 samples per class) |          |          |       |                          |
| mDeBERTa-v3 (Vietnamese)      |          |          |       |                          |

Numbers are filled in by Phase 7 of the pipeline once training completes.

## Limitations and biases

- AG News articles were scraped between 2004 and 2005; vocabulary and topic
  conventions reflect that era and may not generalise to contemporary news
  feeds.
- Vietnamese performance depends on machine-translation quality; manual
  spot-checks are documented in ``reports/risk_register.md``.

## Citation

```
@inproceedings{zhang2015character,
  title     = {Character-level Convolutional Networks for Text Classification},
  author    = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2015}
}
```
