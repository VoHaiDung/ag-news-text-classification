---
language: [en, vi, fr]
license: mit
tags:
  - text-classification
  - news
  - ag-news
  - deberta-v3
  - modernbert
  - mdeberta-v3
  - xlm-roberta
  - setfit
  - r-drop
  - ensembling
  - calibration
  - explainable-ai
datasets:
  - ag_news
metrics:
  - accuracy
  - f1
  - ece
pipeline_tag: text-classification
---

# AG News Topic Classifier (SIC AI Capstone)

This artefact bundles twelve fine-tuned transformer encoders that
classify short news articles (in **English**, **Vietnamese**, or
**French**) into one of four topics: **World**, **Sports**,
**Business**, **Sci/Tech**. It was produced as the final artefact of
the Samsung Innovation Campus (SIC) AI Course capstone project.

## Intended use

- Educational demonstration of modern NLP techniques: transformer
  fine-tuning, R-Drop regularisation, multi-seed soft-voting
  ensembling, few-shot SetFit learning, OPUS-MT machine translation
  with back-translation augmentation, post-hoc confidence
  calibration, and SHAP / LIME explainability.
- Reproducible reference for the AG News benchmark across three
  languages.
- Twelve-row latency-accuracy Pareto front on dynamic INT8 ONNX,
  useful for choosing a single-checkpoint deployment-tier model.

This artefact is **not** intended for production-grade content
moderation, fact-checking, or any high-stakes decision making. AG
News articles were scraped between 2004 and 2005; vocabulary and
topic conventions may not generalise to contemporary news.

## Twelve-model lineup

| Language | Encoder family | Variants |
|----------|----------------|----------|
| English | DeBERTa-v3, ModernBERT | small (44 M), base (149 M), base (184 M), large (395 M) |
| Vietnamese | mDeBERTa-v3-base, XLM-RoBERTa-large | encoder × {target-only, target + back-translation} (2 × 2 grid) |
| French | mDeBERTa-v3-base, XLM-RoBERTa-large | encoder × {target-only, target + back-translation} (2 × 2 grid) |

## Training data

- **English (primary)**: AG News (Zhang, Zhao and LeCun, 2015) —
  120,000 train / 7,600 test news articles in four classes, retrieved
  via the Hugging Face `datasets` library. Stratified 90/10 split of
  the training set produces 108,000 train / 12,000 validation
  examples; the test split is touched only for final reporting.
- **Vietnamese (Phase 5A)**: machine-translation of AG News produced
  with `Helsinki-NLP/opus-mt-en-vi`, with back-translation
  augmentation `vi → en → vi` applied to the training split only.
- **French (Phase 5B)**: machine-translation of AG News produced with
  `Helsinki-NLP/opus-mt-en-fr`, with back-translation augmentation
  `fr → en → fr` applied to the training split only.

## Training procedure

- **Backbones (English)**: `microsoft/deberta-v3-small`,
  `microsoft/deberta-v3-base`, `answerdotai/ModernBERT-base`,
  `answerdotai/ModernBERT-large`.
- **Backbones (Vietnamese, French)**: `microsoft/mdeberta-v3-base`
  and `FacebookAI/xlm-roberta-large` for each target language.
- **Optimiser**: AdamW (epsilon = 1e-8, weight decay = 0.01).
- **Schedule**: linear warm-up on 10 % of the steps followed by a
  cosine schedule.
- **Label smoothing**: 0.1 on the primary configuration (base / large
  tier).
- **Early stopping**: on validation F1-macro with patience 2.
- **Sequence length**: 256 tokens (covers > 99 % of AG News inputs).
- **Variance-reduction ablation** (post-hoc, ModernBERT-large only):
  a 2 × 3 grid pairing {vanilla, +R-Drop} with seeds {13, 42, 73};
  R-Drop is implemented as a custom `RDropTrainer(Trainer)` subclass
  that adds the symmetric KL divergence between two stochastic
  forward passes to the cross-entropy loss. The six checkpoints are
  combined by soft-voting accuracy ensembling to produce the official
  O1 figure.

## Evaluation (test set, n = 7,600)

| Model | Lang | Test Accuracy | Test F1-macro | ECE (chained T → isotonic) |
|---|:---:|---:|---:|---:|
| TF-IDF + Logistic Regression | en | 0.9166 | 0.9164 | — |
| TF-IDF + Linear SVM | en | 0.9255 | 0.9254 | — |
| FastText | en | 0.9171 | 0.9170 | — |
| DeBERTa-v3-small (3 epoch, vanilla) | en | 0.9463 | 0.9463 | — |
| ModernBERT-base (3 epoch, vanilla) | en | 0.9471 | 0.9471 | — |
| DeBERTa-v3-base (5 epoch, LS 0.1) | en | 0.9493 | 0.9493 | — |
| ModernBERT-large (5 epoch, LS 0.1) | en | 0.9499 | 0.9498 | 0.0285 |
| **ModernBERT-large + R-Drop (3-seed ensemble)** | en | **0.9528** | **0.9528** | — |
| SetFit (K = 64, mean of 3 seeds) | en | 0.8759 | 0.8755 | — |
| mDeBERTa-v3 (vi, vi-only) | vi | 0.8867 | 0.8863 | — |
| mDeBERTa-v3 (vi, +back-translation) | vi | 0.8960 | 0.8960 | — |
| XLM-R-large (vi, vi-only) | vi | 0.9011 | 0.9011 | — |
| **XLM-R-large (vi, +back-translation)** | vi | **0.9041** | **0.9041** | — |
| mDeBERTa-v3 (fr, fr-only) | fr | 0.9361 | 0.9361 | — |
| mDeBERTa-v3 (fr, +back-translation) | fr | 0.9395 | 0.9395 | — |
| **XLM-R-large (fr, fr-only)** | fr | **0.9466** | **0.9466** | — |
| XLM-R-large (fr, +back-translation) | fr | 0.9451 | 0.9451 | — |

ECE was measured only on the primary English checkpoint (ModernBERT-
large seed 42, label-smoothed); the chained T → isotonic post-hoc
calibrator clears the O4 target (≤ 0.03). Full reliability diagrams
live under `outputs/evaluation/calibrated_seed13_v2/`.

## Latency

The full twelve-model latency-accuracy Pareto front (single-stream
CPU, batch = 1, INT8 ONNX) is reported in Section 3.5.3 of the Final
Report. **ModernBERT-base INT8** is the Pareto-optimal
single-checkpoint model (147 MB INT8 footprint, 289.83 ms / sample
on the benchmarked AMX-INT8 Intel Xeon Platinum 8558 host). No model
clears the 50 ms target at batch 1 on commodity x86 CPUs at this
batch size; the gap is reported as hardware-bounded.

## Limitations and biases

- **Benchmark age**: AG News articles were scraped between 2004 and
  2005; vocabulary and topic conventions reflect that era and may
  not generalise to contemporary news feeds.
- **Translation quality**: Vietnamese and French corpora are
  100 % machine-generated and have not been audited against
  human-validated references via BLEU, chrF, or COMET. Section 4.1.4
  of the Final Report flags this as a scope-out.
- **Class boundaries**: Cleanlab flags 2,852 training examples
  (2.6 %) as potentially mis-labelled, most of them on the
  Business ↔ Sci/Tech boundary (fintech, IPOs, tech-industry
  market dynamics).
- **Statistical power**: the variance-reduction ablation uses three
  seeds; the F-test and paired t-test for the variance and mean
  improvements are underpowered at N = 3 (see Section 3.3.2.5).
- **Latency**: O5 (≤ 50 ms CPU at batch 1) is not met by any of the
  twelve INT8 models; the bound is hardware-related, not
  model-related, on the benchmarked host.
- **Responsible-release scope-out**: no Model Card per dimension, no
  Dataset Card per language, no carbon report, no fairness audit
  beyond per-class — see Section 4.1.4 of the Final Report.

## Citation

```bibtex
@inproceedings{zhang2015character,
  title     = {Character-level Convolutional Networks for Text Classification},
  author    = {Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2015}
}

@misc{vohaidung2025agnewscapstone,
  title  = {AG News Text Classification: A Multilingual, Multi-Architecture Capstone Study},
  author = {Vo, Hai Dung},
  year   = {2025},
  note   = {Samsung Innovation Campus AI Course capstone project},
  url    = {https://github.com/VoHaiDung/ag-news-text-classification}
}
```
