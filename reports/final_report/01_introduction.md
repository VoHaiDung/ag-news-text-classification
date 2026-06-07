# 1. Introduction

## 1.1 Background Information

News topic classification is one of the canonical text classification
benchmarks in modern Natural Language Processing. The AG News corpus,
introduced by Zhang, Zhao, and LeCun<sup>[1]</sup> as part of the
*Character-level Convolutional Networks for Text Classification* paper,
is a four-class balanced subset of the AG news search portal. It contains 120,000 training
articles and 7,600 test articles, evenly distributed across the four
categories *World*, *Sports*, *Business*, and *Sci/Tech*.

Although the original benchmark dates back ten years, the AG News dataset
remains widely used as a fast, well-understood proxy for text-classification
research because (i) it is small enough to iterate quickly on a single GPU,
(ii) the four-class setup makes errors visually interpretable through
confusion matrices, and (iii) the short headline-plus-abstract format mimics
many real-world content moderation and recommendation pipelines.

In recent years, the dominant approach to text classification has shifted
from bag-of-words pipelines toward fine-tuning large pre-trained language
models such as BERT, DeBERTa, and most recently ModernBERT. Beyond raw
accuracy, modern practice also requires confidence calibration, model
explainability, and cost-effective deployment. This project re-examines
AG News with the *2025 NLP toolkit* and reports performance, calibration,
explainability, multilingual transfer, and deployment latency in a single
unified pipeline.

## 1.2 Motivation and Objective

### 1.2.1 Motivation

A typical undergraduate capstone on AG News reports a single fine-tuned
BERT score and stops there. The motivation of this project is to go
substantially further and demonstrate the *engineering breadth* expected of
a junior NLP practitioner in 2025:

- Compare classical baselines (TF-IDF + Logistic Regression / Linear SVM,
  FastText) against modern transformers from two scale tiers - small/base
  (DeBERTa-v3-small, ModernBERT-base) and base/large (DeBERTa-v3-base,
  ModernBERT-large) - under a single reproducible protocol.
- Close the residual ~0.02 pp gap between the best single-seed
  transformer and the 0.95 F1 target through a *variance-reduction
  ablation*: a 2 × 3 grid (variant ∈ {vanilla, +R-Drop} × seed ∈ {13,
  42, 73}) on ModernBERT-large, followed by 3-seed soft-voting
  ensembling. R-Drop<sup>[9]</sup> is implemented as a custom
  HuggingFace ``Trainer`` subclass.
- Quantify *data efficiency* via SetFit<sup>[12]</sup> at 8, 16, 32, and
  64 samples per class.
- Audit label quality with Cleanlab<sup>[16]</sup>.
- Provide *explainability* through SHAP<sup>[18]</sup> and
  LIME<sup>[19]</sup>.
- Measure *calibration* through the Expected Calibration
  Error<sup>[21]</sup> and close any gap to the target through a
  four-calibrator post-hoc grid (baseline / temperature scaling /
  isotonic regression / chained), following Guo et al.<sup>[22]</sup>
  and Zadrozny and Elkan<sup>[23]</sup>.
- Extend to **Vietnamese** and **French** through OPUS-MT<sup>[14]</sup>
  and back-translation augmentation<sup>[15]</sup>, then fine-tune
  mDeBERTa-v3 and XLM-R-large on each translated corpus to form a
  language × encoder × back-translation ablation grid.
- Deploy all twelve trained transformers behind an interactive Gradio
  demo with automatic English / Vietnamese / French language routing.
  Every model is also exported to dynamic INT8 ONNX, and the resulting
  twelve-row latency-accuracy Pareto front identifies the
  deployment-tier model (lowest single-stream CPU latency, lowest INT8
  footprint, within 0.3 pp of the best F1).

### 1.2.2 Objectives

| ID | Objective                                                | Target           | Outcome (Section)                                              | Status      |
|----|----------------------------------------------------------|------------------|----------------------------------------------------------------|:-----------:|
| O1 | F1-macro on AG News test set (best model)                | ≥ 0.95           | **0.9528** - R-Drop 3-seed ensemble (Section 3.3.2.5)          | **met**     |
| O2 | F1-macro of SetFit at 64 samples per class               | ≥ 0.93           | 0.8755 - few-shot ceiling on balanced 4-class corpus (3.3.4)   | **not met** |
| O3 | F1-macro on Vietnamese AG News (best multilingual model) | ≥ 0.90           | **0.9041** - XLM-R-large + back-translation (Section 3.3.3)    | **met**     |
| O4 | Expected Calibration Error of the best model             | ≤ 0.03           | **0.0285** - temperature → isotonic chained (Section 3.5.2)    | **met**     |
| O5 | ONNX INT8 inference latency (CPU, batch size 1)          | ≤ 50 ms / sample | 289.83 ms - hardware-bounded on x86 CPUs at batch 1 (3.5.3)    | **not met** |
| O6 | Public GitHub repository accessible end-to-end           | URL 200 OK       | [VoHaiDung/ag-news-text-classification](https://github.com/VoHaiDung/ag-news-text-classification) | **met**     |
| O7 | Final report submitted using the SIC ``.docx`` template  | One document     | ``SIC_AI_Capstone Project_Final Report_FILLED.docx``           | **met**     |
| O8 | F1-macro on French AG News (best multilingual model)     | ≥ 0.90           | **0.9466** - XLM-R-large (fr-only) (Section 3.3.3)             | **met**     |

## 1.3 Members and Role Assignments

| Name           | Role                       | Responsibilities                                                        |
|----------------|----------------------------|-------------------------------------------------------------------------|
| Vo Hai Dung    | Team Leader               | Project management, literature review, environment and tracking setup, EDA, baselines, transformer fine-tuning, multilingual extension, few-shot learning, evaluation, explainability, deployment, report and presentation. |

The team is named **Aimer PAM**. Every deliverable from environment
setup to final video is owned by the team leader.

## 1.4 Schedule and Milestones

The project follows a 76-day plan organised into nine phases (Sun 1 Jun
2025 to Fri 15 Aug 2025). Each phase has a runnable entry point under
``scripts/`` and produces a measurable deliverable.

| Phase | Window         | Title                                         | Key deliverable                                                                                                                |
|-------|----------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| 1     | 1 - 7 Jun      | Project Kickoff and Research                  | Repository, environment, action plan                                                                                           |
| 2     | 8 - 14 Jun     | Exploratory Data Analysis                     | EDA notebook, Cleanlab audit, BERTopic                                                                                         |
| 3     | 15 - 21 Jun    | Baseline Models                               | TF-IDF + LR / SVM, FastText comparison                                                                                         |
| 4     | 22 Jun - 8 Jul | Transformer Fine-tuning                       | Four-model scale ablation + variance-reduction grid (2 × 3 cells: vanilla × R-Drop × {seed 13, 42, 73}) + 3-seed ensembling     |
| 5A    | 9 - 15 Jul     | Multilingual Extension - Vietnamese           | Vietnamese AG News, 2 × 2 ablation (encoder × back-translation)                                                                |
| 5B    | 16 - 22 Jul    | Multilingual Extension - French               | French AG News, 2 × 2 ablation (encoder × back-translation)                                                                    |
| 6     | 23 - 25 Jul    | Few-shot Learning with SetFit                 | SetFit checkpoints at K ∈ {8, 16, 32, 64}, learning curve                                                                      |
| 7     | 26 Jul - 5 Aug | Evaluation and Explainability                 | Per-model metrics, four-calibrator post-hoc ECE grid, SHAP, LIME, error analysis, sliding-window long-document strategy        |
| 8     | 6 - 11 Aug     | Demo and Deployment                           | Twelve-model INT8 ONNX export, latency-accuracy Pareto front, multi-model Gradio app (local), public GitHub repository         |
| 9     | 12 - 15 Aug    | Report and Presentation                       | Final report, slides, demo video script, logo design, ~36-second demo video                                                    |

The full daily breakdown (approximately 103 atomic tasks across four
hierarchy levels) lives in
``SIC_AI_Capstone_Project_Work_Breakdown_Structure.xlsx``.
