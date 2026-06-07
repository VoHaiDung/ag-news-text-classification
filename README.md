<div align="center">

<img src="assets/logo-ag-news-text-classification.png" alt="AG News Text Classification logo" width="200" />

# AG News Text Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-D7FF64?style=flat-square&logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)
[![Type Checked: MyPy](https://img.shields.io/badge/type%20checked-mypy-2A6DB2?style=flat-square)](https://mypy-lang.org/)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)](https://docs.pytest.org/)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.41%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![Datasets](https://img.shields.io/badge/Datasets-2.18%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/datasets)
[![Accelerate](https://img.shields.io/badge/Accelerate-0.30%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/accelerate)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

[![SetFit](https://img.shields.io/badge/SetFit-1.0%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://github.com/huggingface/setfit)
[![FastText](https://img.shields.io/badge/FastText-0.9%2B-0467DF?style=flat-square&logo=meta&logoColor=white)](https://fasttext.cc/)
[![Cleanlab](https://img.shields.io/badge/Cleanlab-2.6%2B-7E57C2?style=flat-square)](https://cleanlab.ai/)
[![BERTopic](https://img.shields.io/badge/BERTopic-0.16%2B-2C3E50?style=flat-square)](https://maartengr.github.io/BERTopic/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45%2B-1F77B4?style=flat-square)](https://shap.readthedocs.io/)
[![LIME](https://img.shields.io/badge/LIME-0.2%2B-2ECC71?style=flat-square)](https://github.com/marcotcr/lime)

[![ONNX](https://img.shields.io/badge/ONNX-1.16%2B-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnx.ai/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.18%2B-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Optimum](https://img.shields.io/badge/Optimum-1.20%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/optimum)
[![Gradio](https://img.shields.io/badge/Gradio-4.31%2B-F97316?style=flat-square&logo=gradio&logoColor=white)](https://www.gradio.app/)

[![Weights & Biases](https://img.shields.io/badge/W%26B-0.17%2B-FFBE00?style=flat-square&logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
[![MLflow](https://img.shields.io/badge/MLflow-2.13%2B-0194E2?style=flat-square&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-3.6%2B-1F77B4?style=flat-square)](https://optuna.org/)
[![Demo](https://img.shields.io/badge/Demo-Local%20Gradio-F97316?style=flat-square&logo=gradio&logoColor=white)](src/deployment/gradio_app.py)

</div>

End-to-end NLP pipeline for the AG News topic classification benchmark, developed as
the Samsung Innovation Campus (SIC) AI Course capstone project. The project re-examines
a classic four-class news topic dataset using the modern NLP toolkit: state-of-the-art
transformers, R-Drop variance-reduction ablation, multi-seed soft-voting ensembling,
post-hoc confidence calibration, few-shot learning, label-noise auditing,
explainability, multilingual extension to Vietnamese and French, and a multi-model
local Gradio demo with automatic three-way language routing.

## 1. Headline results

| Objective | Metric | Target | Achieved | Status |
|-----------|--------|--------|---------:|:------:|
| O1 | F1-macro on AG News test set (best English model) | ≥ 0.95 | **0.9528** (R-Drop 3-seed soft-voting ensemble of ModernBERT-large) | met |
| O2 | F1-macro of SetFit at 64 samples per class | ≥ 0.93 | 0.8755 (mean of 3 seeds) | not met |
| O3 | F1-macro on Vietnamese AG News (best multilingual model) | ≥ 0.90 | **0.9041** (XLM-R-large + back-translation) | met |
| O4 | Expected Calibration Error of the best model | ≤ 0.03 | **0.0285** (temperature scaling → isotonic regression, chained) | met |
| O5 | ONNX INT8 inference latency (CPU, batch size 1) | ≤ 50 ms / sample | 290 ms (best of 12 models) | not met, hardware-bounded |
| O6 | Public GitHub repository accessible end-to-end | URL responds 200 OK | [VoHaiDung/ag-news-text-classification](https://github.com/VoHaiDung/ag-news-text-classification) | met |
| O7 | Final report submitted using the SIC template | One ``.docx`` file | One file under ``reports/`` | met |
| O8 | F1-macro on French AG News (best multilingual model) | ≥ 0.90 | **0.9466** (XLM-R-large) | met |

Six of eight numbered objectives are met. The two unmet objectives are documented
honestly in the final report rather than hidden: O2 (SetFit ceiling on balanced
four-class corpora, Section 3.3.4) and O5 (batch-1 transformer inference is
sequential-bound on commodity x86 CPUs, Section 3.5.3).

## Demo video

A ~36-second product walkthrough of the multi-model Gradio app — English,
Vietnamese and French classification, automatic model routing, the
sliding-window long-document path, the full softmax breakdown and the SHAP
explanation — is available at
[`SIC_AI_Capstone_Project_Demo_Video.mp4`](SIC_AI_Capstone_Project_Demo_Video.mp4).
The shot-by-shot script and the static frames behind it live under
[`reports/presentation/`](reports/presentation/)
(`demo_video_script.md`, `demo_inputs.txt`, `demo_frames/`).

## 2. The twelve-model lineup

Twelve supervised transformer checkpoints span four English encoders, four
Vietnamese encoders and four French encoders (the multilingual languages each
form a 2 x 2 grid of encoder x back-translation augmentation). Three classical
baselines and a SetFit learning curve sit alongside the transformer matrix.

| Language | Encoder family | Variants | Best test F1-macro |
|----------|----------------|----------|-------------------:|
| English | DeBERTa-v3 / ModernBERT | small (44 M) / base (149 M) / base (184 M) / large (395 M) | 0.9528 (R-Drop 3-seed ensemble) |
| Vietnamese | mDeBERTa-v3 / XLM-R-large | encoder x back-translation 2 x 2 | 0.9041 (XLM-R-large + BT) |
| French | mDeBERTa-v3 / XLM-R-large | encoder x back-translation 2 x 2 | 0.9466 (XLM-R-large) |

All twelve transformers are exported to dynamic INT8 ONNX (Optimum, opset 17)
and benchmarked end-to-end on the same single-stream CPU host to produce the
latency-accuracy Pareto front used to identify the deployment-tier model
(ModernBERT-base INT8, 147 MB, 289.83 ms per sample, F1 = 0.9471).

## 3. Project structure

```text
ag-news-text-classification/
├── assets/                  Brand assets rendered by the Gradio UI
│   ├── logo-ag-news-text-classification.png    Project logo (centered above the title)
│   └── logo-aimer-pam.png                       Team logo (top-right attribution badge)
├── configs/                 YAML configuration files (data, model, training, sweeps)
├── data/                    Datasets (git-ignored, managed by HuggingFace Hub)
├── notebooks/               Jupyter notebooks for exploration
├── outputs/                 Trained checkpoints, metrics, figures, INT8 models (git-ignored)
├── reports/                 Final report (Markdown + SIC docx), slides, demo video script
├── scripts/                 Phase-level entry points mapping 1:1 to the WBS
├── src/                     Reusable library code
│   ├── configs/             Dataclass-based configuration schema
│   ├── data/                Loaders, EDA, augmentation, translation
│   ├── deployment/          ONNX export, quantization, multi-model Gradio app
│   ├── evaluation/          Metrics, calibration, latency, error analysis
│   ├── explainability/      SHAP and LIME explainers
│   ├── inference/           Long-document classifier (sliding-window aggregation)
│   ├── models/              Baselines and transformer wrappers
│   ├── training/            Trainers (including the R-Drop subclass), sweep utilities
│   └── utils/               Logging, reproducibility, IO, tracking
├── tests/                   Pytest unit tests
├── pyproject.toml           Build system + tool configuration
├── requirements.txt         Pinned runtime dependencies
└── README.md                This file
```

## 4. Phase to script mapping

The [Work Breakdown Structure](SIC_AI_Capstone_Project_Work_Breakdown_Structure.xlsx)
defines nine sequential phases. Each phase has a corresponding entry point in
``scripts/``; the post-hoc ablation, calibration, ensembling and latency studies
that produced the headline numbers above are factored into dedicated helper
scripts that are invoked from Phase 7 and Phase 8.

| Phase | Title                                          | Primary script                    |
|-------|------------------------------------------------|-----------------------------------|
| 1     | Project Kickoff and Research                   | ``scripts/phase1_kickoff.py``     |
| 2     | Exploratory Data Analysis                      | ``scripts/phase2_eda.py``         |
| 3     | Baseline Models                                | ``scripts/phase3_baselines.py``   |
| 4     | Transformer Fine-tuning                        | ``scripts/phase4_transformers.py``|
| 5     | Multilingual Extension (Vietnamese and French) | ``scripts/phase5_multilingual.py``|
| 6     | Few-shot Learning with SetFit                  | ``scripts/phase6_setfit.py``      |
| 7     | Evaluation and Explainability                  | ``scripts/phase7_evaluation.py``  |
| 8     | Demo and Deployment                            | ``scripts/phase8_deployment.py``  |
| 9     | Report and Presentation                        | ``scripts/phase9_report.py``      |

### Post-hoc helper scripts

| Script | Purpose | Cited in |
|--------|---------|----------|
| ``scripts/ensemble_inference.py`` | Soft-voting ensemble across N HuggingFace checkpoints; produces the R-Drop 3-seed ensemble that meets O1. | Section 3.3.2.5 |
| ``scripts/calibrate_model.py`` | Four-calibrator ECE grid (baseline / temperature scaling / isotonic regression / temperature → isotonic chained); produces the calibrator that meets O4. | Section 3.5.2 |
| ``scripts/export_all_int8.py`` | Batch ONNX FP32 export + dynamic INT8 quantization of the entire twelve-model lineup. | Section 3.5.3 |
| ``scripts/benchmark_all_latency.py`` | Single-stream and multi-thread CPU latency benchmark across the twelve INT8 models; produces the Pareto-front table cited in Section 3.5.3. | Section 3.5.3 |
| ``scripts/local_inference.py`` | Offline CPU inference helper for any downloaded checkpoint (regular PyTorch path or ONNX INT8 path); useful for reviewers without a GPU. | - |
| ``scripts/sync_from_remote.py`` | Filtered tar-over-SSH sync that pulls metric files / figures / INT8 artefacts from a rented cloud GPU host to the local machine. | - |

## 5. Quick start

```bash
# 1. Clone and create a virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
.venv\Scripts\activate        # Windows PowerShell

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure secrets (optional, only needed if WANDB / HF tracking is enabled)
cp .env.example .env
# edit .env to add WANDB_API_KEY and HF_TOKEN

# 4. Run a phase end-to-end (example: Phase 2 EDA)
python -m scripts.phase2_eda --config configs/data/ag_news.yaml
```

### Launch the multi-model Gradio demo

```bash
python -m src.deployment.gradio_app
```

The demo loads every fine-tuned encoder behind a single dropdown picker, with
automatic English / Vietnamese / French detection and a sliding-window path
for inputs that exceed the 512-token native context of DeBERTa-style models.

## 6. Reproducibility

All random seeds are controlled centrally by ``src.utils.repro.set_global_seed``.
Configurations are versioned through YAML files under ``configs/``. Experiment
artefacts (metrics, model checkpoints, plots) are logged to Weights and Biases
and mirrored locally under ``outputs/``. The R-Drop ablation grid is fully
declarative: each of the five cells of the 2 x 2 grid (variant x seed) is
materialised as its own ``configs/models/modernbert_large_seed{13,42,73}{_rdrop}.yaml``
file so the grid can be re-run by anyone with no code changes.

## 7. License

Code is released under the MIT license. The AG News dataset is the property of
its original authors (Zhang et al., 2015).
