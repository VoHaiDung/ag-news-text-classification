# AG News Text Classification

<div align="center">

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
[![Hugging Face Spaces](https://img.shields.io/badge/Demo-HF%20Spaces-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/spaces)

</div>

End-to-end NLP pipeline for the AG News topic classification benchmark, developed as
the Samsung Innovation Campus (SIC) AI Course capstone project. The project re-examines
a classic four-class news topic dataset using the modern NLP toolkit: state-of-the-art
transformers, few-shot learning, label-noise auditing, explainability, confidence
calibration, multilingual extension to Vietnamese and a public Gradio demo.

## 1. Project structure

```text
ag-news-text-classification/
├── configs/                 YAML configuration files (data, model, training)
├── data/                    Datasets (git-ignored, managed by DVC / HF Hub)
├── notebooks/               Jupyter notebooks for exploration
├── reports/                 Final report, slides, demo video
├── scripts/                 Phase-level entry points mapping 1:1 to the WBS
├── src/                     Reusable library code
│   ├── configs/             Dataclass-based configuration schema
│   ├── data/                Loaders, EDA, augmentation, translation
│   ├── deployment/          ONNX export, quantization, Gradio app
│   ├── evaluation/          Metrics, calibration, latency, error analysis
│   ├── explainability/      SHAP and LIME explainers
│   ├── models/              Baselines and transformer wrappers
│   ├── training/            Trainers, sweep utilities
│   └── utils/                Logging, reproducibility, IO, tracking
├── tests/                   Pytest unit tests
├── pyproject.toml           Build system + tool configuration
├── requirements.txt         Pinned runtime dependencies
└── README.md                This file
```

## 2. Phase to script mapping

The [Work Breakdown Structure](SIC_AI_Capstone_Project_Work_Breakdown_Structure.xlsx)
defines nine sequential phases. Each phase has a corresponding entry point in
`scripts/`:

| Phase | Title                              | Script                          |
|-------|------------------------------------|---------------------------------|
| 1     | Project Kickoff and Research       | `scripts/phase1_kickoff.py`     |
| 2     | Exploratory Data Analysis          | `scripts/phase2_eda.py`         |
| 3     | Baseline Models                    | `scripts/phase3_baselines.py`   |
| 4     | Transformer Fine-tuning            | `scripts/phase4_transformers.py`|
| 5     | Multilingual Extension             | `scripts/phase5_multilingual.py`|
| 6     | Few-shot Learning with SetFit      | `scripts/phase6_setfit.py`      |
| 7     | Evaluation and Explainability      | `scripts/phase7_evaluation.py`  |
| 8     | Demo and Deployment                | `scripts/phase8_deployment.py`  |
| 9     | Report and Presentation            | `scripts/phase9_report.py`      |

## 3. Quick start

```bash
# 1. Clone and create a virtual environment
python -m venv .venv
source .venv/bin/activate     # Linux / macOS
.venv\Scripts\activate        # Windows PowerShell

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure secrets
cp .env.example .env
# edit .env to add WANDB_API_KEY and HF_TOKEN

# 4. Run a phase end-to-end
python -m scripts.phase2_eda --config configs/data/ag_news.yaml
```

## 4. Reproducibility

All random seeds are controlled centrally by `src.utils.repro.set_global_seed`.
Configurations are versioned through YAML files under `configs/`. Experiment
artefacts (metrics, model checkpoints, plots) are logged to Weights and Biases
and mirrored locally under `outputs/`.

## 5. License

Code is released under the MIT license. The AG News dataset is the property of
its original authors (Zhang et al., 2015).
