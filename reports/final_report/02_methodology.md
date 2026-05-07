# 2. Methodology and Implementation

> WBS task 9.1.2.

## 2.1 Pipeline overview

The project is split into nine phases that are executed sequentially. Each
phase has a runnable entry point under ``scripts/`` and writes its
artefacts to a dedicated directory under ``outputs/``. The dependency
graph between phases is linear: every phase consumes outputs from earlier
phases and produces inputs for later ones.

| Phase | Title                              | Entry point                       | Outputs                                  |
|-------|------------------------------------|-----------------------------------|------------------------------------------|
| 1     | Kickoff and Research               | ``scripts/phase1_kickoff.py``     | environment diagnostics, scope document  |
| 2     | Exploratory Data Analysis          | ``scripts/phase2_eda.py``         | EDA tables, plots, Cleanlab and BERTopic |
| 3     | Baseline Models                    | ``scripts/phase3_baselines.py``   | TF-IDF + LR/SVM, FastText, comparison    |
| 4     | Transformer Fine-tuning            | ``scripts/phase4_transformers.py``| DeBERTa-v3, ModernBERT, Optuna trials    |
| 5     | Multilingual Extension             | ``scripts/phase5_multilingual.py``| Vietnamese AG News, mDeBERTa-v3 model    |
| 6     | Few-shot Learning                  | ``scripts/phase6_setfit.py``      | SetFit checkpoints, learning curve       |
| 7     | Evaluation and XAI                 | ``scripts/phase7_evaluation.py``  | metrics, calibration, SHAP, LIME, errors |
| 8     | Demo and Deployment                | ``scripts/phase8_deployment.py``  | ONNX, INT8 model, Gradio app, HF Space   |
| 9     | Report and Presentation            | ``scripts/phase9_report.py``      | report sections, slide outline           |

## 2.2 Configuration management

Every run is driven by a YAML file under ``configs/`` that is parsed into
the typed :class:`ExperimentConfig` defined in ``src/configs/schema.py``.
Reproducibility is anchored by ``src.utils.repro.set_global_seed``, which
seeds the Python, NumPy and PyTorch RNGs and enables cuDNN's deterministic
mode.

## 2.3 Data preparation

The English AG News dataset is loaded with ``datasets.load_dataset`` and
split into train / validation / test in a 90 / 10 ratio (stratified by
class) on top of the original train and test splits. Whitespace is
collapsed and the integer labels keep the order ``World, Sports, Business,
Sci/Tech``. A Cleanlab audit (Northcutt et al., 2021) is run on the
training split using a TF-IDF + Logistic Regression probe to flag
candidate noisy labels.

## 2.4 Modelling

Three model families are compared:

- *Classical*: TF-IDF (1-2 grams) followed by logistic regression or a
  calibrated linear SVM, and FastText (Joulin et al., 2017) trained on
  word-and-character n-grams.
- *Transformers*: DeBERTa-v3-small (He et al., 2023) and ModernBERT-base
  (Warner et al., 2024) fine-tuned via the Hugging Face ``Trainer`` with
  early stopping on the validation F1-macro.
- *Few-shot*: SetFit (Tunstall et al., 2022) trained on 8, 16, 32 and 64
  samples per class with three random seeds, yielding a learning curve.

## 2.5 Multilingual extension

WBS phase 5 produces a Vietnamese copy of AG News with the OPUS-MT
``Helsinki-NLP/opus-mt-en-vi`` model and augments the training set with
back-translation (``vi-en`` followed by ``en-vi``). mDeBERTa-v3-base is
fine-tuned twice: once on the translated data, and once on the union of the
translated and back-translated data.

## 2.6 Evaluation

For every model we report accuracy, F1-macro, the confusion matrix, the
expected calibration error (Naeini et al., 2015), and inference latency on
CPU. SHAP and LIME explanations are produced on the most confidently
mis-classified examples. Error analysis bins the predictions by class pair
and by text length.

## 2.7 Deployment

The best transformer is exported to ONNX with opset 17 via
``optimum.onnxruntime``. INT8 dynamic quantization (AVX-512 VNNI) is
applied to keep the model under 200 MB and the per-sample CPU latency
under 50 ms. The Gradio app exposes a *Classify* and an *Explain* tab and
is mirrored on Hugging Face Spaces.
