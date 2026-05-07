# 1. Introduction and Background

> WBS task 9.1.1.

## 1.1 Motivation

News topic classification is a classical NLP benchmark and a standard
testbed for modern transformer pipelines. AG News (Zhang et al., 2015)
remains widely used because the four-class setting, balanced size and short
texts make it cheap to train and easy to interpret. The aim of this
capstone is to revisit the benchmark with the 2025 toolkit: state-of-the-art
transformers (DeBERTa-v3, ModernBERT), few-shot learning (SetFit),
explainability (SHAP, LIME), confidence calibration, and a multilingual
extension to Vietnamese.

## 1.2 Research questions

1. How much improvement can a modern transformer (DeBERTa-v3-small,
   ModernBERT-base) deliver over classical baselines (TF-IDF + LR/SVM,
   FastText) on AG News?
2. To what extent can SetFit recover the supervised performance with only a
   small number of labelled examples per class?
3. Does machine-translation-driven cross-lingual transfer (English ->
   Vietnamese) yield a usable Vietnamese classifier without native labels?
4. How well calibrated are the resulting models, and which examples are
   most often misclassified?
5. Can the best model be deployed as a public, latency-bounded demo?

## 1.3 Contributions

- An end-to-end pipeline (data, models, evaluation, deployment) reproducible
  from a single repository.
- A Vietnamese benchmark for AG News produced via OPUS-MT and back-
  translation augmentation.
- A direct comparison of DeBERTa-v3-small, ModernBERT-base and SetFit
  on the same train/test split.
- A SHAP-aware Gradio demo deployed on Hugging Face Spaces, with INT8
  ONNX quantization keeping inference under 50 ms per sample on CPU.

## 1.4 Document structure

Section 2 reviews the relevant literature. Section 3 describes the data and
exploratory analysis. Section 4 details the baselines. Section 5 covers
transformer fine-tuning. Section 6 discusses the multilingual extension.
Section 7 presents the few-shot results. Section 8 reports the evaluation
and explainability findings. Section 9 documents the deployment. Section 10
concludes.
