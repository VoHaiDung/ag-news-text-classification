# Scope, Objectives and Success Metrics

WBS task 1.3.1. The aim of this document is to make the project's scope
explicit and unambiguous, so that we can decide objectively whether each
deliverable is complete.

## 1. Problem statement

Build an end-to-end news article classifier on the AG News corpus that goes
beyond a single fine-tuned BERT score. The system is evaluated on
classification accuracy *and* on robustness, calibration, explainability,
data efficiency and inference latency.

## 2. In scope

1. Classical baselines (TF-IDF + Logistic Regression / SVM, FastText).
2. Transformer fine-tuning of DeBERTa-v3-small and ModernBERT-base.
3. Few-shot training with SetFit at 8/16/32/64 samples per class.
4. Vietnamese extension via OPUS-MT translation and back-translation
   augmentation, fine-tuning mDeBERTa-v3 / XLM-R.
5. Evaluation: accuracy, F1-macro, confusion matrix, expected calibration
   error (ECE), inference latency, SHAP and LIME explanations, error
   analysis by class and text length.
6. Deployment: Gradio demo with SHAP highlighting, ONNX + INT8 export, public
   Hugging Face Space.
7. Final report (SIC template), slide deck and 3-5 minute demo video.

## 3. Out of scope

- Joint training on additional news corpora (Reuters, BBC News).
- Languages other than English and Vietnamese.
- Production-grade serving (Triton, AWS Inferentia). The HF Space is a demo,
  not a production endpoint.

## 4. Success metrics

| ID | Metric                                       | Target              | Phase |
|----|----------------------------------------------|---------------------|-------|
| S1 | F1-macro on AG News test set (best model)    | >= 0.95             | 4, 7  |
| S2 | F1-macro of SetFit at 64 samples per class   | >= 0.93             | 6, 7  |
| S3 | F1-macro on Vietnamese AG News (mDeBERTa)    | >= 0.90             | 5, 7  |
| S4 | Expected calibration error (best model)      | <= 0.03             | 7     |
| S5 | ONNX INT8 inference latency (CPU, batch 1)   | <= 50 ms / sample   | 8     |
| S6 | Public Hugging Face Space accessible         | URL responds 200 OK | 8     |
| S7 | Final report submitted using SIC template    | One ``.docx`` file  | 9     |

## 5. Definition of done

A phase is considered complete when (a) every WBS task in the phase is
checked off in the workbook, (b) the corresponding output artefacts exist
under ``outputs/`` or ``reports/``, and (c) the metrics defined above are
reported (or explicitly waived with justification) in the phase summary
file.
