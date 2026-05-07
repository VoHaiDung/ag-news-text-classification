# 3. Results, Discussion and Conclusion

> WBS task 9.1.3.

## 3.1 Headline results

| Model                              | Accuracy | F1-macro | ECE   | Latency (CPU, ms/sample) |
|------------------------------------|---------:|---------:|------:|-------------------------:|
| TF-IDF + Logistic Regression       |          |          |       |                          |
| TF-IDF + Linear SVM                |          |          |       |                          |
| FastText                           |          |          |       |                          |
| DeBERTa-v3-small (fine-tune)       |          |          |       |                          |
| ModernBERT-base (fine-tune)        |          |          |       |                          |
| SetFit (64 samples per class)      |          |          |       |                          |
| mDeBERTa-v3 (Vietnamese, vi only)  |          |          |       |                          |
| mDeBERTa-v3 (Vietnamese, vi + BT)  |          |          |       |                          |

Numbers are populated automatically by ``scripts/phase9_report.py``
once Phase 7 metrics are available.

## 3.2 Findings

- *Transformer vs. classical baselines.* Discuss the absolute gain in
  F1-macro and the cost in CPU latency.
- *Data efficiency.* Comment on the SetFit learning curve; in particular,
  the F1-macro reached at 8, 16, 32 and 64 samples per class.
- *Multilingual transfer.* Discuss the impact of back-translation
  augmentation on the Vietnamese model and the cross-lingual benchmark.
- *Calibration.* Compare ECE before and after deployment quantization.

## 3.3 Limitations

- AG News is a short-text, four-class dataset with limited vocabulary
  diversity. Results may not transfer to long-form articles or to
  fine-grained taxonomies.
- Vietnamese performance is bounded by the quality of the OPUS-MT
  translations; native Vietnamese news data is not used.
- INT8 dynamic quantization can degrade calibration; this is monitored in
  Phase 7 but the trade-off is application-specific.

## 3.4 Conclusion

Summarise the contributions:

1. Reproducible pipeline from raw data to deployed demo.
2. Comparative study of classical, transformer, and few-shot approaches.
3. Vietnamese benchmark for AG News.
4. Calibrated, low-latency CPU deployment.

## 3.5 Future work

- Fine-tune on a larger multilingual corpus (XNLI-like) to confirm
  cross-lingual transfer beyond machine-translated data.
- Replace OPUS-MT with a stronger translation model (e.g., NLLB) and
  measure the downstream improvement.
- Explore prompt-based classification with instruction-tuned small LLMs
  and compare against the SetFit baseline.
