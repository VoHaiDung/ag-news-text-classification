# Risk Register

WBS task 1.3.2: identify risks that could derail the 76-day plan, score them
on likelihood and impact, and document the mitigation strategy. The register
is reviewed at the start of each phase and updated when new risks emerge.

| ID  | Risk                                                          | Phase  | Likelihood | Impact | Mitigation                                                                               |
|-----|---------------------------------------------------------------|--------|------------|--------|------------------------------------------------------------------------------------------|
| R01 | Free-tier GPU quota exhausted before tuning is complete       | 4, 5   | Medium     | High   | Run hyper-parameter sweeps on a 10% subset; switch to ``deberta-v3-small`` if needed.    |
| R02 | OPUS-MT translations contain systematic errors for headlines  | 5      | Medium     | Medium | Sample 100 translations for human review; report BLEU vs. a held-out reference set.     |
| R03 | Cleanlab flags too many candidates to review manually         | 2      | Low        | Medium | Cap manual review at 200 highest-confidence suspects; document the threshold.            |
| R04 | DeBERTa-v3 tokenizer requires ``sentencepiece`` not on the    | 4      | Low        | Low    | Pin ``sentencepiece`` in ``requirements.txt`` and verify in Phase 1 diagnostics.        |
|     | base image                                                    |        |            |        |                                                                                          |
| R05 | Hugging Face Spaces CPU is too slow for the SHAP demo         | 8      | Medium     | Medium | Pre-compute SHAP for a fixed sample bank; allow on-demand inference only for short text. |
| R06 | ONNX export numerical drift exceeds 1e-3 vs. PyTorch          | 8      | Low        | Medium | Validate logits on a 500-example calibration set; fall back to FP32 ONNX if needed.     |
| R07 | Weights and Biases free quota exceeded                        | All    | Low        | Low    | Mirror the most important metrics to local CSV under ``outputs/metrics/``.               |
| R08 | Reviewer requests scope changes after the freeze              | 9      | Medium     | High   | Reserve 2 buffer days at the end of each phase; log scope decisions in the report.       |

## Review log

| Date       | Reviewer        | Notes                                                                |
|------------|-----------------|----------------------------------------------------------------------|
| 2025-06-07 | Vo Hai Dung     | Initial register, approved by Ths. Nguyen Trung Hau.                 |
