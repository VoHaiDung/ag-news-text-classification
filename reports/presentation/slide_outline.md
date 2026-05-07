# Presentation Slide Outline

> WBS tasks 9.2.1 and 9.2.2. Slides target a 12-minute defence with a
> 3-minute embedded demo video.

| # | Title                                                                | Key visual                                  | Speaking notes                                                                                |
|---|----------------------------------------------------------------------|---------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Title slide                                                          | Project logo, author, instructor            | One sentence on the goal: modern NLP toolkit on AG News.                                       |
| 2 | Motivation                                                           | Problem statement and research questions    | Why AG News, why this scope, what is new.                                                       |
| 3 | Pipeline overview                                                    | Phase diagram (9 boxes)                     | One line per phase; show the linear dependency chain.                                           |
| 4 | Dataset and EDA                                                      | Class distribution and length histogram     | Mention Cleanlab audit findings and BERTopic topic count.                                       |
| 5 | Baselines: TF-IDF and FastText                                       | Comparison bar chart                        | Frame as a sanity check and a reference point for the transformer gain.                         |
| 6 | Transformer fine-tuning                                              | Validation F1 over epochs                   | Compare DeBERTa-v3 and ModernBERT, highlight Optuna sweep gain.                                 |
| 7 | Vietnamese extension                                                 | EN versus VI confusion matrices             | OPUS-MT pipeline, back-translation augmentation, cross-lingual transfer.                        |
| 8 | Few-shot SetFit                                                      | Learning curve (samples per class)          | Show the F1 at 8, 16, 32, 64; emphasise data efficiency.                                        |
| 9 | Evaluation: calibration and explainability                           | Reliability diagram + SHAP example          | Report ECE before and after quantization; pick one striking SHAP output.                        |
| 10 | Latency and deployment                                              | Latency table + Gradio screenshot           | Highlight the under-50 ms target on CPU and the public Hugging Face Space URL.                  |
| 11 | Conclusions                                                         | Three bullets                               | Performance, data efficiency, deployment.                                                       |
| 12 | Q&A                                                                 | Contact information                         | Backup slides for additional metrics, hyper-parameters and risk register.                       |
