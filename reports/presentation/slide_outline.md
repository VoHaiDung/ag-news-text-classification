# Presentation Slide Outline

> WBS tasks 9.2.1 and 9.2.2. Slides target a 12-minute defence with a
> ~36-second embedded demo video.

| # | Title                                                                | Key visual                                  | Speaking notes                                                                                |
|---|----------------------------------------------------------------------|---------------------------------------------|------------------------------------------------------------------------------------------------|
| 1 | Title slide                                                          | Project logo, author, instructor            | One sentence on the goal: modern NLP toolkit on AG News.                                       |
| 2 | Motivation                                                           | Problem statement and research questions    | Why AG News, why this scope, what is new.                                                       |
| 3 | Pipeline overview                                                    | Phase diagram (9 boxes)                     | One line per phase; show the linear dependency chain.                                           |
| 4 | Dataset and EDA                                                      | Class distribution and length histogram     | Mention Cleanlab audit findings and BERTopic topic count.                                       |
| 5 | Baselines: TF-IDF and FastText                                       | Comparison bar chart                        | Frame as a sanity check and a reference point for the transformer gain.                         |
| 6 | Transformer fine-tuning + R-Drop ablation                            | 2x2 ablation grid + 3-seed ensemble F1      | Scale ablation (small/base/large), then variance-reduction grid; R-Drop 3-seed ensemble F1 = 0.9528 closes the 0.95 gap. |
| 7 | Multilingual extension (Vietnamese + French)                         | Tri-lingual 2x2 grids                       | OPUS-MT pipeline, back-translation augmentation, XLM-R + BT on VI (0.9041), XLM-R on FR (0.9466).|
| 8 | Few-shot SetFit                                                      | Learning curve (samples per class)          | Show the F1 at 8, 16, 32, 64; emphasise data efficiency.                                        |
| 9 | Evaluation: calibration and explainability                           | 4-calibrator ECE comparison + SHAP example  | T→isotonic chained calibrator drops ECE from 0.0502 to 0.0285 (meets O4); one striking SHAP output. |
| 10 | Latency and deployment                                              | 12-model Pareto front + Gradio screenshot   | Twelve-model latency table; ModernBERT-base is Pareto-optimal deployment tier; O5 hardware-bounded discussion. |
| 11 | Conclusions                                                         | Three bullets                               | Performance, data efficiency, deployment.                                                       |
| 12 | Q&A                                                                 | Contact information                         | Backup slides for additional metrics, hyper-parameters and risk register.                       |
