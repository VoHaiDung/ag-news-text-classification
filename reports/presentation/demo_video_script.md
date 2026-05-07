# Demo Video Script (3-5 minutes)

> WBS task 9.3.

## Storyboard

| Time      | Scene                                          | What is shown                                                     | Voice-over                                                                                                                       |
|-----------|------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| 00:00-00:15 | Title card                                    | Project name, author, instructor                                  | "AG News Text Classification - Samsung Innovation Campus AI capstone."                                                          |
| 00:15-00:45 | Pipeline overview                             | Animated 9-phase diagram                                           | Walk through the phases at a high level: data, baselines, transformers, multilingual, few-shot, evaluation, deployment, report. |
| 00:45-01:30 | Live Gradio demo - English                    | Browser screen capture                                             | Paste a Reuters-style headline, click Classify, show predicted class and probability bars.                                      |
| 01:30-02:15 | SHAP explanation                              | Switch to the Explain tab, screenshot of token heat-map            | Highlight which tokens drove the decision; explain the colour scale.                                                            |
| 02:15-02:45 | Vietnamese demo                               | Repeat with a Vietnamese sample                                    | Note the cross-lingual capability and the underlying mDeBERTa-v3.                                                               |
| 02:45-03:30 | Headline numbers                              | Slide with comparison table                                        | F1-macro per model, ECE, latency. Stress that ONNX INT8 keeps inference under 50 ms on CPU.                                     |
| 03:30-04:00 | Few-shot story                                | Learning curve chart                                               | "With 64 samples per class, SetFit recovers 93% of the supervised F1."                                                          |
| 04:00-04:45 | Lessons learned                               | Three bullet points                                                | Calibration matters; back-translation helps; CPU deployment is feasible with quantization.                                       |
| 04:45-05:00 | Closing                                       | Repository URL, Hugging Face Space URL                            | "Code, model and demo are public; thank you."                                                                                   |

## Recording checklist

- Use 1920 x 1080 resolution at 30 fps; encode to H.264 + AAC.
- Record narration separately and align in the editor.
- Capture the Gradio app at full browser width; hide bookmark bar and tab list.
- Run the screen recorder on the deployed Hugging Face Space, not the local
  development server, to confirm public availability.
