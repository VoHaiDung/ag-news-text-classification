# 4. Projected Impact

## 4.1 Accomplishments and Benefits

### 4.1.1 Technical accomplishments

By the end of the nine-phase plan the project delivers a single repository
that demonstrates the full breadth of modern NLP engineering practice:

- **End-to-end reproducibility**. From raw AG News download to a
  multi-model local Gradio launch, every step is reproducible by
  running the matching ``scripts/phase{N}.py`` script. No piece of
  state is implicit; every output lives under ``outputs/`` and every
  input under ``configs/``.
- **A modern model-comparison study** on AG News spanning **twelve
  trained transformers across four families**, in addition to three
  classical baselines (best: TF-IDF + Linear SVM at 92.54 % F1-macro).
  English: four scale-ablation rows (small / base / base / large)
  followed by a 2 × 2 ablation grid that pairs three random seeds
  with the R-Drop regulariser; the resulting **R-Drop 3-seed
  soft-voting ensemble reaches 95.28 % test F1-macro, meeting
  objective O1** by +0.28 pp. Vietnamese: four multilingual fine-tunes
  forming a 2 × 2 grid (encoder × back-translation; best: XLM-R-large
  with back-translation at 90.41 % - meets O3). French: four
  multilingual fine-tunes on the same 2 × 2 grid (best:
  XLM-R-large at 94.66 % - meets O8). A SetFit data-efficiency
  learning curve recovers 92.2 % of supervised performance with only
  0.24 % of the training set (256 labelled examples).
- **Quality and trustworthiness checks**. Cleanlab-driven label-noise
  audit, Expected Calibration Error, reliability diagram, and detailed
  error analysis, including the most confidently mis-classified examples,
  are all produced automatically.
- **Explainability by default**. SHAP TextExplainer and LIME are run on
  the deployed model and exposed in the user interface, not just in the
  appendix of the report.
- **Vietnamese benchmark**. A machine-translated Vietnamese AG News
  corpus and back-translation augmentation are versioned with the code,
  enabling future work on Vietnamese topic classification.
- **Production-flavoured deployment**. All twelve fine-tuned encoders
  are exported to ONNX with opset 17 and dynamically INT8-quantised
  through Optimum, producing a complete latency-accuracy Pareto front
  on a single host (Section 3.5.3). The Pareto-optimal single-
  checkpoint model is **ModernBERT-base INT8** (147 MB, 94.71 %
  F1-macro, 289.83 ms single-stream CPU latency on a modern server-
  grade x86_64 host with AVX-512 VNNI and AMX_INT8 support); it has
  the lowest latency, the smallest footprint and an F1 within 0.6 pp
  of the R-Drop ensemble (Section 3.3.2.5). The local Gradio
  application exposes every trained checkpoint behind a single
  dropdown picker with automatic English/Vietnamese/French routing
  and a sliding-window long-document path, so reviewers can
  interrogate the full model matrix without retraining anything.

### 4.1.2 Educational benefits

Beyond the deliverables, the project is designed to maximise *learning*
value:

- Hands-on use of every layer of the modern NLP stack:
  ``transformers``, ``datasets``, ``accelerate``, ``setfit``,
  ``sentence-transformers``, ``cleanlab``, ``bertopic``, ``shap``,
  ``lime``, ``optuna``, ``onnxruntime``, ``optimum``, and ``gradio``.
- Practical exposure to MLOps fundamentals: dataclass-driven
  configuration, central seed management, structured logging, experiment
  tracking via Weights and Biases or MLflow, and reproducible artefact
  storage.
- Clear academic documentation discipline: each module cites the paper
  it implements, every phase script lists the WBS task IDs it covers, and
  the final report is generated from versioned Markdown sources.

### 4.1.3 Wider benefits

- The codebase under ``src/`` is generic enough to be re-targeted at any
  topic-classification dataset by editing only the YAML files in
  ``configs/data/``.
- The Vietnamese corpus released alongside the project provides a
  reproducible reference for future Vietnamese NLP research.
- The full project codebase, including the multi-model Gradio
  application and every phase script, lives at
  [VoHaiDung/ag-news-text-classification](https://github.com/VoHaiDung/ag-news-text-classification) so that
  potential employers and reviewers can clone the repository,
  reproduce every table in this report from scratch, and interact
  with all twelve trained encoders through a single local launch.

### 4.1.4 Responsible-release scope-out

For transparency the following items, which would be expected of a
public release into a research community rather than of an
institutional capstone deliverable, are listed as **out of scope for
this report** rather than completed work:

- **Licence inheritance**. The released Vietnamese and French
  derivatives inherit constraints from both AG News
  (CC-BY-SA 3.0 / 4.0 via Antonio Gulli's archive) and from the
  OPUS-MT translation models (CC-BY-4.0); the share-alike obligation
  of AG News applies transitively to the translated corpora.
  Downstream users of the derivative corpora should re-release under
  a compatible share-alike licence.
- **No Model Card and no Dataset Card.** The artefact does not ship
  Model Cards<sup>[39]</sup> for the twelve trained
  encoders, nor Dataset Cards<sup>[40]</sup> for the
  machine-translated Vietnamese and French corpora.
- **No translation-quality measurement.** The Vietnamese and French
  corpora are 100 % machine-generated and have not been audited
  against human-validated references via BLEU, chrF, or COMET; the
  back-translation cells in Section 3.3.3 should therefore be read
  as bounded by unmeasured translation noise.
- **No carbon report.** The total energy expenditure of the twelve
  fine-tunes, the 2 × 3 R-Drop ablation grid, the SetFit learning
  curve, the Optuna sweep, and the twelve-model INT8 export was not
  estimated via CodeCarbon or an equivalent tool.
- **No fairness audit beyond per-class.** The error analysis in
  Section 3.5.4 stratifies errors by class and by text length, but
  not by source publication, time period, or named entities; AG
  News spans pre-2005 articles and downstream distribution shift on
  modern news has not been measured.

These scope-outs do not invalidate any reported result; they mark the
boundary between the capstone deliverable and a community-release
artefact and are listed here so that any external user can decide
whether the gap is acceptable for their use case.

## 4.2 Future Improvements

Three directions stand out as natural follow-ups, each of which would
give a measurable improvement on a specific objective from Section 1.2.2.

### 4.2.1 Better cross-lingual transfer (improves O3 and O8)

The Vietnamese and French benchmarks are both bounded by OPUS-MT
translation quality. Replacing OPUS-MT with the NLLB-200 model
(``facebook/nllb-200-distilled-600M``) or, for headline-style text,
with a state-of-the-art commercial translation API, will likely
improve F1-macro on the Vietnamese test set by 1-3 points; the
French test set, which already sits above 0.94, is likely to gain
less but should still benefit from cleaner training data. A second
improvement is to fine-tune mDeBERTa-v3 with a *shared-encoder
multi-task* setup, training jointly on English, Vietnamese, and
French AG News, instead of training one model per language.

### 4.2.2 Instruction-tuned LLMs as zero-shot baselines (improves O2)

A small instruction-tuned LLM such as ``Qwen2.5-7B-Instruct`` or
``Phi-3-mini-4k-instruct`` can serve as a zero-shot baseline that does
not need any labelled examples at all. Comparing SetFit at 64 samples per
class against an LLM zero-shot on the same test set provides a richer
data-efficiency story than SetFit alone and aligns with current 2025
practice.

### 4.2.3 Static-calibration INT8 quantisation (improves O5)

Dynamic INT8 quantisation is convenient but can degrade accuracy. A
static quantisation pass with a 200-example calibration set typically
preserves the original model's accuracy while keeping the same latency
benefit. This is a low-risk improvement that would be the first item of
the next iteration. (Confidence calibration itself — Objective O4 — is
already addressed by the post-hoc chained calibrator documented in
Section 3.5.2, which achieves ECE = 0.0285 ≤ 0.03.)

### 4.2.4 Continuous evaluation pipeline

Wrapping the evaluation phase in a GitHub Actions workflow that runs on
every push and writes the metrics to a small dashboard (Hugging Face
Datasets viewer, or a simple Streamlit app) would make regressions in
calibration or latency immediately visible. This is the natural next
step toward a *production* posture for the project.
