"""Gradio demo with SHAP highlighting for the AG News classifier.

The app exposes two tabs:

1. *Classify*: enter or paste a news snippet, see the predicted class and
   the full softmax distribution.
2. *Explain*: re-run the same prediction with SHAP and render an HTML token
   heat-map.

The model is loaded from a directory provided via the ``MODEL_DIR``
environment variable so the same file can drive both a local launch and the
Hugging Face Spaces deployment.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)
_DEFAULT_MODEL_DIR = "outputs/transformers/ag_news_en__deberta_v3_small/best"
_DEFAULT_LABELS = ("World", "Sports", "Business", "Sci/Tech")


@dataclass
class _DemoState:
    """Lazy container for the loaded model, tokenizer and SHAP explainer."""

    model_dir: Path
    label_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _logger.info("Loading classification model from %s", self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            truncation=True,
            max_length=256,
        )
        self.predict_fn = self._build_predict_fn()
        self.explainer = shap.Explainer(
            self.predict_fn,
            shap.maskers.Text(r"\W+"),
            output_names=list(self.label_names),
        )

    def _build_predict_fn(self) -> Callable[[list[str]], np.ndarray]:
        label2id = self.model.config.label2id

        def predict_proba(texts: list[str]) -> np.ndarray:
            outputs = self.pipeline(texts)
            n_classes = len(outputs[0])
            matrix = np.zeros((len(outputs), n_classes), dtype=np.float64)
            for row, scores in enumerate(outputs):
                for entry in scores:
                    matrix[row, label2id[entry["label"]]] = entry["score"]
            return matrix

        return predict_proba


def _build_state() -> _DemoState:
    model_dir = Path(os.environ.get("MODEL_DIR", _DEFAULT_MODEL_DIR))
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory '{model_dir}' does not exist. Set MODEL_DIR or run Phase 4."
        )
    return _DemoState(model_dir=model_dir, label_names=_DEFAULT_LABELS)


def build_demo(state: _DemoState | None = None) -> gr.Blocks:
    """Construct the Gradio Blocks UI."""

    state = state or _build_state()

    def classify(text: str) -> tuple[str, dict[str, float]]:
        if not text.strip():
            return "Please enter a news snippet.", {}
        probs = state.predict_fn([text])[0]
        prediction = int(np.argmax(probs))
        return state.label_names[prediction], {
            name: float(prob) for name, prob in zip(state.label_names, probs)
        }

    def explain(text: str) -> str:
        if not text.strip():
            return "<p>Please enter a news snippet.</p>"
        explanation = state.explainer([text])
        return shap.plots.text(explanation, display=False)

    description = (
        "AG News topic classifier. Tab 1 returns the predicted topic; tab 2 "
        "displays a SHAP token-level heat-map for the same prediction."
    )

    with gr.Blocks(title="AG News Classifier") as demo:
        gr.Markdown("# AG News Classifier")
        gr.Markdown(description)
        with gr.Tab("Classify"):
            text_in = gr.Textbox(label="News snippet", lines=4)
            with gr.Row():
                label_out = gr.Label(label="Predicted class")
                proba_out = gr.Label(label="Class probabilities")
            classify_btn = gr.Button("Classify")
            classify_btn.click(fn=classify, inputs=text_in, outputs=[label_out, proba_out])
        with gr.Tab("Explain"):
            text_explain = gr.Textbox(label="News snippet", lines=4)
            html_out = gr.HTML(label="SHAP explanation")
            explain_btn = gr.Button("Explain")
            explain_btn.click(fn=explain, inputs=text_explain, outputs=html_out)
    return demo


def main() -> None:
    demo = build_demo()
    share = bool(int(os.environ.get("GRADIO_SHARE", "0")))
    demo.launch(share=share)


if __name__ == "__main__":
    main()
