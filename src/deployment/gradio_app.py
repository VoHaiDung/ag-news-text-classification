# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Gradio demo with SHAP highlighting for the AG News classifier.

The app exposes two tabs:

1. *Classify*: enter or paste a news snippet, see the predicted class and
   the full softmax distribution.
2. *Explain*: re-run the same prediction with SHAP and render an HTML token
   heat-map.

The model is loaded from a directory provided via the ``MODEL_DIR``
environment variable so the same file can drive any of the trained
checkpoints from a single local launch.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import shap
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)
_DEFAULT_MODEL_DIR = "outputs/transformers/ag_news_en__modernbert_large/best"
_DEFAULT_LABELS = ("World", "Sports", "Business", "Sci/Tech")

# Registered model checkpoints exposed through the Gradio model picker.
# All twelve supervised transformer checkpoints from Section 3.3 of the
# report are listed (4 English + 4 Vietnamese + 4 French); the picker
# is filtered at runtime to the matching language subset based on the
# language detected in the textbox. The first English entry is the
# "Max" model loaded at startup when no input has been entered yet.
# SetFit is excluded from the dropdown because it uses a different
# inference API; the curve results live in Section 3.3.4 and the CLI
# loader handles SetFit. The official O1 figure (F1 = 0.9528) is
# produced by the R-Drop 3-seed soft-voting ensemble documented in
# Section 3.3.2.5; the ensemble is not loaded inside the app because
# soft voting requires three checkpoints in memory at once. The
# single-model dropdown lists the strongest single seed
# (ModernBERT-large seed 42 vanilla, F1 = 0.9505) as the headline
# English entry, with ModernBERT-base flagged in the comparison table
# below as the deployment-tier Pareto-optimal model.
AVAILABLE_MODELS: dict[str, str] = {
    # English models (4 rows, primary + ablation tier)
    "ModernBERT-large (EN, 395 M, 8192 tokens, F1 0.9505)": "outputs/transformers/ag_news_en__modernbert_large/best",
    "DeBERTa-v3-base (EN, 184 M, 512 tokens, F1 0.9493)": "outputs/transformers/ag_news_en__deberta_v3_base/best",
    "ModernBERT-base (EN, 149 M, 8192 tokens, F1 0.9471)": "outputs/transformers/ag_news_en__modernbert_base/best",
    "DeBERTa-v3-small (EN, 44 M, 512 tokens, F1 0.9463)": "outputs/transformers/ag_news_en__deberta_v3_small/best",
    # Vietnamese models (4 rows, full 2x2 ablation: encoder x BT)
    "XLM-R-large vi+BT (VI, 550 M, 512 tokens, F1 0.9041)": "outputs/multilingual/ag_news_vi__xlm_r_large__vi_with_bt/best",
    "XLM-R-large vi-only (VI, 550 M, 512 tokens, F1 0.9011)": "outputs/multilingual/ag_news_vi__xlm_r_large__vi_only/best",
    "mDeBERTa-v3 vi+BT (VI, 184 M, 512 tokens, F1 0.8960)": "outputs/multilingual/ag_news_vi__mdeberta_v3__vi_with_bt/best",
    "mDeBERTa-v3 vi-only (VI, 184 M, 512 tokens, F1 0.8863)": "outputs/multilingual/ag_news_vi__mdeberta_v3__vi_only/best",
    # French models (4 rows, full 2x2 ablation: encoder x BT)
    "XLM-R-large fr-only (FR, 550 M, 512 tokens, F1 0.9466)": "outputs/multilingual/ag_news_fr__xlm_r_large__fr_only/best",
    "XLM-R-large fr+BT (FR, 550 M, 512 tokens, F1 0.9451)": "outputs/multilingual/ag_news_fr__xlm_r_large__fr_with_bt/best",
    "mDeBERTa-v3 fr+BT (FR, 184 M, 512 tokens, F1 0.9395)": "outputs/multilingual/ag_news_fr__mdeberta_v3__fr_with_bt/best",
    "mDeBERTa-v3 fr-only (FR, 184 M, 512 tokens, F1 0.9361)": "outputs/multilingual/ag_news_fr__mdeberta_v3__fr_only/best",
}
_EN_MODEL_LABELS = [k for k in AVAILABLE_MODELS if "(EN" in k]
_VI_MODEL_LABELS = [k for k in AVAILABLE_MODELS if "(VI" in k]
_FR_MODEL_LABELS = [k for k in AVAILABLE_MODELS if "(FR" in k]

# Vietnamese diacritics split into two tiers:
# - EXCLUSIVE: characters that effectively never appear in French,
#   Spanish, Portuguese or Italian (tone-marked vowels, ă, ơ, ư, đ,
#   plus the long list of combined-tone variants ệ, ợ, ổ, ấ, etc.).
#   A single occurrence is decisive evidence of Vietnamese, so we can
#   detect single-word inputs like "Việt" or "Đông" without false
#   positives on Romance text.
# - SHARED: vowels with acute, grave or circumflex (é, à, ê, ô, etc.)
#   that occur both in Vietnamese AND in French/Spanish/Portuguese.
#   These are kept in a separate set so the detector requires at
#   least two of them before voting Vietnamese, which avoids false
#   positives such as "café" or "le président" being misrouted.
_VI_DIACRITICS_EXCLUSIVE = set(
    "ảãạăằắẳẵặầấẩẫậẻẽẹềếểễệỉĩịỏõọồốổỗộơờớởỡợủũụưừứửữựỳỷỹỵđ"
)
_VI_DIACRITICS_SHARED = set("àáèéìíòóùúýâêô")
_VI_DIACRITICS = _VI_DIACRITICS_EXCLUSIVE | _VI_DIACRITICS_SHARED

# High-precision Vietnamese tokens that very rarely appear in English
# news. Both diacritic and non-diacritic (Telex/VNI) forms are listed
# so the detector works on short snippets and on Vietnamese pasted
# without diacritics from chat applications.
_VI_TOKENS_STRONG: set[str] = {
    # Pronouns and demonstratives
    "tôi", "toi", "mình", "minh", "bạn", "ban", "chúng", "chung",
    "nó", "no", "đây", "đấy", "ấy", "này", "nay", "đó",
    # Frequent function words
    "và", "va", "của", "cua", "với", "voi", "trong",
    "đã", "da", "không", "khong", "có", "co", "được", "duoc",
    "các", "cac", "để", "de", "người", "nguoi", "một", "mot",
    "những", "nhung", "cũng", "cung", "hoặc", "hoac", "sẽ", "se",
    "đang", "dang", "thì", "thi", "vì", "vi", "như", "nhu",
    "khi", "khi", "làm", "lam", "cho", "cho", "nếu", "neu",
    "nhưng", "đến", "den", "từ", "tu", "theo", "sau",
    "trước", "truoc", "hay", "mà", "ma", "bị", "bi",
    "mới", "moi", "thêm", "them", "vẫn", "van", "chỉ", "chi",
    "tại", "tai", "rất", "rat", "chưa", "chua", "đều", "deu",
    "luôn", "luon", "rồi", "roi", "thôi", "thoi", "nữa", "nua",
    "lại", "lai", "đi", "di", "ra", "vào", "vao", "lên", "len",
    # Common verbs / nouns specific to Vietnamese
    "nói", "noi", "thấy", "thay", "biết", "biet", "việc", "viec",
    "ngày", "ngay", "hôm", "hom", "năm", "lần", "lan",
    # Vietnam-specific named entities
    "việt", "viet", "nam", "hà", "ha", "nội", "sài", "sai",
    "gòn", "gon", "hồ", "ho", "chí", "chi", "minh", "đà", "nẵng", "nang",
}

# Vietnamese bigrams that survive diacritic stripping and are unlikely
# to appear in English news. A single bigram match is decisive.
_VI_BIGRAMS_STRONG: set[str] = {
    "viet nam", "ha noi", "sai gon", "ho chi minh", "da nang",
    "thai lan", "co the", "khong co", "khong phai", "duoc cho",
    "cua minh", "cua toi", "cua ban", "lam viec", "noi voi",
    "dam phan", "hoa binh", "khac nhau", "moi day", "hom nay",
    "ngay nay", "hien nay", "viec lam", "duoc phep",
}

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ỹ]+")


# French detection. Diacritic set is much smaller than Vietnamese, but
# function words like ``le``, ``la``, ``les`` and ``des`` are highly
# discriminative against English on news-style snippets.
_FR_DIACRITICS = set("àâäæçéèêëîïôœùûüÿ")
_FR_TOKENS_STRONG: set[str] = {
    # Articles + determiners (very rare in English news in this density)
    "le", "la", "les", "des", "du", "un", "une", "au", "aux",
    "ce", "cette", "ces", "son", "sa", "ses", "leur", "leurs",
    # Frequent function words
    "et", "ou", "mais", "donc", "car", "ni", "ne", "pas",
    "est", "sont", "était", "etait", "été", "ete", "être", "etre",
    "avec", "sans", "pour", "par", "sur", "sous", "dans", "vers",
    "qui", "que", "quoi", "dont", "où", "ou",
    "plus", "moins", "très", "tres", "aussi", "encore",
    "déjà", "deja", "alors", "ainsi", "puis", "ensuite",
    # Pronouns
    "il", "elle", "ils", "elles", "nous", "vous", "je", "tu",
    "lui", "leur", "moi", "toi",
    # Common verbs / nouns specific to French news context
    "selon", "après", "apres", "avant", "pendant",
    "français", "francais", "française", "francaise", "france",
    "paris", "français", "francophone",
}
_FR_BIGRAMS_STRONG: set[str] = {
    "de la", "de l", "de cette", "qui ne", "qui a", "c est",
    "il y a", "y a", "n est", "n a", "à la", "a la",
    "en france", "le president", "le président", "selon les",
    "selon le", "à partir", "a partir", "ne sont", "n ont",
}


def _is_french(text: str) -> bool:
    """Detect French via diacritics, function words or fixed bigrams.

    The function word list is intentionally biased toward closed-class
    items (articles, common verbs) that are rare in English news text;
    two matches keep the detector robust against incidental occurrences
    of single tokens such as "le" in proper nouns.
    """

    if not text:
        return False
    lowered = text.lower()
    if sum(1 for c in lowered if c in _FR_DIACRITICS) >= 2:
        return True
    tokens = _TOKEN_RE.findall(lowered)
    if len(set(tokens) & _FR_TOKENS_STRONG) >= 2:
        return True
    # Match bigrams on whole-token boundaries, not as raw substrings.
    # Padding both sides with spaces means " y a " only matches a real
    # "y" "a" token pair, never the inside of English words like
    # "galax[y a]bout" or "Su[n a]nd", which previously triggered false
    # French positives on English news text.
    joined = " " + " ".join(tokens) + " "
    return any(f" {bigram} " in joined for bigram in _FR_BIGRAMS_STRONG)


def _is_vietnamese(text: str) -> bool:
    """Detect Vietnamese (with or without diacritics).

    Aggregates three independent signals: (a) the count of
    Vietnamese-specific diacritics, (b) tokens from a curated
    high-precision word list that rarely surface in English news, and
    (c) common Vietnamese bigrams that survive Telex/VNI input methods
    even when all diacritics have been stripped. Any single signal
    above its threshold is decisive because a false positive only
    misroutes the model picker, not the prediction itself.
    """

    if not text:
        return False
    lowered = text.lower()

    # (a) A single VN-exclusive diacritic (ệ, ơ, đ, ấ, ổ, ...) is
    # decisive because those characters never appear in French,
    # Spanish or English news text. Diacritics that VN shares with
    # Romance languages (é, à, ê, ô) are intentionally ignored here
    # so that French sentences with several é/à characters do not
    # produce false positives.
    if any(c in _VI_DIACRITICS_EXCLUSIVE for c in lowered):
        return True

    # (b) High-precision token list. Two matches required for the
    # diacritic-stripped path so that English homographs ("tai" inside
    # "Taipei", "moi" inside French) do not trip the detector.
    tokens = _TOKEN_RE.findall(lowered)
    if len(set(tokens) & _VI_TOKENS_STRONG) >= 2:
        return True

    # (c) Bigram match for Telex/VNI-style undiacritised Vietnamese.
    # Pad with spaces so bigrams match on whole-token boundaries rather
    # than as raw substrings inside longer English/French words.
    joined = " " + " ".join(tokens) + " "
    return any(f" {bigram} " in joined for bigram in _VI_BIGRAMS_STRONG)


def _recommend_models(text: str) -> tuple[list[str], str]:
    """Filter the model picker based on the detected language.

    Returns ``(choices, default_value)``. When the textbox is empty
    every model is shown and the first key (the "Max" English model)
    is pre-selected. When the input is Vietnamese only the Vietnamese
    checkpoints are offered; French inputs route to the French
    checkpoints; otherwise the English checkpoints are offered with
    ModernBERT-large pre-selected. Vietnamese is checked before French
    because Vietnamese diacritics are far more distinctive (no overlap
    with French/English) - if any Vietnamese marker is present the
    text is Vietnamese.
    """

    if not text.strip():
        return list(AVAILABLE_MODELS.keys()), next(iter(AVAILABLE_MODELS))
    if _is_vietnamese(text):
        return list(_VI_MODEL_LABELS), _VI_MODEL_LABELS[0]
    if _is_french(text):
        return list(_FR_MODEL_LABELS), _FR_MODEL_LABELS[0]
    return list(_EN_MODEL_LABELS), _EN_MODEL_LABELS[0]


@dataclass
class _DemoState:
    """Lazy container for the loaded model, tokenizer and SHAP explainer."""

    model_dir: Path
    label_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _logger.info("Loading classification model from %s", self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        native_limit = int(getattr(self.model.config, "max_position_embeddings", 512))
        # Reserve a couple of slots for the CLS/SEP special tokens so that
        # tokeniser truncation never overflows the model's positional table.
        self.native_token_limit = max(8, native_limit - 2)
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,
            truncation=True,
            max_length=self.native_token_limit,
        )
        self.predict_fn = self._build_predict_fn()
        self.explainer = shap.Explainer(
            self.predict_fn,
            shap.maskers.Text(r"\W+"),
            output_names=list(self.label_names),
        )

    def count_tokens(self, text: str) -> int:
        """Return the number of subword tokens (excluding special tokens)."""

        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _build_predict_fn(self) -> Callable[[list[str]], np.ndarray]:
        label2id = self.model.config.label2id

        def predict_proba(texts) -> np.ndarray:
            # SHAP passes a numpy array of strings; the HF tokenizer only
            # accepts a plain Python str or list[str], so normalise first.
            if isinstance(texts, np.ndarray):
                texts = [str(t) for t in texts.tolist()]
            elif isinstance(texts, str):
                texts = [texts]
            outputs = self.pipeline(list(texts))
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

    # The active state is held in a mutable container so that the model
    # picker can swap it from a callback without rebuilding the whole UI.
    active: dict[str, _DemoState | dict[str, object]] = {
        "state": state or _build_state(),
        "long_doc": {},
    }

    def _ensure_long_doc():
        long_doc = active["long_doc"]
        if "classifier" not in long_doc:
            from src.inference.long_doc import LongDocumentClassifier

            current_state = active["state"]  # type: ignore[assignment]
            long_doc["classifier"] = LongDocumentClassifier(current_state.model_dir)
        return long_doc["classifier"]

    def _switch_model(display_label: str) -> str:
        """Load a new checkpoint into the active slot and reset caches.

        The function short-circuits when the requested checkpoint is
        already the active one so that repeated dropdown updates (for
        example, those issued by the auto-routing on text input) do not
        trigger a redundant model reload.
        """

        rel_path = AVAILABLE_MODELS.get(display_label)
        if rel_path is None:
            return f"Unknown model: {display_label}"
        new_dir = Path(rel_path)
        if not new_dir.is_absolute():
            new_dir = Path(__file__).resolve().parents[2] / new_dir
        if not new_dir.exists():
            return f"Model directory not found on disk: {new_dir}"
        current_state = active["state"]  # type: ignore[assignment]
        if Path(current_state.model_dir).resolve() == new_dir.resolve():
            return (
                f"Active model: **{display_label}**. "
                f"Native token limit: {current_state.native_token_limit}."
            )
        active["state"] = _DemoState(model_dir=new_dir, label_names=_DEFAULT_LABELS)
        active["long_doc"] = {}
        new_state = active["state"]  # type: ignore[assignment]
        return (
            f"Loaded **{display_label}**. Native token limit: "
            f"{new_state.native_token_limit}."
        )

    def _on_text_change(text: str):
        """Show and filter the model picker when the textbox content changes.

        The picker stays hidden while the textbox is empty so the demo
        opens with a clean surface and reveals model choices only once
        the user has typed something the detector can act on. Routing
        is delegated to :func:`_recommend_models` so the Vietnamese,
        French and English branches stay in a single decision point.
        """

        if not text.strip():
            return gr.update(visible=False)
        choices, default_value = _recommend_models(text)
        return gr.update(visible=True, choices=choices, value=default_value)

    def classify(text: str) -> tuple[str, dict[str, float], str]:
        if not text.strip():
            return "Please enter a news snippet.", {}, ""
        state = active["state"]  # type: ignore[assignment]
        token_count = state.count_tokens(text)
        if token_count > state.native_token_limit:
            classifier = _ensure_long_doc()
            prediction = classifier.classify(text)
            mode_info = (
                f"Mode: sliding window (Pappagari et al. 2019). "
                f"{prediction.num_tokens} tokens > "
                f"{state.native_token_limit}-token native limit -> "
                f"{prediction.num_windows} window(s) of "
                f"{classifier.window_size} tokens, stride {classifier.stride}, "
                f"mean-pooled softmax."
            )
            return (
                prediction.label,
                {
                    name: float(p)
                    for name, p in zip(prediction.label_names, prediction.probabilities)
                },
                mode_info,
            )
        probs = state.predict_fn([text])[0]
        index = int(np.argmax(probs))
        mode_info = (
            f"Mode: single forward pass. {token_count} tokens "
            f"(fits in {state.native_token_limit}-token native limit)."
        )
        return (
            state.label_names[index],
            {name: float(prob) for name, prob in zip(state.label_names, probs)},
            mode_info,
        )

    def explain(text: str) -> tuple[str, str]:
        if not text.strip():
            return "<p>Please enter a news snippet.</p>", ""
        state = active["state"]  # type: ignore[assignment]
        token_count = state.count_tokens(text)
        if token_count <= state.native_token_limit:
            probs = state.predict_fn([text])[0]
            predicted_idx = int(np.argmax(probs))
            explanation = state.explainer([text])
            # Slice the Explanation object on the class axis so that the
            # rendered SHAP plot defaults to the predicted class rather
            # than always showing label index 0. Users can still inspect
            # other classes by re-running with a different input.
            focused = explanation[..., predicted_idx]
            mode_info = (
                f"Mode: SHAP on full input for predicted class "
                f"**{state.label_names[predicted_idx]}** "
                f"({float(probs[predicted_idx]) * 100:.1f}%). "
                f"{token_count} tokens (fits in {state.native_token_limit}-token "
                f"native limit)."
            )
            return shap.plots.text(focused, display=False), mode_info

        # Long input: classify with sliding window, locate the window that
        # contributes most to the document-level prediction, then run SHAP
        # only on that window. This bounds the SHAP cost to a single
        # window while still attributing the explanation to the textual
        # region the model relied on (Lei et al. 2016 rationale-extraction
        # idea adapted to a sliding-window classifier).
        classifier = _ensure_long_doc()
        prediction = classifier.classify(text)
        chosen = int(np.argmax(prediction.window_probabilities[:, prediction.label_id]))
        chosen_score = float(prediction.window_probabilities[chosen, prediction.label_id])
        encoded = classifier.tokenizer.encode(text, add_special_tokens=False)
        window_ids = classifier._window_ids(encoded)[chosen]
        window_text = classifier.tokenizer.decode(
            window_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        explanation = state.explainer([window_text])
        focused = explanation[..., prediction.label_id]
        mode_info = (
            f"Mode: salient-window SHAP for predicted class "
            f"**{prediction.label}**. {prediction.num_tokens} tokens > "
            f"{state.native_token_limit}-token native limit -> "
            f"{prediction.num_windows} window(s); explained window #{chosen + 1} "
            f"({len(window_ids)} tokens, P({prediction.label}) = {chosen_score:.3f})."
        )
        return shap.plots.text(focused, display=False), mode_info

    description = (
        '<div style="text-align: justify;">\n\n'
        "**AG News Text Classification** is a complete end-to-end "
        "study of modern news topic classification, developed as a "
        "capstone for the Samsung Innovation Campus AI course. The "
        "project re-examines a classic four-class news topic "
        "benchmark using the contemporary NLP toolkit: large "
        "pre-trained transformer encoders, machine-translated "
        "multilingual extensions with back-translation augmentation, "
        "few-shot sentence-embedding classifiers, post-hoc confidence "
        "calibration, label-noise auditing, model explainability and "
        "production-grade INT8 ONNX deployment.\n\n"
        "The system classifies short news headlines and abstracts "
        "into one of four broad topics - **World**, **Sports**, "
        "**Business** and **Sci/Tech** - drawn from the AG News "
        "corpus (Zhang, Zhao and LeCun, 2015), a long-standing "
        "topic-classification benchmark with 120,000 training "
        "articles and 7,600 held-out test articles balanced evenly "
        "across the four classes.\n\n"
        "</div>\n\n"
        "Twelve fine-tuned transformer encoders are exposed for "
        "side-by-side comparison on the same input, covering three "
        "languages and two encoder families per language:\n\n"
        "- **English (4 encoders)** - DeBERTa-v3-small, "
        "DeBERTa-v3-base, ModernBERT-base and "
        "**ModernBERT-large** (strongest English encoder), "
        "spanning two scale tiers from 44 M to 395 M parameters and "
        "two native context lengths (512 and 8192 tokens).\n"
        "- **Vietnamese (4 encoders)** - mDeBERTa-v3-base and "
        "**XLM-RoBERTa-large + back-translation** "
        "(strongest Vietnamese encoder), each backbone fine-tuned in "
        "two augmentation regimes (target-only and target plus "
        "back-translation) on a machine-translated Vietnamese "
        "AG News corpus.\n"
        "- **French (4 encoders)** - the same encoder x "
        "back-translation 2 x 2 grid replicated on a "
        "machine-translated French AG News corpus, with "
        "**XLM-RoBERTa-large (fr-only)** the strongest French "
        "encoder.\n\n"
        "Two tabs are provided:\n\n"
        "- **Classify** - predicts the topic and shows the full "
        "softmax distribution over the four classes, together with "
        "the confidence of the top prediction.\n"
        "- **Explain** - renders a SHAP token-level heat-map "
        "highlighting which words pushed the model toward the "
        "predicted class. Red tokens push toward the class; blue "
        "tokens push away."
    )

    # Resolve the dropdown label that matches the initially loaded model
    # so the picker reflects reality at startup.
    initial_state_obj = active["state"]  # type: ignore[assignment]
    initial_model_path = str(initial_state_obj.model_dir).replace("\\", "/")
    initial_label = next(
        (k for k, v in AVAILABLE_MODELS.items() if initial_model_path.endswith(v)),
        next(iter(AVAILABLE_MODELS)),
    )

    full_width_css = """
    .gradio-container { max-width: 100% !important; padding: 0 24px !important; }
    .main { max-width: 100% !important; }
    /* Brand row: pack the project logo, separator and team logo into
       a single horizontally-centered flex line with a tight 16 px gap
       between elements, so the three brand items read as a single
       grouping. The default Gradio row stretches each cell to fill the
       page width, which is why the spacer-column layout left visible
       gaps. */
    #brand-row {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 16px !important;
        flex-wrap: nowrap !important;
        margin-bottom: 0.6rem !important;
    }
    #brand-row > * { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
    #project-logo img,
    #team-logo img { object-fit: contain; display: block; }
    /* Hide the Gradio image toolbar (fullscreen / download / share)
       on the brand assets without removing the image itself.
       The toolbar buttons render as <svg> nodes sitting in absolutely
       positioned wrappers above the image; targeting them by aria-label
       and class fragments keeps the underlying <img> visible. */
    #project-logo [aria-label*="ownload"],
    #project-logo [aria-label*="ullscreen"],
    #project-logo [aria-label*="hare"],
    #project-logo [aria-label*="emove"],
    #team-logo [aria-label*="ownload"],
    #team-logo [aria-label*="ullscreen"],
    #team-logo [aria-label*="hare"],
    #team-logo [aria-label*="emove"] { display: none !important; }
    #project-logo .icon-buttons,
    #team-logo .icon-buttons,
    #project-logo [class*="icon-buttons"],
    #team-logo [class*="icon-buttons"],
    #project-logo [class*="button-wrap"],
    #team-logo [class*="button-wrap"] { display: none !important; }
    """

    # Resolve project + team logos. The files are optional: when they are
    # missing the UI falls back to a plain title so the launch still works
    # on a freshly cloned repository that has not yet committed any brand
    # assets. Both PNGs are inlined as base64 data URIs so the browser
    # cannot fail to load them through Gradio's file-serving allowlist,
    # which has previously dropped one of the two logos at random on
    # Windows hosts.
    import base64

    _assets_root = Path(__file__).resolve().parents[2] / "assets"
    project_logo_file = _assets_root / "logo-ag-news-text-classification.png"
    team_logo_file = _assets_root / "logo-aimer-pam.png"

    def _logo_data_uri(path: Path) -> str | None:
        if not path.exists():
            return None
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"

    project_logo_uri = _logo_data_uri(project_logo_file)
    team_logo_uri = _logo_data_uri(team_logo_file)

    # The CSS string is attached to the Blocks instance so that ``main()``
    # can read it back and pass it to ``launch()``; Gradio 6 deprecated the
    # ``css`` argument on the constructor.
    with gr.Blocks(title="AG News Classifier") as demo:
        demo._custom_css = full_width_css
        gr.HTML(
            "<h1 style='text-align:center; font-size:2.8rem; "
            "font-weight:700; margin:0.4em 0 0.3em 0; letter-spacing:0.5px;'>"
            "AG News Text Classification</h1>"
        )
        # Brand row: render both logos as inline <img> tags inside a single
        # flex container so they sit on the same baseline and stay centered
        # together as a unit. Rendering via gr.HTML rather than gr.Image
        # bypasses Gradio's file-serving layer entirely.
        _logo_imgs: list[str] = []
        if project_logo_uri is not None:
            _logo_imgs.append(
                f'<img src="{project_logo_uri}" alt="Project logo" '
                f'style="height:70px; width:auto; object-fit:contain; display:block;"/>'
            )
        _logo_imgs.append(
            '<div style="width:1px; height:55px; '
            'background:rgba(255,255,255,0.30);"></div>'
        )
        if team_logo_uri is not None:
            _logo_imgs.append(
                f'<img src="{team_logo_uri}" alt="Team logo" '
                f'style="height:70px; width:auto; object-fit:contain; display:block;"/>'
            )
        gr.HTML(
            '<div style="display:flex; align-items:center; '
            'justify-content:center; gap:16px; margin:0.4em 0 0.8em 0;">'
            + "".join(_logo_imgs)
            + "</div>"
        )
        gr.Markdown(description)
        model_picker = gr.Dropdown(
            choices=list(AVAILABLE_MODELS.keys()),
            value=initial_label,
            label="Model checkpoint",
            interactive=True,
            visible=False,
        )
        model_status = gr.Markdown(
            f"Loaded **{initial_label}**. "
            f"Native token limit: {initial_state_obj.native_token_limit}."
        )
        model_picker.change(fn=_switch_model, inputs=model_picker, outputs=model_status)

        comparison_table = """
<div style="margin: 1.2rem auto; max-width: 1280px; text-align: center;">
  <h3 style="margin-bottom: 0.4rem;">Model comparison</h3>
  <table style="margin: 0.6rem auto; border-collapse: collapse; width: 100%;
                font-size: 0.93em; text-align: left;">
    <thead>
      <tr style="background: rgba(255,255,255,0.05);
                 border-bottom: 2px solid rgba(255,255,255,0.25);">
        <th style="padding: 8px 12px; white-space: nowrap;">Model</th>
        <th style="padding: 8px 12px; white-space: nowrap;">Family</th>
        <th style="padding: 8px 12px; text-align: center; white-space: nowrap;">Language</th>
        <th style="padding: 8px 12px; text-align: right; white-space: nowrap;">Parameters</th>
        <th style="padding: 8px 12px; text-align: right; white-space: nowrap;">Native context</th>
        <th style="padding: 8px 12px; text-align: right; white-space: nowrap;">Test F1-macro</th>
        <th style="padding: 8px 12px; white-space: nowrap;">Notes</th>
      </tr>
    </thead>
    <tbody>
      <tr><td style="padding:6px 12px;"><strong>ModernBERT-large</strong></td><td style="padding:6px 12px;">ModernBERT</td><td style="padding:6px 12px;text-align:center;">English</td><td style="padding:6px 12px;text-align:right;">395 M</td><td style="padding:6px 12px;text-align:right;"><strong>8192 tokens</strong></td><td style="padding:6px 12px;text-align:right;"><strong>0.9505</strong></td><td style="padding:6px 12px;">Primary English encoder; strongest single-model F1.</td></tr>
      <tr><td style="padding:6px 12px;">DeBERTa-v3-base</td><td style="padding:6px 12px;">DeBERTa-v3</td><td style="padding:6px 12px;text-align:center;">English</td><td style="padding:6px 12px;text-align:right;">184 M</td><td style="padding:6px 12px;text-align:right;">512 tokens</td><td style="padding:6px 12px;text-align:right;">0.9493</td><td style="padding:6px 12px;">English encoder at the base scale tier; same recipe as ModernBERT-large.</td></tr>
      <tr><td style="padding:6px 12px;">ModernBERT-base</td><td style="padding:6px 12px;">ModernBERT</td><td style="padding:6px 12px;text-align:center;">English</td><td style="padding:6px 12px;text-align:right;">149 M</td><td style="padding:6px 12px;text-align:right;">8192 tokens</td><td style="padding:6px 12px;text-align:right;">0.9471</td><td style="padding:6px 12px;">Compact English encoder with the same long-context support as ModernBERT-large.</td></tr>
      <tr><td style="padding:6px 12px;">DeBERTa-v3-small</td><td style="padding:6px 12px;">DeBERTa-v3</td><td style="padding:6px 12px;text-align:center;">English</td><td style="padding:6px 12px;text-align:right;">44 M</td><td style="padding:6px 12px;text-align:right;">512 tokens</td><td style="padding:6px 12px;text-align:right;">0.9463</td><td style="padding:6px 12px;">Smallest English encoder by parameter count; fastest to fine-tune.</td></tr>
      <tr><td style="padding:6px 12px;"><strong>XLM-R-large (vi + BT)</strong></td><td style="padding:6px 12px;">XLM-R</td><td style="padding:6px 12px;text-align:center;">Vietnamese</td><td style="padding:6px 12px;text-align:right;">550 M</td><td style="padding:6px 12px;text-align:right;">512 tokens</td><td style="padding:6px 12px;text-align:right;"><strong>0.9041</strong></td><td style="padding:6px 12px;">Strongest Vietnamese encoder; trained with back-translation augmentation.</td></tr>
      <tr><td style="padding:6px 12px;">XLM-R-large (vi-only)</td><td style="padding:6px 12px;">XLM-R</td><td style="padding:6px 12px;text-align:center;">Vietnamese</td><td style="padding:6px 12px;text-align:right;">550 M</td><td style="padding:6px 12px;text-align:right;">512 tokens</td><td style="padding:6px 12px;text-align:right;">0.9011</td><td style="padding:6px 12px;">Same backbone without back-translation; isolates the augmentation effect.</td></tr>
      <tr><td style="padding:6px 12px;">mDeBERTa-v3 (vi + BT)</td><td style="padding:6px 12px;">mDeBERTa</td><td style="padding:6px 12px;text-align:center;">Vietnamese</td><td style="padding:6px 12px;text-align:right;">184 M</td><td style="padding:6px 12px;text-align:right;">512 tokens</td><td style="padding:6px 12px;text-align:right;">0.8960</td><td style="padding:6px 12px;">Compact multilingual encoder with back-translation; +0.97 pp over the vi-only baseline.</td></tr>
      <tr><td style="padding:6px 12px;">mDeBERTa-v3 (vi-only)</td><td style="padding:6px 12px;">mDeBERTa</td><td style="padding:6px 12px;text-align:center;">Vietnamese</td><td style="padding:6px 12px;text-align:right;">184 M</td><td style="padding:6px 12px;text-align:right;">512 tokens</td><td style="padding:6px 12px;text-align:right;">0.8863</td><td style="padding:6px 12px;">Baseline Vietnamese encoder without augmentation.</td></tr>
      <tr><td style="padding:6px 12px;"><strong>XLM-R-large (fr-only)</strong></td><td style="padding:6px 12px;">XLM-R</td><td style="padding:6px 12px;text-align:center;">French</td><td style="padding:6px 12px;text-align:right;">550 M</td><td style="padding:6px 12px;text-align:right;">512 tokens</td><td style="padding:6px 12px;text-align:right;"><strong>0.9466</strong></td><td style="padding:6px 12px;">Strongest French encoder; surprisingly back-translation hurts XLM-R on French.</td></tr>
      <tr><td style="padding:6px 12px;">XLM-R-large (fr + BT)</td><td style="padding:6px 12px;">XLM-R</td><td style="padding:6px 12px;text-align:center;">French</td><td style="padding:6px 12px;text-align:right;">550 M</td><td style="padding:6px 12px;text-align:right;">512 tokens</td><td style="padding:6px 12px;text-align:right;">0.9451</td><td style="padding:6px 12px;">Same backbone with back-translation; slight regression compared to fr-only.</td></tr>
      <tr><td style="padding:6px 12px;">mDeBERTa-v3 (fr + BT)</td><td style="padding:6px 12px;">mDeBERTa</td><td style="padding:6px 12px;text-align:center;">French</td><td style="padding:6px 12px;text-align:right;">184 M</td><td style="padding:6px 12px;text-align:right;">512 tokens</td><td style="padding:6px 12px;text-align:right;">0.9395</td><td style="padding:6px 12px;">Compact multilingual encoder with back-translation; +0.34 pp over the fr-only baseline.</td></tr>
      <tr><td style="padding:6px 12px;">mDeBERTa-v3 (fr-only)</td><td style="padding:6px 12px;">mDeBERTa</td><td style="padding:6px 12px;text-align:center;">French</td><td style="padding:6px 12px;text-align:right;">184 M</td><td style="padding:6px 12px;text-align:right;">512 tokens</td><td style="padding:6px 12px;text-align:right;">0.9361</td><td style="padding:6px 12px;">Baseline French encoder without augmentation.</td></tr>
      <tr><td style="padding:6px 12px;">SetFit K=64</td><td style="padding:6px 12px;">Sentence-Transformer few-shot</td><td style="padding:6px 12px;text-align:center;">English</td><td style="padding:6px 12px;text-align:right;">110 M</td><td style="padding:6px 12px;text-align:right;">512 tokens</td><td style="padding:6px 12px;text-align:right;">0.8755</td><td style="padding:6px 12px;">Few-shot encoder trained on only 64 labelled examples per class (256 total).</td></tr>
    </tbody>
  </table>
</div>
"""

        with gr.Tab("Classify"):
            text_in = gr.Textbox(label="News snippet", lines=4)
            with gr.Row():
                label_out = gr.Label(label="Predicted class")
                proba_out = gr.Label(label="Class probabilities")
            mode_out = gr.Markdown(label="Inference mode")
            classify_btn = gr.Button("Classify")
            classify_btn.click(
                fn=classify,
                inputs=text_in,
                outputs=[label_out, proba_out, mode_out],
            )
            gr.HTML(comparison_table)
            # Auto-route the dropdown based on the language of the input
            # whenever the textbox content changes. The dropdown change
            # handler still drives the actual model load, with a
            # short-circuit when the requested model is already active.
            text_in.change(fn=_on_text_change, inputs=text_in, outputs=model_picker)
        with gr.Tab("Explain"):
            text_explain = gr.Textbox(label="News snippet", lines=4)
            html_out = gr.HTML(label="SHAP explanation")
            explain_mode_out = gr.Markdown(label="Inference mode")
            explain_btn = gr.Button("Explain")
            explain_btn.click(
                fn=explain,
                inputs=text_explain,
                outputs=[html_out, explain_mode_out],
            )
            gr.HTML(comparison_table)
            text_explain.change(fn=_on_text_change, inputs=text_explain, outputs=model_picker)

        gr.HTML(
            "<div style='text-align:center; margin: 1.5rem auto 0.6rem auto; "
            "padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.12); "
            "font-size: 0.95rem; opacity: 0.75;'>"
            "AG News Text Classification &mdash; Developed by Vo Hai Dung"
            "</div>"
        )
    return demo


def main() -> None:
    demo = build_demo()
    share = bool(int(os.environ.get("GRADIO_SHARE", "0")))
    assets_root = Path(__file__).resolve().parents[2] / "assets"
    css = getattr(demo, "_custom_css", None)
    demo.launch(share=share, allowed_paths=[str(assets_root)], css=css)


if __name__ == "__main__":
    main()
