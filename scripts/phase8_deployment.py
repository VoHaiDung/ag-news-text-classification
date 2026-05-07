"""Phase 8 entry point: demo and deployment.

Mapped Work Breakdown Structure tasks:

* 8.1.1 Design the Gradio UI layout (input plus output panels)
* 8.1.2 Integrate model inference (delegated to ``src.deployment.gradio_app``)
* 8.1.3 Add SHAP highlight visualisation
* 8.2.1 Convert PyTorch -> ONNX
* 8.2.2 Apply INT8 quantization
* 8.3.1 Set up the Hugging Face Spaces config and ``requirements.txt``
* 8.3.2 Push and test the deployment
* 8.4   Write the README and Model Card (Hugging Face standard)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.deployment.gradio_app import build_demo
from src.deployment.onnx_export import export_to_onnx
from src.deployment.quantization import quantize_int8
from src.utils import configure_logging, ensure_dir, get_logger
from src.utils.paths import OUTPUTS_DIR, REPORTS_DIR

_logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8 - demo and deployment.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="HuggingFace fine-tuned model directory.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["onnx", "quantize", "spaces", "launch"],
        choices=["onnx", "quantize", "spaces", "launch"],
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "deployment")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app via tunnel.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _prepare_spaces_directory(*, model_dir: Path, target_dir: Path) -> Path:
    """Assemble a directory ready for ``huggingface-cli upload`` to a Space."""

    space_dir = ensure_dir(target_dir)
    shutil.copytree(model_dir, space_dir / "model", dirs_exist_ok=True)

    spaces_requirements = "\n".join(
        [
            "transformers>=4.41",
            "torch>=2.2",
            "gradio>=4.31",
            "shap>=0.45",
            "onnxruntime>=1.18",
            "optimum[onnxruntime]>=1.20",
        ]
    )
    (space_dir / "requirements.txt").write_text(spaces_requirements + "\n", encoding="utf-8")

    app_text = (
        "import os\n"
        "from src.deployment.gradio_app import build_demo\n"
        "\n"
        "os.environ.setdefault(\"MODEL_DIR\", \"model\")\n"
        "demo = build_demo()\n"
        "if __name__ == \"__main__\":\n"
        "    demo.launch()\n"
    )
    (space_dir / "app.py").write_text(app_text, encoding="utf-8")

    model_card = (REPORTS_DIR / "model_card.md").read_text(encoding="utf-8") if (REPORTS_DIR / "model_card.md").exists() else "# Model card\n"
    (space_dir / "README.md").write_text(model_card, encoding="utf-8")

    return space_dir


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    output_dir = ensure_dir(args.output_dir)

    if "onnx" in args.steps:
        report = export_to_onnx(args.model_dir, output_dir=output_dir / "onnx")
        _logger.info("ONNX export written to %s", report.output_dir)
    if "quantize" in args.steps:
        report = quantize_int8(output_dir / "onnx", output_dir=output_dir / "onnx_int8")
        _logger.info("Quantised ONNX written to %s", report.quantized_model_path)
    if "spaces" in args.steps:
        space_dir = _prepare_spaces_directory(
            model_dir=args.model_dir, target_dir=output_dir / "huggingface_space"
        )
        _logger.info("Hugging Face Space staged at %s", space_dir)
    if "launch" in args.steps:
        demo = build_demo()
        demo.launch(share=args.share)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
