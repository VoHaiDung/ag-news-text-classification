__version__ = "0.1.0"

# Configure global logger for the ag_news package
import logging

logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("ag_news")

# Optional expose key utilities for convenient importing
try:
    from .train import train_model
except (ImportError, AttributeError):
    try:
        from .train import main as train_model
    except Exception:
        pass

try:
    from .ensemble import main as run_ensemble
except Exception:
    pass

__all__ = [
    "train_model",
    "run_ensemble",
    "logger",
]
