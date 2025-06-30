__version__ = "0.1.0"

# Configure global logger for the ag_news package
import logging

logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("ag_news")

# Optional: expose key utilities for convenient importing
from .data_utils import prepare_data_pipeline  # noqa: F401
from .train import train_model                 # noqa: F401
from .ensemble import main as run_ensemble     # noqa: F401
