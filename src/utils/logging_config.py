"""
Logging configuration for AG News Text Classification Framework.

Provides centralized logging setup with support for multiple handlers,
formatters, and log levels for different components.
"""

import logging
import logging.handlers
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import colorlog
import warnings
from functools import lru_cache

# Suppress warnings from libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Log format templates
FORMATS = {
    "simple": "%(levelname)s - %(message)s",
    "standard": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "detailed": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s",
    "json": '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "file": "%(filename)s", "line": %(lineno)d}',
    "colored": "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s - %(message)s",
    "research": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
}

# Color scheme for colored output
COLOR_SCHEME = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}

# Component-specific log levels
COMPONENT_LEVELS = {
    "src.models": logging.INFO,
    "src.training": logging.INFO,
    "src.data": logging.INFO,
    "src.evaluation": logging.INFO,
    "src.api": logging.WARNING,
    "transformers": logging.WARNING,
    "datasets": logging.WARNING,
    "torch": logging.WARNING,
    "urllib3": logging.ERROR,
    "requests": logging.WARNING,
    "PIL": logging.WARNING,
}

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", "funcName",
                          "levelname", "levelno", "lineno", "module", "msecs",
                          "message", "pathname", "process", "processName", "relativeCreated",
                          "thread", "threadName", "exc_info", "exc_text", "stack_info"]:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)

class ExperimentLogFormatter(logging.Formatter):
    """Custom formatter for experiment tracking."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with experiment context."""
        # Add experiment context if available
        if hasattr(record, "experiment_id"):
            record.msg = f"[EXP:{record.experiment_id}] {record.msg}"
        
        if hasattr(record, "epoch"):
            record.msg = f"[Epoch {record.epoch}] {record.msg}"
        
        if hasattr(record, "step"):
            record.msg = f"[Step {record.step}] {record.msg}"
        
        return super().format(record)

class ProgressLogFilter(logging.Filter):
    """Filter to handle progress bar logs from tqdm."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out progress bar updates."""
        # Skip tqdm progress updates
        if "it/s" in record.getMessage() or "s/it" in record.getMessage():
            return False
        return True

class MetricsLogger:
    """Logger specifically for metrics tracking."""
    
    def __init__(self, name: str = "metrics"):
        """Initialize metrics logger."""
        self.logger = logging.getLogger(f"metrics.{name}")
        self.metrics_history = []
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log metrics with optional step and epoch information.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
            epoch: Training epoch
            prefix: Prefix for metric names
        """
        # Create log message
        metric_strs = []
        for key, value in metrics.items():
            if isinstance(value, float):
                metric_strs.append(f"{prefix}{key}: {value:.4f}")
            else:
                metric_strs.append(f"{prefix}{key}: {value}")
        
        message = " | ".join(metric_strs)
        
        # Add context
        extra = {}
        if step is not None:
            extra["step"] = step
        if epoch is not None:
            extra["epoch"] = epoch
        
        # Log and store
        self.logger.info(message, extra=extra)
        self.metrics_history.append({
            "metrics": metrics,
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now(),
        })
    
    def get_history(self) -> list:
        """Get metrics history."""
        return self.metrics_history

class LoggerManager:
    """Manager for centralized logger configuration."""
    
    def __init__(self):
        """Initialize logger manager."""
        self.loggers = {}
        self.handlers = {}
        self.configured = False
    
    def setup(
        self,
        log_level: Union[str, int] = "INFO",
        log_dir: Optional[Path] = None,
        log_file: Optional[str] = None,
        format_style: str = "standard",
        use_colors: bool = True,
        json_logs: bool = False,
        experiment_id: Optional[str] = None,
    ):
        """
        Setup logging configuration.
        
        Args:
            log_level: Default log level
            log_dir: Directory for log files
            log_file: Log file name
            format_style: Format style to use
            use_colors: Whether to use colored output
            json_logs: Whether to use JSON format
            experiment_id: Experiment ID for tracking
        """
        if self.configured:
            return
        
        # Create log directory
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = self._create_console_handler(
            format_style, use_colors, json_logs
        )
        root_logger.addHandler(console_handler)
        self.handlers["console"] = console_handler
        
        # File handler
        if log_dir and log_file:
            file_handler = self._create_file_handler(
                log_dir / log_file, json_logs
            )
            root_logger.addHandler(file_handler)
            self.handlers["file"] = file_handler
        
        # Rotating file handler for long-running experiments
        if log_dir:
            rotating_handler = self._create_rotating_handler(
                log_dir / "experiment.log", json_logs
            )
            root_logger.addHandler(rotating_handler)
            self.handlers["rotating"] = rotating_handler
        
        # Set component-specific levels
        for component, level in COMPONENT_LEVELS.items():
            logging.getLogger(component).setLevel(level)
        
        # Store experiment ID for context
        if experiment_id:
            logging.LoggerAdapter(root_logger, {"experiment_id": experiment_id})
        
        self.configured = True
        
        # Log setup completion
        root_logger.info(f"Logging configured - Level: {log_level}, Format: {format_style}")
    
    def _create_console_handler(
        self,
        format_style: str,
        use_colors: bool,
        json_logs: bool
    ) -> logging.Handler:
        """Create console handler."""
        handler = logging.StreamHandler(sys.stdout)
        
        if json_logs:
            handler.setFormatter(JSONFormatter())
        elif use_colors and format_style == "colored":
            handler.setFormatter(
                colorlog.ColoredFormatter(
                    FORMATS["colored"],
                    log_colors=COLOR_SCHEME
                )
            )
        else:
            handler.setFormatter(
                ExperimentLogFormatter(FORMATS.get(format_style, FORMATS["standard"]))
            )
        
        handler.addFilter(ProgressLogFilter())
        return handler
    
    def _create_file_handler(
        self,
        log_file: Path,
        json_logs: bool
    ) -> logging.Handler:
        """Create file handler."""
        handler = logging.FileHandler(log_file, encoding="utf-8")
        
        if json_logs:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(
                ExperimentLogFormatter(FORMATS["detailed"])
            )
        
        return handler
    
    def _create_rotating_handler(
        self,
        log_file: Path,
        json_logs: bool,
        max_bytes: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 10
    ) -> logging.Handler:
        """Create rotating file handler."""
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        
        if json_logs:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(
                ExperimentLogFormatter(FORMATS["detailed"])
            )
        
        return handler
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger.
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def set_level(self, level: Union[str, int], logger_name: Optional[str] = None):
        """
        Set log level.
        
        Args:
            level: Log level
            logger_name: Specific logger name (None for root)
        """
        level_int = getattr(logging, level.upper()) if isinstance(level, str) else level
        
        if logger_name:
            logging.getLogger(logger_name).setLevel(level_int)
        else:
            logging.getLogger().setLevel(level_int)
    
    def add_file_handler(
        self,
        log_file: Path,
        logger_name: Optional[str] = None,
        level: Optional[str] = None
    ):
        """
        Add additional file handler.
        
        Args:
            log_file: Log file path
            logger_name: Logger to add handler to
            level: Handler log level
        """
        handler = self._create_file_handler(Path(log_file), json_logs=False)
        
        if level:
            handler.setLevel(getattr(logging, level.upper()))
        
        logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        logger.addHandler(handler)
    
    def shutdown(self):
        """Shutdown logging system."""
        logging.shutdown()
        self.configured = False

# Global logger manager instance
_logger_manager = LoggerManager()

@lru_cache(maxsize=None)
def setup_logging(
    name: Optional[str] = None,
    log_level: str = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    log_file: Optional[str] = None,
    format_style: str = "standard",
    use_colors: bool = True,
    json_logs: bool = False,
    experiment_id: Optional[str] = None,
) -> logging.Logger:
    """
    Setup and get a logger.
    
    Args:
        name: Logger name
        log_level: Log level
        log_dir: Directory for log files
        log_file: Log file name
        format_style: Format style
        use_colors: Whether to use colors
        json_logs: Whether to use JSON format
        experiment_id: Experiment ID
        
    Returns:
        Configured logger
    """
    # Setup global logging if not configured
    if not _logger_manager.configured:
        # Get from environment if not provided
        if log_level == "INFO":
            log_level = os.getenv("LOG_LEVEL", "INFO")
        
        if log_dir is None:
            log_dir = os.getenv("LOG_DIR", "outputs/logs")
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"agnews_{timestamp}.log"
        
        _logger_manager.setup(
            log_level=log_level,
            log_dir=Path(log_dir) if log_dir else None,
            log_file=log_file,
            format_style=format_style,
            use_colors=use_colors,
            json_logs=json_logs,
            experiment_id=experiment_id,
        )
    
    # Return logger
    return _logger_manager.get_logger(name or __name__)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return _logger_manager.get_logger(name)

def set_verbosity(level: str):
    """
    Set global verbosity level.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    _logger_manager.set_level(level)

def enable_progress_bars():
    """Enable progress bar logging."""
    logging.getLogger("tqdm").setLevel(logging.INFO)

def disable_progress_bars():
    """Disable progress bar logging."""
    logging.getLogger("tqdm").setLevel(logging.ERROR)

def log_system_info(logger: Optional[logging.Logger] = None):
    """
    Log system information.
    
    Args:
        logger: Logger to use (default: root)
    """
    if logger is None:
        logger = logging.getLogger()
    
    import platform
    import torch
    import transformers
    
    logger.info("=" * 80)
    logger.info("System Information")
    logger.info("=" * 80)
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Transformers: {transformers.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
    else:
        logger.info("CUDA: Not available")
    
    logger.info("=" * 80)

# Export public API
__all__ = [
    "setup_logging",
    "get_logger",
    "set_verbosity",
    "enable_progress_bars",
    "disable_progress_bars",
    "log_system_info",
    "MetricsLogger",
    "JSONFormatter",
    "ExperimentLogFormatter",
    "LoggerManager",
]
