"""
Logging Configuration for AG News Text Classification Framework.

This module implements a comprehensive logging system following best practices
from software engineering and distributed systems literature.

Theoretical Foundation:
The logging architecture is based on principles from:
- Lampson, B. W. (1983). "Hints for computer system design". ACM SIGOPS 
  Operating Systems Review, 17(5), 33-48.
- Xu, W., et al. (2009). "Detecting large-scale system problems by mining 
  console logs". In Proceedings of SOSP '09, ACM, 117-132.
- Yuan, D., et al. (2012). "Improving software diagnosability via log 
  enhancement". ACM Transactions on Computer Systems, 30(1), 1-28.

Design Principles:
1. Structured Logging: Following Graylog Extended Log Format (GELF) for 
   machine-readable logs enabling automated analysis (Xu et al., 2009).
2. Hierarchical Organization: Logger hierarchy based on module structure 
   following Log4j patterns (Gülcü, C., 2003. "The Complete Log4j Manual").
3. Performance Optimization: Lazy evaluation and buffering to minimize 
   logging overhead (Yuan et al., 2012).

Information Theory:
The log level selection follows information-theoretic principles where:
- H(log) = -Σ p(level) × log₂(p(level))
- Optimal log verbosity maximizes information while minimizing noise

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
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

# Suppress warnings from external libraries following best practices
# from Beazley, D. (2009). "Python Essential Reference" (4th ed.)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Log format templates following syslog RFC 5424 and structured logging principles
FORMATS = {
    "simple": "%(levelname)s - %(message)s",
    "standard": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "detailed": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s",
    "json": '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "file": "%(filename)s", "line": %(lineno)d}',
    "colored": "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s - %(message)s",
    "research": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
}

# Color scheme based on human perception studies
# Reference: Ware, C. (2012). "Information Visualization: Perception for Design"
COLOR_SCHEME = {
    "DEBUG": "cyan",      # Low importance, cool color
    "INFO": "green",      # Normal operation, positive association
    "WARNING": "yellow",  # Attention needed, caution color
    "ERROR": "red",       # Error state, danger color
    "CRITICAL": "red,bg_white",  # Critical state, maximum contrast
}

# Component-specific log levels following principle of least privilege
# Reference: Saltzer, J. H., & Schroeder, M. D. (1975). "The protection of 
# information in computer systems". Proceedings of the IEEE, 63(9), 1278-1308.
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
    """
    Custom JSON formatter for structured logging.
    
    Implements structured logging following principles from:
    - Chuvakin, A., et al. (2012). "Logging and Log Management: The Authoritative 
      Guide to Understanding the Concepts Surrounding Logging and Log Management".
    
    The JSON format enables:
    1. Machine-readable logs for automated analysis
    2. Consistent structure for log aggregation
    3. Rich contextual information preservation
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON with comprehensive metadata.
        
        The output follows Elastic Common Schema (ECS) principles for
        compatibility with log analysis tools.
        
        Args:
            record: LogRecord instance containing log information
            
        Returns:
            JSON-formatted log string
            
        Time Complexity: O(n) where n = number of extra fields
        Space Complexity: O(m) where m = size of log message
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),  # ISO 8601 format
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        
        # Add exception information if present (error tracking)
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields (extensibility principle)
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", "funcName",
                          "levelname", "levelno", "lineno", "module", "msecs",
                          "message", "pathname", "process", "processName", "relativeCreated",
                          "thread", "threadName", "exc_info", "exc_text", "stack_info"]:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)

class ExperimentLogFormatter(logging.Formatter):
    """
    Custom formatter for machine learning experiment tracking.
    
    Based on experimental methodology from:
    - Sculley, D., et al. (2015). "Hidden technical debt in machine learning 
      systems". In Advances in Neural Information Processing Systems.
    
    Adds experimental context (epoch, step, experiment_id) to facilitate
    reproducibility and debugging of ML experiments.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with ML experiment context.
        
        Implements contextual logging for experiment tracking following
        MLOps best practices.
        
        Args:
            record: LogRecord with potential experiment metadata
            
        Returns:
            Formatted log string with experiment context
        """
        # Add experiment identifier for traceability
        if hasattr(record, "experiment_id"):
            record.msg = f"[EXP:{record.experiment_id}] {record.msg}"
        
        # Add training epoch for temporal context
        if hasattr(record, "epoch"):
            record.msg = f"[Epoch {record.epoch}] {record.msg}"
        
        # Add training step for fine-grained tracking
        if hasattr(record, "step"):
            record.msg = f"[Step {record.step}] {record.msg}"
        
        return super().format(record)

class ProgressLogFilter(logging.Filter):
    """
    Filter to handle progress bar logs from tqdm.
    
    Implements selective filtering to prevent progress bar interference
    with structured logs, following UI/UX principles from:
    - Nielsen, J. (1993). "Usability Engineering". Academic Press.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter out progress bar updates to maintain log clarity.
        
        Args:
            record: Log record to filter
            
        Returns:
            False for progress updates, True otherwise
        """
        # Skip tqdm progress updates (contains iteration metrics)
        if "it/s" in record.getMessage() or "s/it" in record.getMessage():
            return False
        return True

class MetricsLogger:
    """
    Specialized logger for ML metrics tracking.
    
    Implements metrics logging following principles from:
    - Bengio, Y. (2012). "Practical recommendations for gradient-based 
      training of deep architectures". In Neural Networks: Tricks of the Trade.
    
    Provides structured tracking of training metrics with temporal context.
    """
    
    def __init__(self, name: str = "metrics"):
        """
        Initialize metrics logger with history tracking.
        
        Args:
            name: Logger name for hierarchical organization
        """
        self.logger = logging.getLogger(f"metrics.{name}")
        self.metrics_history = []  # Time-series storage
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log metrics with temporal and contextual information.
        
        Implements structured metrics logging for experiment tracking
        and analysis.
        
        Args:
            metrics: Dictionary of metric name-value pairs
            step: Training step (iteration number)
            epoch: Training epoch
            prefix: Metric name prefix for namespacing
            
        Mathematical Note:
            Metrics are stored as time series: M(t) = {m₁(t), m₂(t), ..., mₙ(t)}
            where t represents the temporal index (step/epoch)
        """
        # Format metrics for human readability
        metric_strs = []
        for key, value in metrics.items():
            if isinstance(value, float):
                metric_strs.append(f"{prefix}{key}: {value:.4f}")
            else:
                metric_strs.append(f"{prefix}{key}: {value}")
        
        message = " | ".join(metric_strs)
        
        # Add temporal context
        extra = {}
        if step is not None:
            extra["step"] = step
        if epoch is not None:
            extra["epoch"] = epoch
        
        # Log and store in history
        self.logger.info(message, extra=extra)
        self.metrics_history.append({
            "metrics": metrics,
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now(),
        })
    
    def get_history(self) -> list:
        """
        Retrieve metrics history for analysis.
        
        Returns:
            List of metric records with temporal information
        """
        return self.metrics_history

class LoggerManager:
    """
    Centralized logger configuration manager.
    
    Implements the Singleton pattern for global logger management following:
    - Gamma, E., et al. (1994). "Design Patterns: Elements of Reusable 
      Object-Oriented Software". Addison-Wesley.
    
    Provides unified configuration and management of the logging subsystem.
    """
    
    def __init__(self):
        """Initialize logger manager with empty state."""
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
        Setup comprehensive logging configuration.
        
        Implements logging setup following best practices from:
        - The Twelve-Factor App methodology (Wiggins, A., 2012)
        - OWASP Logging Cheat Sheet
        
        Args:
            log_level: Minimum log level threshold
            log_dir: Directory for log file storage
            log_file: Log file name
            format_style: Log format template selection
            use_colors: Enable colored console output
            json_logs: Use JSON structured logging
            experiment_id: Experiment identifier for tracking
            
        Configuration Principle:
            Follows separation of concerns with distinct handlers for
            console (human-readable) and file (machine-readable) output.
        """
        if self.configured:
            return
        
        # Create log directory with proper permissions
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger (top of hierarchy)
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers (idempotency)
        root_logger.handlers.clear()
        
        # Console handler for human interaction
        console_handler = self._create_console_handler(
            format_style, use_colors, json_logs
        )
        root_logger.addHandler(console_handler)
        self.handlers["console"] = console_handler
        
        # File handler for persistence
        if log_dir and log_file:
            file_handler = self._create_file_handler(
                log_dir / log_file, json_logs
            )
            root_logger.addHandler(file_handler)
            self.handlers["file"] = file_handler
        
        # Rotating file handler for long-running processes
        if log_dir:
            rotating_handler = self._create_rotating_handler(
                log_dir / "experiment.log", json_logs
            )
            root_logger.addHandler(rotating_handler)
            self.handlers["rotating"] = rotating_handler
        
        # Apply component-specific levels (principle of least privilege)
        for component, level in COMPONENT_LEVELS.items():
            logging.getLogger(component).setLevel(level)
        
        # Add experiment context if provided
        if experiment_id:
            logging.LoggerAdapter(root_logger, {"experiment_id": experiment_id})
        
        self.configured = True
        
        # Log configuration completion
        root_logger.info(f"Logging configured - Level: {log_level}, Format: {format_style}")
    
    def _create_console_handler(
        self,
        format_style: str,
        use_colors: bool,
        json_logs: bool
    ) -> logging.Handler:
        """
        Create console handler with appropriate formatting.
        
        Implements console output following human-computer interaction
        principles for optimal readability.
        
        Args:
            format_style: Format template name
            use_colors: Enable ANSI color codes
            json_logs: Use JSON formatting
            
        Returns:
            Configured console handler
        """
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
        """
        Create file handler for persistent logging.
        
        Implements file-based logging with proper encoding and formatting
        for long-term storage and analysis.
        
        Args:
            log_file: Path to log file
            json_logs: Use JSON formatting
            
        Returns:
            Configured file handler
        """
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
        """
        Create rotating file handler for log rotation.
        
        Implements log rotation following syslog principles to prevent
        unbounded disk usage.
        
        Args:
            log_file: Base log file path
            json_logs: Use JSON formatting
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep
            
        Returns:
            Configured rotating file handler
            
        Storage Analysis:
            Maximum disk usage = max_bytes × (backup_count + 1)
            With defaults: 100MB × 11 = 1.1GB maximum
        """
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
        Get or create a logger instance.
        
        Implements lazy initialization pattern for efficient resource usage.
        
        Args:
            name: Logger name for hierarchical organization
            
        Returns:
            Logger instance
        """
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
        return self.loggers[name]
    
    def set_level(self, level: Union[str, int], logger_name: Optional[str] = None):
        """
        Dynamically adjust log level.
        
        Enables runtime log level adjustment for debugging and monitoring.
        
        Args:
            level: New log level
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
        Add additional file handler for specialized logging.
        
        Enables creation of separate log files for different components
        or purposes.
        
        Args:
            log_file: Log file path
            logger_name: Target logger name
            level: Handler-specific log level
        """
        handler = self._create_file_handler(Path(log_file), json_logs=False)
        
        if level:
            handler.setLevel(getattr(logging, level.upper()))
        
        logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        logger.addHandler(handler)
    
    def shutdown(self):
        """
        Gracefully shutdown logging system.
        
        Ensures all buffered logs are flushed and resources are released.
        """
        logging.shutdown()
        self.configured = False

# Global logger manager instance (Singleton pattern)
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
    Setup and retrieve a configured logger.
    
    Implements memoized logger configuration for efficiency using
    LRU cache decorator.
    
    Args:
        name: Logger name
        log_level: Minimum log level
        log_dir: Log directory path
        log_file: Log file name
        format_style: Format template name
        use_colors: Enable colored output
        json_logs: Use JSON formatting
        experiment_id: Experiment identifier
        
    Returns:
        Configured logger instance
        
    Design Pattern:
        Uses memoization (dynamic programming) to avoid redundant
        configuration overhead.
    """
    # Initialize global logging if not configured
    if not _logger_manager.configured:
        # Get configuration from environment (12-factor app)
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
    
    # Return logger instance
    return _logger_manager.get_logger(name or __name__)

def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a logger by name.
    
    Simple accessor following the Facade pattern for ease of use.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return _logger_manager.get_logger(name)

def set_verbosity(level: str):
    """
    Set global verbosity level.
    
    Provides simple interface for adjusting log verbosity.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    _logger_manager.set_level(level)

def enable_progress_bars():
    """
    Enable progress bar logging.
    
    Adjusts tqdm logger level for progress visibility.
    """
    logging.getLogger("tqdm").setLevel(logging.INFO)

def disable_progress_bars():
    """
    Disable progress bar logging.
    
    Suppresses tqdm output for cleaner logs.
    """
    logging.getLogger("tqdm").setLevel(logging.ERROR)

def log_system_info(logger: Optional[logging.Logger] = None):
    """
    Log comprehensive system information.
    
    Implements system profiling for reproducibility and debugging
    following practices from:
    - Stodden, V., et al. (2016). "Enhancing reproducibility for 
      computational methods". Science, 354(6317), 1240-1241.
    
    Args:
        logger: Logger instance (uses root if None)
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
