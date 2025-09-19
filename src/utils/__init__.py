"""
Utility modules for AG News Text Classification Framework.

This package provides essential utilities for:
- I/O operations and file handling
- Logging configuration and management
- Reproducibility and random seed control
- Distributed training support
- Memory optimization and monitoring
- Performance profiling and benchmarking
- Experiment tracking and management
- Prompt engineering and template management

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
"""

# Import core utilities
from src.utils.io_utils import (
    ensure_dir,
    safe_save,
    safe_load,
    download_file,
    extract_archive,
    compute_file_hash,
    get_file_size,
    temp_directory,
    atomic_write,
    find_files,
    copy_tree,
    cleanup_directory,
)

from src.utils.logging_config import (
    setup_logging,
    get_logger,
    set_verbosity,
    enable_progress_bars,
    disable_progress_bars,
    log_system_info,
    MetricsLogger,
    JSONFormatter,
    ExperimentLogFormatter,
    LoggerManager,
)

from src.utils.reproducibility import (
    ReproducibilityManager,
    RandomStateManager,
    set_global_seed,
    ensure_reproducibility,
    worker_init_fn,
    get_reproducible_dataloader_kwargs,
    create_reproducibility_report,
)

# Import distributed utilities
from src.utils.distributed_utils import (
    DistributedManager,
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    all_reduce,
    all_gather,
    broadcast,
)

# Import memory utilities
from src.utils.memory_utils import (
    MemoryMonitor,
    get_memory_usage,
    log_memory_usage,
    optimize_memory,
    clear_memory,
    estimate_model_memory,
    profile_memory_usage,
    enable_memory_efficient_mode,
)

# Import profiling utilities
from src.utils.profiling_utils import (
    Profiler,
    Timer,
    profile_function,
    profile_model,
    benchmark_function,
    log_profiling_results,
    create_profiling_report,
)

# Import experiment tracking
from src.utils.experiment_tracking import (
    ExperimentTracker,
    log_hyperparameters,
    log_metrics,
    log_artifacts,
    log_model,
    create_experiment,
    get_experiment_id,
    compare_experiments,
)

# Import prompt utilities
from src.utils.prompt_utils import (
    PromptTemplate,
    PromptManager,
    create_prompt,
    format_prompt,
    load_prompt_template,
    save_prompt_template,
    validate_prompt,
    optimize_prompt_length,
)

# Module metadata
__all__ = [
    # I/O utilities
    "ensure_dir",
    "safe_save",
    "safe_load",
    "download_file",
    "extract_archive",
    "compute_file_hash",
    "get_file_size",
    "temp_directory",
    "atomic_write",
    "find_files",
    "copy_tree",
    "cleanup_directory",
    # Logging utilities
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
    # Reproducibility utilities
    "ReproducibilityManager",
    "RandomStateManager",
    "set_global_seed",
    "ensure_reproducibility",
    "worker_init_fn",
    "get_reproducible_dataloader_kwargs",
    "create_reproducibility_report",
    # Distributed utilities
    "DistributedManager",
    "setup_distributed",
    "cleanup_distributed",
    "is_distributed",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "barrier",
    "all_reduce",
    "all_gather",
    "broadcast",
    # Memory utilities
    "MemoryMonitor",
    "get_memory_usage",
    "log_memory_usage",
    "optimize_memory",
    "clear_memory",
    "estimate_model_memory",
    "profile_memory_usage",
    "enable_memory_efficient_mode",
    # Profiling utilities
    "Profiler",
    "Timer",
    "profile_function",
    "profile_model",
    "benchmark_function",
    "log_profiling_results",
    "create_profiling_report",
    # Experiment tracking
    "ExperimentTracker",
    "log_hyperparameters",
    "log_metrics",
    "log_artifacts",
    "log_model",
    "create_experiment",
    "get_experiment_id",
    "compare_experiments",
    # Prompt utilities
    "PromptTemplate",
    "PromptManager",
    "create_prompt",
    "format_prompt",
    "load_prompt_template",
    "save_prompt_template",
    "validate_prompt",
    "optimize_prompt_length",
]

# Initialize default logger for utils module
logger = get_logger(__name__)
logger.debug("Utils module initialized")
