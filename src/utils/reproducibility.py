"""
Reproducibility utilities for AG News Text Classification Framework.

Ensures reproducible results across different runs and environments.
"""

import os
import random
import logging
from typing import Optional, Dict, Any, List
import json
from pathlib import Path
from datetime import datetime
import hashlib
import platform

import numpy as np
import torch
import torch.backends.cudnn as cudnn

logger = logging.getLogger(__name__)

class ReproducibilityManager:
    """
    Manager for ensuring reproducible experiments.
    
    Handles random seed setting, environment configuration, and
    tracking of experimental conditions.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize reproducibility manager.
        
        Args:
            seed: Random seed
        """
        self.seed = seed
        self.config = {}
        self.checksums = {}
        self.environment_snapshot = None
    
    def set_seed(self, seed: Optional[int] = None):
        """
        Set random seed for all libraries.
        
        Args:
            seed: Random seed (uses self.seed if None)
        """
        if seed is None:
            seed = self.seed
        else:
            self.seed = seed
        
        # Python random
        random.seed(seed)
        
        # Numpy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU
        
        # Environment variables
        os.environ["PYTHONHASHSEED"] = str(seed)
        
        # Additional seeds for other libraries
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
        except ImportError:
            pass
        
        logger.info(f"Random seed set to {seed}")
    
    def set_deterministic_mode(self, deterministic: bool = True):
        """
        Enable/disable deterministic mode.
        
        Args:
            deterministic: Whether to enable deterministic mode
        """
        if deterministic:
            # PyTorch deterministic operations
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Environment variable for CUBLAS
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            
            logger.info("Deterministic mode enabled")
        else:
            torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            
            logger.info("Deterministic mode disabled (better performance)")
    
    def configure_for_reproducibility(
        self,
        seed: Optional[int] = None,
        deterministic: bool = True,
        disable_warnings: bool = True
    ):
        """
        Configure environment for reproducibility.
        
        Args:
            seed: Random seed
            deterministic: Whether to use deterministic algorithms
            disable_warnings: Whether to disable non-deterministic warnings
        """
        # Set seed
        self.set_seed(seed)
        
        # Set deterministic mode
        self.set_deterministic_mode(deterministic)
        
        # Disable warnings if requested
        if disable_warnings:
            import warnings
            warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
        
        # Store configuration
        self.config = {
            "seed": self.seed,
            "deterministic": deterministic,
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info("Environment configured for reproducibility")
    
    def capture_environment(self) -> Dict[str, Any]:
        """
        Capture current environment state.
        
        Returns:
            Dictionary with environment information
        """
        import sys
        import torch
        import transformers
        
        env_info = {
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "processor": platform.processor(),
            },
            "packages": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "cuda": torch.version.cuda if torch.cuda.is_available() else None,
                "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            },
            "hardware": {
                "cpu_count": os.cpu_count(),
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            },
            "environment_variables": {
                k: v for k, v in os.environ.items()
                if any(pattern in k for pattern in [
                    "CUDA", "TORCH", "PYTHON", "SEED", "OMP", "MKL"
                ])
            },
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            env_info["hardware"]["gpus"] = []
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                }
                env_info["hardware"]["gpus"].append(gpu_info)
        
        self.environment_snapshot = env_info
        return env_info
    
    def save_environment(self, save_path: Path):
        """
        Save environment snapshot to file.
        
        Args:
            save_path: Path to save environment info
        """
        if self.environment_snapshot is None:
            self.capture_environment()
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w") as f:
            json.dump(self.environment_snapshot, f, indent=2, default=str)
        
        logger.info(f"Environment snapshot saved to {save_path}")
    
    def compute_data_checksum(self, data_path: Path) -> str:
        """
        Compute checksum for data file.
        
        Args:
            data_path: Path to data file
            
        Returns:
            Checksum string
        """
        hasher = hashlib.sha256()
        
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        
        checksum = hasher.hexdigest()
        self.checksums[str(data_path)] = checksum
        
        return checksum
    
    def verify_data_integrity(self, data_path: Path, expected_checksum: str) -> bool:
        """
        Verify data file integrity.
        
        Args:
            data_path: Path to data file
            expected_checksum: Expected checksum
            
        Returns:
            True if checksum matches
        """
        actual_checksum = self.compute_data_checksum(data_path)
        
        if actual_checksum != expected_checksum:
            logger.warning(
                f"Data integrity check failed for {data_path}. "
                f"Expected: {expected_checksum}, Got: {actual_checksum}"
            )
            return False
        
        logger.info(f"Data integrity verified for {data_path}")
        return True
    
    def create_experiment_hash(self, config: Dict[str, Any]) -> str:
        """
        Create unique hash for experiment configuration.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment hash
        """
        # Create deterministic string representation
        config_str = json.dumps(config, sort_keys=True, default=str)
        
        # Generate hash
        hasher = hashlib.sha256()
        hasher.update(config_str.encode())
        
        return hasher.hexdigest()[:16]  # Use first 16 characters
    
    def log_reproducibility_info(self):
        """Log reproducibility information."""
        logger.info("=" * 80)
        logger.info("Reproducibility Information")
        logger.info("=" * 80)
        logger.info(f"Random seed: {self.seed}")
        logger.info(f"Deterministic mode: {torch.are_deterministic_algorithms_enabled()}")
        logger.info(f"CUDNN deterministic: {cudnn.deterministic}")
        logger.info(f"CUDNN benchmark: {cudnn.benchmark}")
        logger.info(f"Python hash seed: {os.environ.get('PYTHONHASHSEED', 'not set')}")
        logger.info("=" * 80)

class RandomStateManager:
    """
    Context manager for temporary random state changes.
    
    Useful for ensuring specific operations use a fixed seed without
    affecting the global random state.
    """
    
    def __init__(self, seed: int):
        """
        Initialize random state manager.
        
        Args:
            seed: Random seed to use
        """
        self.seed = seed
        self.python_state = None
        self.numpy_state = None
        self.torch_state = None
        self.torch_cuda_state = None
    
    def __enter__(self):
        """Enter context and save current state."""
        # Save current states
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        
        if torch.cuda.is_available():
            self.torch_cuda_state = torch.cuda.get_rng_state_all()
        
        # Set new seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous state."""
        # Restore previous states
        random.setstate(self.python_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)
        
        if torch.cuda.is_available() and self.torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(self.torch_cuda_state)

def set_global_seed(seed: int = 42):
    """
    Set global random seed for all libraries.
    
    Args:
        seed: Random seed
    """
    manager = ReproducibilityManager(seed)
    manager.set_seed()

def ensure_reproducibility(seed: int = 42, deterministic: bool = True):
    """
    Ensure reproducibility for experiments.
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms
    """
    manager = ReproducibilityManager(seed)
    manager.configure_for_reproducibility(deterministic=deterministic)
    manager.log_reproducibility_info()
    return manager

def worker_init_fn(worker_id: int):
    """
    Worker initialization function for DataLoader.
    
    Ensures each worker has a different but deterministic seed.
    
    Args:
        worker_id: Worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_reproducible_dataloader_kwargs(seed: int = 42) -> Dict[str, Any]:
    """
    Get DataLoader kwargs for reproducibility.
    
    Args:
        seed: Base random seed
        
    Returns:
        DataLoader kwargs
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return {
        "worker_init_fn": worker_init_fn,
        "generator": generator,
    }

def create_reproducibility_report(
    experiment_config: Dict[str, Any],
    results: Dict[str, Any],
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Create reproducibility report for experiment.
    
    Args:
        experiment_config: Experiment configuration
        results: Experiment results
        save_path: Optional path to save report
        
    Returns:
        Reproducibility report
    """
    manager = ReproducibilityManager()
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "experiment_config": experiment_config,
        "results": results,
        "environment": manager.capture_environment(),
        "checksums": manager.checksums,
        "experiment_hash": manager.create_experiment_hash(experiment_config),
    }
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Reproducibility report saved to {save_path}")
    
    return report

# Export public API
__all__ = [
    "ReproducibilityManager",
    "RandomStateManager",
    "set_global_seed",
    "ensure_reproducibility",
    "worker_init_fn",
    "get_reproducible_dataloader_kwargs",
    "create_reproducibility_report",
]
