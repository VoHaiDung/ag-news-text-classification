import os
import sys
import json
import shutil
import subprocess
import asyncio
import pickle
import hashlib
import logging
import platform
import tempfile
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from setuptools import setup, find_packages, Command
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.test import test as TestCommand

# Try importing optional dependencies
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    
try:
    from packaging.requirements import Requirement
    from packaging.version import Version
    HAS_PACKAGING = True
except ImportError:
    HAS_PACKAGING = False

# Project directory structure
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
CACHE_DIR = ROOT_DIR / ".cache" / "setup"
CONFIG_DIR = ROOT_DIR / "configs"
LOG_DIR = ROOT_DIR / "outputs" / "logs" / "setup"
BACKUP_DIR = ROOT_DIR / ".backups"

# Create necessary directories
for dir_path in [CACHE_DIR, LOG_DIR, BACKUP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"setup_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("setup")

# Custom exception for setup-specific errors
class SetupError(Exception):
    """Custom exception for setup errors"""
    pass

# Enum for different installation modes
class InstallMode(Enum):
    MINIMAL = "minimal"
    BASE = "base"
    RESEARCH = "research"
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    ALL = "all"

# Data class for environment check configuration
@dataclass
class EnvironmentCheck:
    name: str
    check_func: Callable
    required: bool = True
    min_version: Optional[str] = None

# Class to track setup performance metrics
class SetupMetrics:
    """Track setup metrics for optimization"""
    
    def __init__(self):
        self.start_time = time.time()
        self.operations = []
        self.cache_hits = 0
        self.cache_misses = 0
        
    def record_operation(self, name: str, duration: float):
        """Record an operation's duration"""
        self.operations.append({"name": name, "duration": duration})
        
    def record_cache_hit(self):
        """Increment cache hit counter"""
        self.cache_hits += 1
        
    def record_cache_miss(self):
        """Increment cache miss counter"""
        self.cache_misses += 1
        
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        total_time = time.time() - self.start_time
        return {
            "total_time": total_time,
            "operations": self.operations,
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            "avg_operation_time": sum(op["duration"] for op in self.operations) / max(len(self.operations), 1)
        }

# Initialize global metrics tracker
metrics = SetupMetrics()

# Context manager for timing operations
@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations"""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        metrics.record_operation(operation_name, duration)
        logger.debug(f"{operation_name} took {duration:.2f}s")

# Decorator for retrying failed operations
def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return None
        return wrapper
    return decorator

# Load shared configuration between setup.py and Makefile
def get_shared_config() -> Dict[str, Any]:
    """Load shared configuration used by both setup.py and Makefile"""
    config_file = CONFIG_DIR / "build_config.yaml"
    
    # Return existing config if available
    if config_file.exists():
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
    
    # Default configuration
    default_config = {
        "project_name": "ag-news-text-classification",
        "python_version": "3.10",
        "cuda_version": "11.8",
        "min_memory_gb": 16,
        "min_disk_gb": 50,
        "parallel_jobs": os.cpu_count() or 4,
        "docker_registry": "agnews-research",
        "cloud_provider": "aws",
        "week_schedule": {
            "week1": "Classical ML Baselines",
            "week2-3": "Deep Learning & Transformers",
            "week4-5": "Advanced Training Strategies",
            "week6-7": "SOTA Models & LLMs",
            "week8-9": "Optimization & Compression",
            "week10": "Production Deployment"
        }
    }
    
    # Save default config for future use
    with open(config_file, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    return default_config

# Load shared configuration
SHARED_CONFIG = get_shared_config()

# Project metadata
METADATA = {
    "name": SHARED_CONFIG["project_name"],
    "author": "AG News Research Team",
    "author_email": "team@agnews-research.ai",
    "maintainer": "AG News ML Engineering Team",
    "maintainer_email": "ml-team@agnews-research.ai",
    "license": "MIT",
    "url": "https://github.com/agnews-research/ag-news-text-classification",
}

# Get version from __version__.py file or git tags
def get_version() -> str:
    """Get version from __version__.py file or git tags"""
    with timer("get_version"):
        version_file = SRC_DIR / "__version__.py"
        version_dict = {}
        
        # Try reading from version file
        if version_file.exists():
            with open(version_file, "r", encoding="utf-8") as f:
                exec(f.read(), version_dict)
            return version_dict.get("__version__", "1.0.0")
        
        # Fallback to git tags
        try:
            version = subprocess.check_output(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=ROOT_DIR,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return version.lstrip("v")
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "1.0.0"

# Get project version
VERSION = get_version()

# Read README for long description
def get_long_description() -> str:
    """Read and return README content"""
    readme_file = ROOT_DIR / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return "AG News Text Classification Framework - State-of-the-art research framework"

# Calculate file hash for cache validation
def get_file_hash(filepath: Path) -> str:
    """Get hash of file for cache validation"""
    if not filepath.exists():
        return ""
    
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        # Read file in chunks for memory efficiency
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# Class to resolve and validate dependencies
class DependencyResolver:
    """Resolve and validate dependencies"""
    
    def __init__(self):
        self.conflicts = []
        self.all_requirements = {}
        
    def add_requirement(self, req_str: str):
        """Add a requirement and check for conflicts"""
        if not HAS_PACKAGING:
            return
            
        try:
            req = Requirement(req_str)
            # Check for version conflicts with existing requirements
            if req.name in self.all_requirements:
                existing = self.all_requirements[req.name]
                if not self._compatible_versions(existing, req):
                    self.conflicts.append((req.name, existing, req))
            self.all_requirements[req.name] = req
        except Exception as e:
            logger.warning(f"Could not parse requirement: {req_str} - {e}")
    
    def _compatible_versions(self, req1: 'Requirement', req2: 'Requirement') -> bool:
        """Check if two requirements are compatible"""
        return str(req1.specifier) == str(req2.specifier)
    
    def resolve(self) -> bool:
        """Resolve dependencies and return True if successful"""
        if self.conflicts:
            logger.warning("Dependency conflicts detected:")
            for name, req1, req2 in self.conflicts:
                logger.warning(f"  {name}: {req1} vs {req2}")
            return False
        return True

# Initialize global dependency resolver
dependency_resolver = DependencyResolver()

# Cached requirements reader with hash validation
@lru_cache(maxsize=32)
def read_requirements_cached(filename: str, base_dir: Path = REQUIREMENTS_DIR) -> List[str]:
    """Read requirements from file with caching support"""
    with timer(f"read_requirements_{filename}"):
        filepath = base_dir / filename
        cache_file = CACHE_DIR / f"{filename}.cache"
        hash_file = CACHE_DIR / f"{filename}.hash"
        
        # Check if cache is valid
        current_hash = get_file_hash(filepath)
        if cache_file.exists() and hash_file.exists():
            try:
                with open(hash_file, "r") as f:
                    cached_hash = f.read().strip()
                
                # Use cache if hash matches
                if cached_hash == current_hash:
                    metrics.record_cache_hit()
                    with open(cache_file, "rb") as f:
                        return pickle.load(f)
            except Exception:
                pass  # Cache invalid, will regenerate
        
        # Cache miss - read requirements from file
        metrics.record_cache_miss()
        requirements = read_requirements(filename, base_dir)
        
        # Update cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(requirements, f)
            with open(hash_file, "w") as f:
                f.write(current_hash)
        except Exception:
            pass  # Cache write failed, continue without caching
        
        # Add requirements to dependency resolver
        for req in requirements:
            dependency_resolver.add_requirement(req)
        
        return requirements

# Read requirements from file, handling includes
def read_requirements(filename: str, base_dir: Path = REQUIREMENTS_DIR) -> List[str]:
    """Read requirements from file, handling includes and various formats"""
    requirements = []
    seen_files = set()  # Prevent circular imports
    
    def _read_file(filepath: Path, seen: Set[Path]) -> List[str]:
        """Recursively read requirements file"""
        # Check for circular imports
        if filepath in seen:
            return []
        seen.add(filepath)
        
        # Check file exists
        if not filepath.exists():
            logger.warning(f"{filepath} not found")
            return []
        
        file_requirements = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                
                # Handle -r includes for requirement files
                if line.startswith("-r "):
                    included_file = line[3:].strip()
                    included_path = filepath.parent / included_file
                    file_requirements.extend(_read_file(included_path, seen))
                    continue
                
                # Handle git+ requirements
                if "git+" in line:
                    if "#egg=" in line:
                        pkg_name = line.split("#egg=")[-1]
                        file_requirements.append(f"{pkg_name} @ {line}")
                    else:
                        # Extract package name from git URL
                        pkg_name = line.split("/")[-1].replace(".git", "")
                        file_requirements.append(f"{pkg_name} @ {line}")
                else:
                    file_requirements.append(line)
        
        return file_requirements
    
    # Start reading from main file
    filepath = base_dir / filename
    requirements = _read_file(filepath, seen_files)
    
    # Remove duplicates while preserving order
    seen_packages = set()
    unique_requirements = []
    for req in requirements:
        # Extract package name for deduplication
        req_name = req.split(">=")[0].split("==")[0].split("<")[0].split("@")[0].strip()
        req_name = req_name.split("[")[0]  # Handle extras like package[extra]
        if req_name not in seen_packages:
            seen_packages.add(req_name)
            unique_requirements.append(req)
    
    return unique_requirements

# Run subprocess asynchronously
async def run_subprocess_async(cmd: List[str], **kwargs) -> Tuple[int, str, str]:
    """Run subprocess asynchronously"""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode()

# Run multiple commands asynchronously
def run_async_commands(commands: List[List[str]]) -> List[Tuple[int, str, str]]:
    """Run multiple commands asynchronously"""
    async def run_all():
        tasks = [run_subprocess_async(cmd) for cmd in commands]
        return await asyncio.gather(*tasks)
    
    # Set event loop policy for Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    return asyncio.run(run_all())

# Safe subprocess call with error handling
def safe_subprocess_call(cmd: List[str], error_msg: Optional[str] = None) -> Optional[int]:
    """Wrapper for subprocess calls with better error handling"""
    try:
        result = subprocess.check_call(cmd, stderr=subprocess.PIPE)
        return result
    except subprocess.CalledProcessError as e:
        if error_msg:
            logger.error(f"{error_msg}: {e.stderr.decode() if e.stderr else str(e)}")
        return None
    except FileNotFoundError:
        logger.error(f"Command not found: {cmd[0]}")
        return None

# Check if command exists in PATH
def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH"""
    return shutil.which(command) is not None

# Validate Python version
def check_python_version() -> bool:
    """Check if Python version meets requirements"""
    required = tuple(map(int, SHARED_CONFIG["python_version"].split(".")))
    current = sys.version_info[:2]
    return current >= required[:2]

# Check available disk space
def check_disk_space(min_gb: int = None) -> bool:
    """Check available disk space"""
    min_gb = min_gb or SHARED_CONFIG["min_disk_gb"]
    stat = shutil.disk_usage(ROOT_DIR)
    available_gb = stat.free / (1024**3)
    return available_gb >= min_gb

# Check available memory
def check_memory() -> bool:
    """Check available memory"""
    try:
        import psutil
        min_gb = SHARED_CONFIG["min_memory_gb"]
        available_gb = psutil.virtual_memory().available / (1024**3)
        return available_gb >= min_gb
    except ImportError:
        return True  # Assume OK if psutil not available

# Check GPU availability
def check_gpu() -> bool:
    """Check GPU availability"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# Validate environment before setup
def validate_environment() -> bool:
    """Validate environment before setup"""
    # Define environment checks
    checks = [
        EnvironmentCheck("Python version", check_python_version),
        EnvironmentCheck("Disk space", check_disk_space),
        EnvironmentCheck("Memory", check_memory),
        EnvironmentCheck("Git", lambda: check_command_exists("git")),
        EnvironmentCheck("pip", lambda: check_command_exists("pip")),
        EnvironmentCheck("GPU", check_gpu, required=False),
    ]
    
    # Run all checks
    all_passed = True
    for check in checks:
        try:
            result = check.check_func()
            status = "✓" if result else "✗"
            logger.info(f"{status} {check.name}: {'PASS' if result else 'FAIL'}")
            if not result and check.required:
                all_passed = False
        except Exception as e:
            logger.error(f"✗ {check.name}: ERROR - {e}")
            if check.required:
                all_passed = False
    
    return all_passed

# Get platform-specific requirements
def get_platform_specific_requirements() -> List[str]:
    """Get platform-specific requirements"""
    platform_reqs = []
    
    # Windows-specific packages
    if sys.platform == "win32":
        platform_reqs.extend([
            "pywin32>=305",
            "windows-curses>=2.3.0",
        ])
    # macOS-specific packages
    elif sys.platform == "darwin":
        platform_reqs.extend([
            "pyobjc-framework-Cocoa>=9.0",
        ])
    # Linux-specific packages
    elif sys.platform.startswith("linux"):
        platform_reqs.extend([
            "python-apt>=2.4.0;platform_system=='Linux'",
        ])
    
    return platform_reqs

# Install packages with progress bar
def install_with_progress(packages: List[str], description: str = "Installing"):
    """Install packages with progress bar"""
    if HAS_TQDM:
        # Use tqdm progress bar if available
        with tqdm(total=len(packages), desc=description) as pbar:
            for pkg in packages:
                pbar.set_description(f"{description}: {pkg}")
                result = safe_subprocess_call(
                    [sys.executable, "-m", "pip", "install", pkg],
                    error_msg=f"Failed to install {pkg}"
                )
                pbar.update(1)
    else:
        # Fallback to simple progress indication
        for i, pkg in enumerate(packages, 1):
            logger.info(f"{description} ({i}/{len(packages)}): {pkg}")
            safe_subprocess_call(
                [sys.executable, "-m", "pip", "install", pkg],
                error_msg=f"Failed to install {pkg}"
            )

# Backup manager for safe operations
class BackupManager:
    """Manage backups before major changes"""
    
    def __init__(self):
        self.backup_dir = BACKUP_DIR
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, name: str = "auto") -> Path:
        """Create backup of current state"""
        # Generate timestamp for backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{name}_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup important directories
        dirs_to_backup = ["src", "configs", "requirements"]
        for dir_name in dirs_to_backup:
            src = ROOT_DIR / dir_name
            if src.exists():
                dst = backup_path / dir_name
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        
        # Save backup metadata
        metadata = {
            "timestamp": timestamp,
            "version": VERSION,
            "python_version": sys.version,
            "platform": platform.platform(),
        }
        
        with open(backup_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Backup created: {backup_path}")
        return backup_path
    
    def restore_backup(self, backup_path: Path) -> bool:
        """Restore from backup"""
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False
        
        try:
            # Read backup metadata
            metadata_file = backup_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                logger.info(f"Restoring backup from {metadata['timestamp']}")
            
            # Restore directories
            for dir_name in ["src", "configs", "requirements"]:
                src = backup_path / dir_name
                if src.exists():
                    dst = ROOT_DIR / dir_name
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
            
            logger.info("Backup restored successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False

# Initialize global backup manager
backup_manager = BackupManager()

# Read base requirements
if REQUIREMENTS_DIR.exists():
    INSTALL_REQUIRES = read_requirements_cached("base.txt")
else:
    # Fallback minimal requirements if requirements/ doesn't exist
    INSTALL_REQUIRES = [
        "torch>=2.0.0,<2.2.0",
        "transformers>=4.35.0,<5.0.0",
        "datasets>=2.14.0",
        "tokenizers>=0.15.0",
        "numpy>=1.24.0,<1.26.0",
        "pandas>=2.0.0,<2.2.0",
        "scikit-learn>=1.3.0,<1.4.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "rich>=13.5.0",
        "einops>=0.7.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ]

# Add platform-specific requirements
INSTALL_REQUIRES.extend(get_platform_specific_requirements())

# Build extras_require dictionary from requirements files
def build_extras_require() -> Dict[str, List[str]]:
    """Build extras_require dictionary from requirements files"""
    extras = {}
    
    # Read from requirements files if available
    if REQUIREMENTS_DIR.exists():
        # Map requirements files to extras keys
        requirements_mapping = {
            "ml": "ml.txt",
            "research": "research.txt",
            "prod": "prod.txt",
            "production": "prod.txt",  # Alias
            "dev": "dev.txt",
            "development": "dev.txt",  # Alias
            "data": "data.txt",
            "llm": "llm.txt",
            "ui": "ui.txt",
            "docs": "docs.txt",
            "documentation": "docs.txt",  # Alias
            "robustness": "robustness.txt",
            "minimal": "minimal.txt",
            "all": "all.txt",
        }
        
        # Read each requirements file
        for extra_name, req_file in requirements_mapping.items():
            if (REQUIREMENTS_DIR / req_file).exists():
                extras[extra_name] = read_requirements_cached(req_file)
    
    # Additional specialized bundles
    extras.update({
        # Testing dependencies
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-benchmark>=4.0.0",
            "pytest-mock>=3.12.0",
            "pytest-xdist>=3.5.0",
            "pytest-timeout>=2.2.0",
            "hypothesis>=6.90.0",
            "faker>=20.1.0",
            "factory-boy>=3.3.0",
        ],
        
        # GPU-specific dependencies
        "gpu": [
            "cuda-python>=12.0.0",
            "cupy-cuda12x>=12.0.0",
            "nvidia-ml-py>=12.535.0",
            "gpustat>=1.1.0",
            "py3nvml>=0.2.7",
            "pynvml>=11.5.0",
        ],
        
        # Cloud provider SDKs
        "aws": [
            "boto3>=1.28.0",
            "sagemaker>=2.199.0",
            "awscli>=1.29.0",
        ],
        
        "gcp": [
            "google-cloud-storage>=2.10.0",
            "google-cloud-aiplatform>=1.38.0",
            "google-cloud-logging>=3.5.0",
        ],
        
        "azure": [
            "azure-storage-blob>=12.19.0",
            "azure-ai-ml>=1.12.0",
            "azure-identity>=1.15.0",
        ],
        
        # Database dependencies
        "database": [
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0",
            "redis>=5.0.0",
            "pymongo>=4.5.0",
            "psycopg2-binary>=2.9.0",
            "asyncpg>=0.29.0",
        ],
        
        # Platform-specific bundles
        "colab": [
            "ipywidgets>=8.1.0",
            "gdown>=4.7.0",
        ],
        
        "kaggle": [
            "kaggle>=1.5.0",
        ],
        
        "sagemaker": [
            "sagemaker>=2.199.0",
            "sagemaker-experiments>=0.1.0",
        ],
        
        # Week-based learning dependencies
        "week1": [
            "scikit-learn>=1.3.0",
            "xgboost>=2.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "jupyter>=1.0.0",
            "wandb>=0.15.0",
        ],
        
        "week2-3": [
            "optuna>=3.3.0",
            "ray[tune]>=2.8.0",
            "wandb>=0.15.0",
            "pytorch-lightning>=2.0.0",
        ],
        
        "week4-5": [
            "nlpaug>=1.1.0",
            "textaugment>=1.3.4",
            "dvc>=3.0.0",
            "newspaper3k>=0.2.8",
        ],
        
        "week6-7": [
            "openai>=1.3.0",
            "peft>=0.6.0",
            "trl>=0.7.0",
            "langchain>=0.0.340",
        ],
        
        "week8-9": [
            "onnx>=1.15.0",
            "tensorrt>=8.6.0",
            "fastapi>=0.104.0",
            "bentoml>=1.1.0",
        ],
        
        "week10": [
            "docker>=6.1.0",
            "kubernetes>=28.1.0",
            "bentoml>=1.1.0",
            "prometheus-client>=0.19.0",
        ],
    })
    
    return extras

# Build extras_require
EXTRAS_REQUIRE = build_extras_require()

# Base class for commands with progress indication
class ProgressCommand(Command):
    """Base command with progress indicators"""
    
    def run_with_progress(self, tasks: List[Tuple[str, Callable]], description: str = "Processing"):
        """Run tasks with progress bar"""
        results = []
        
        if HAS_TQDM:
            # Use tqdm for progress visualization
            with tqdm(total=len(tasks), desc=description) as pbar:
                for task_name, task_func in tasks:
                    pbar.set_description(f"{description}: {task_name}")
                    try:
                        result = task_func()
                        results.append((task_name, "SUCCESS", result))
                        logger.info(f"✓ {task_name}: SUCCESS")
                    except Exception as e:
                        results.append((task_name, "FAILED", str(e)))
                        logger.error(f"✗ {task_name}: FAILED - {e}")
                    pbar.update(1)
        else:
            # Fallback to simple progress indication
            for i, (task_name, task_func) in enumerate(tasks, 1):
                logger.info(f"{description} ({i}/{len(tasks)}): {task_name}")
                try:
                    result = task_func()
                    results.append((task_name, "SUCCESS", result))
                    logger.info(f"✓ {task_name}: SUCCESS")
                except Exception as e:
                    results.append((task_name, "FAILED", str(e)))
                    logger.error(f"✗ {task_name}: FAILED - {e}")
        
        return results

# Custom post-installation command
class PostInstallCommand(install):
    """Post-installation for installation mode"""
    
    def run(self):
        """Run standard install and post-install tasks"""
        install.run(self)
        self.execute(self.post_install, [], msg="Running post-installation tasks...")
    
    def post_install(self):
        """Execute post-installation tasks"""
        logger.info("="*80)
        logger.info("AG News Text Classification Framework Installation Complete!")
        logger.info("="*80)
        self.print_next_steps()
        
        # Print installation metrics
        metrics_summary = metrics.get_summary()
        logger.info(f"Installation completed in {metrics_summary['total_time']:.2f}s")
        logger.info(f"Cache hit rate: {metrics_summary['cache_hit_rate']:.2%}")
    
    def print_next_steps(self):
        """Print next steps for user"""
        print("\nQuick Start Guide:")
        print("  1. Verify installation: ag-news --version")
        print("  2. Setup environment: ag-news setup --env research")
        print("  3. Download data: ag-news data download")
        print("  4. Train baseline: ag-news train --config configs/models/single/deberta_v3_xlarge.yaml")
        print("  5. Evaluate: ag-news evaluate --model outputs/models/fine_tuned/best_model.pt")
        
        print("\nEnvironment Setup Commands:")
        print("  - Research: pip install -e '.[research]'")
        print("  - Production: pip install '.[prod]'")
        print("  - Development: pip install -e '.[dev]'")
        print("  - GPU Support: pip install -e '.[gpu]'")
        print("  - Complete: pip install -e '.[all]'")
        
        print("\nDocumentation:")
        print("  - Online: https://ag-news-text-classification.readthedocs.io")
        print("  - Local: ag-news docs serve")
        
        print("\nModel Hub: https://huggingface.co/agnews-research")
        print("Community: https://github.com/agnews-research/ag-news-text-classification/discussions")
        print("="*80 + "\n")

# Development mode setup command
class DevelopCommand(develop):
    """Post-installation for development mode"""
    
    def run(self):
        """Run standard develop and setup development environment"""
        # Validate environment first
        if not validate_environment():
            raise SetupError("Environment validation failed")
        
        # Create backup before making changes
        backup_manager.create_backup("pre_develop")
        
        # Run standard develop command
        develop.run(self)
        self.execute(self.post_develop, [], msg="Setting up development environment...")
    
    def post_develop(self):
        """Setup development environment"""
        logger.info("Setting up development environment...")
        
        # Define development setup tasks
        tasks = [
            ("Pre-commit hooks", self.install_precommit),
            ("Husky hooks", self.install_husky),
            ("Directory structure", self.create_directories),
            ("Development tools", self.install_dev_tools),
        ]
        
        # Run tasks with progress indication
        cmd = ProgressCommand(self.distribution)
        results = cmd.run_with_progress(tasks, "Development Setup")
        
        # Report results
        success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
        logger.info(f"Development setup complete: {success_count}/{len(tasks)} tasks succeeded")
    
    def install_precommit(self):
        """Install pre-commit hooks"""
        if safe_subprocess_call(["pre-commit", "install"]) is not None:
            return True
        raise SetupError("Failed to install pre-commit hooks")
    
    def install_husky(self):
        """Install husky hooks"""
        husky_dir = ROOT_DIR / ".husky"
        if husky_dir.exists():
            result = safe_subprocess_call(["npx", "husky", "install"], error_msg="Husky installation failed")
            if result is not None:
                return True
        return False
    
    def create_directories(self):
        """Create necessary directories"""
        directories_to_create = [
            "outputs/models/checkpoints",
            "outputs/logs/training",
            "outputs/results/experiments",
            "data/raw/ag_news",
            "data/processed/train",
            "data/cache",
            ".cache/setup",
        ]
        
        for dir_path in directories_to_create:
            Path(ROOT_DIR / dir_path).mkdir(parents=True, exist_ok=True)
        return True
    
    def install_dev_tools(self):
        """Install development tools"""
        dev_tools = ["black", "isort", "flake8", "mypy", "pylint"]
        install_with_progress(dev_tools, "Installing dev tools")
        return True

# GPU environment setup command
class SetupGPUEnvironment(ProgressCommand):
    """Setup GPU environment for deep learning"""
    
    description = "Setup GPU environment with CUDA and optimization tools"
    user_options = [
        ("cuda-version=", "c", "CUDA version (11.8, 12.0, 12.1)"),
        ("install-drivers", "d", "Install NVIDIA drivers"),
        ("benchmark", "b", "Run GPU benchmark after setup"),
        ("multi-gpu", "m", "Setup for multi-GPU training"),
    ]
    
    def initialize_options(self):
        """Initialize command options"""
        self.cuda_version = SHARED_CONFIG["cuda_version"]
        self.install_drivers = False
        self.benchmark = False
        self.multi_gpu = False
    
    def finalize_options(self):
        """Validate command options"""
        assert self.cuda_version in ["11.8", "12.0", "12.1", "12.2"], "CUDA version must be 11.8, 12.0, 12.1, or 12.2"
    
    def run(self):
        """Setup GPU environment"""
        logger.info(f"Setting up GPU environment with CUDA {self.cuda_version}...")
        
        # Define GPU setup tasks
        tasks = [
            ("GPU Detection", self.detect_gpu),
            ("GPU Packages", self.install_gpu_packages),
            ("CUDA Environment", self.setup_cuda_environment),
        ]
        
        # Add optional tasks
        if self.multi_gpu:
            tasks.append(("Multi-GPU Setup", self.setup_multi_gpu))
        
        if self.benchmark:
            tasks.append(("GPU Benchmark", self.run_gpu_benchmark))
        
        # Run tasks with progress
        results = self.run_with_progress(tasks, "GPU Setup")
        
        # Report results
        success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
        logger.info(f"GPU setup complete: {success_count}/{len(tasks)} tasks succeeded")
    
    @retry(max_attempts=2)
    def detect_gpu(self):
        """Detect GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"Found {gpu_count} GPU(s)")
                # Log GPU details
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
                return True
            else:
                logger.warning("No CUDA GPUs found")
                if self.install_drivers:
                    self.install_nvidia_drivers()
                return False
        except ImportError:
            logger.warning("PyTorch not installed")
            return False
    
    def install_gpu_packages(self):
        """Install GPU packages"""
        # Define GPU-specific packages
        gpu_packages = [
            f"torch>=2.0.0+cu{self.cuda_version.replace('.', '')}",
            "nvidia-ml-py>=12.535.0",
            "gpustat>=1.1.0",
            "py3nvml>=0.2.7",
            "pynvml>=11.5.0",
        ]
        
        # Add CUDA version specific packages
        if self.cuda_version == "11.8":
            gpu_packages.extend([
                "cupy-cuda11x>=12.0.0",
                "cuda-python>=11.8.0",
            ])
        else:
            gpu_packages.extend([
                "cupy-cuda12x>=12.0.0",
                "cuda-python>=12.0.0",
            ])
        
        # Install packages with progress
        install_with_progress(gpu_packages, "Installing GPU packages")
        return True
    
    def install_nvidia_drivers(self):
        """Install NVIDIA drivers based on OS"""
        system = platform.system()
        
        if system == "Linux":
            logger.info("Installing NVIDIA drivers on Linux...")
            commands = [
                ["sudo", "apt-get", "update"],
                ["sudo", "apt-get", "install", "-y", "nvidia-driver-525"],
                ["sudo", "apt-get", "install", "-y", "nvidia-cuda-toolkit"],
            ]
            for cmd in commands:
                safe_subprocess_call(cmd)
        elif system == "Windows":
            logger.info("Please download NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
        else:
            logger.warning(f"Automatic driver installation not supported for {system}")
    
    def setup_multi_gpu(self):
        """Setup for multi-GPU training"""
        logger.info("Setting up multi-GPU environment...")
        
        # Install distributed training packages
        dist_packages = [
            "horovod>=0.28.0",
            "fairscale>=0.4.13",
            "deepspeed>=0.12.6",
        ]
        
        install_with_progress(dist_packages, "Installing multi-GPU packages")
        
        # Set environment variables for distributed training
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        
        return True
    
    def run_gpu_benchmark(self):
        """Run GPU benchmark"""
        logger.info("Running GPU benchmark...")
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.warning("No GPU available for benchmark")
                return False
            
            device = torch.device("cuda")
            
            # Matrix multiplication benchmark
            sizes = [1024, 2048, 4096, 8192]
            
            for size in sizes:
                # Create random matrices
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                
                # Warmup
                for _ in range(10):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                
                # Benchmark
                start = time.time()
                for _ in range(100):
                    c = torch.matmul(a, b)
                torch.cuda.synchronize()
                elapsed = time.time() - start
                
                # Calculate TFLOPS
                tflops = (2 * size**3 * 100) / (elapsed * 1e12)
                logger.info(f"  Matrix size {size}x{size}: {tflops:.2f} TFLOPS")
            
            return True
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return False
    
    def setup_cuda_environment(self):
        """Setup CUDA environment variables"""
        logger.info("Setting up CUDA environment variables...")
        
        # Define CUDA environment variables
        cuda_vars = {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",  # All GPUs
            "CUDA_LAUNCH_BLOCKING": "0",  # Async execution
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # Deterministic ops
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",  # Memory management
        }
        
        # Save to environment file
        env_file = ROOT_DIR / ".env.gpu"
        with open(env_file, "w") as f:
            for key, value in cuda_vars.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"GPU environment variables saved to {env_file}")
        return True

# Continue with other custom commands...
# [Rest of the code continues with similar detailed comments]

# Check dependency conflicts before setup
if not dependency_resolver.resolve():
    logger.warning("Dependency conflicts detected. Installation may have issues.")

# Main setup configuration
setup(
    # Package metadata
    name=METADATA["name"],
    version=VERSION,
    author=METADATA["author"],
    author_email=METADATA["author_email"],
    maintainer=METADATA["maintainer"],
    maintainer_email=METADATA["maintainer_email"],
    license=METADATA["license"],
    
    # Package description
    description="State-of-the-art framework for AG News text classification with advanced ML/DL techniques",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # URLs
    url=METADATA["url"],
    download_url=f"{METADATA['url']}/archive/v{VERSION}.tar.gz",
    project_urls={
        "Bug Tracker": f"{METADATA['url']}/issues",
        "Documentation": "https://ag-news-text-classification.readthedocs.io",
        "Source Code": METADATA["url"],
        "Research Paper": "https://arxiv.org/abs/2024.agnews",
        "Model Hub": "https://huggingface.co/VoHaiDung",
        "Demo": "https://huggingface.co/spaces/VoHaiDung/ag-news-demo",
        "Changelog": f"{METADATA['url']}/blob/main/CHANGELOG.md",
        "Benchmarks": "https://agnews-benchmarks.github.io",
        "CI/CD": f"{METADATA['url']}/actions",
        "Discussions": f"{METADATA['url']}/discussions",
        "Wiki": f"{METADATA['url']}/wiki",
    },
    
    # Package configuration
    packages=find_packages(
        where=".",
        include=["src*", "quickstart*"],
        exclude=["tests*", "docs*", "examples*"]
    ),
    package_dir={"": "."},
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.8,<3.12",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Console scripts and entry points
    entry_points={
        "console_scripts": [
            "ag-news=src.cli.main:cli",
            # Add more console scripts...
        ],
    },
    
    # Custom commands
    cmdclass={
        "install": PostInstallCommand,
        "develop": DevelopCommand,
        "setup_gpu": SetupGPUEnvironment,
        # Add more custom commands...
    },
    
    # Additional configuration
    zip_safe=False,
    platforms=["any"],
)

# Create necessary files on setup
manifest_path = ROOT_DIR / "MANIFEST.in"
if not manifest_path.exists():
    manifest_content = """
include README.md LICENSE CITATION.cff CHANGELOG.md
include ARCHITECTURE.md PERFORMANCE.md SECURITY.md
recursive-include configs *.yaml *.yml *.json
recursive-include src *.py *.pyi py.typed
# Add more manifest entries...
"""
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest_content)
    logger.info("Created MANIFEST.in file")

# Create py.typed file for type hints support
py_typed_file = SRC_DIR / "py.typed"
if not py_typed_file.exists():
    py_typed_file.touch()
    logger.info("Created py.typed file for type hints support")

# Cleanup on exit
import atexit

def cleanup_on_exit():
    """Cleanup on exit"""
    try:
        # Log final metrics
        metrics_summary = metrics.get_summary()
        logger.info(f"Setup completed in {metrics_summary['total_time']:.2f}s")
        logger.info(f"Cache hit rate: {metrics_summary['cache_hit_rate']:.2%}")
        
        # Clean cache directory if needed
        if CACHE_DIR.exists():
            import shutil
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
    except:
        pass

# Register cleanup function
atexit.register(cleanup_on_exit)
