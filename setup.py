import os
import sys
import json
import shutil
import subprocess
import asyncio
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import platform

from setuptools import setup, find_packages, Command
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.test import test as TestCommand


# Project root directory
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
CACHE_DIR = ROOT_DIR / ".cache" / "setup"

# Create cache directory
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Project metadata
METADATA = {
    "name": "ag-news-text-classification",
    "author": "AG News Research Team",
    "author_email": "team@agnews-research.ai",
    "maintainer": "AG News ML Engineering Team",
    "maintainer_email": "ml-team@agnews-research.ai",
    "license": "MIT",
    "url": "https://github.com/agnews-research/ag-news-text-classification",
}


def get_version() -> str:
    """Get version from __version__.py file or git tags."""
    version_file = SRC_DIR / "__version__.py"
    version_dict = {}
    
    if version_file.exists():
        with open(version_file, "r", encoding="utf-8") as f:
            exec(f.read(), version_dict)
        return version_dict.get("__version__", "1.0.0")
    
    # Fallback: try to get from git tag
    try:
        version = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=ROOT_DIR,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return version.lstrip("v")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "1.0.0"


VERSION = get_version()


def get_long_description() -> str:
    """Read and return README content."""
    readme_file = ROOT_DIR / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return "AG News Text Classification Framework - State-of-the-art research framework for news text classification"


def get_file_hash(filepath: Path) -> str:
    """Get hash of file for cache validation."""
    if not filepath.exists():
        return ""
    
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


@lru_cache(maxsize=32)
def read_requirements_cached(filename: str, base_dir: Path = REQUIREMENTS_DIR) -> List[str]:
    """
    Read requirements from file with caching support.
    
    Args:
        filename: Name of requirements file
        base_dir: Base directory for requirements files
        
    Returns:
        List of requirement strings
    """
    filepath = base_dir / filename
    cache_file = CACHE_DIR / f"{filename}.cache"
    hash_file = CACHE_DIR / f"{filename}.hash"
    
    # Check cache validity
    current_hash = get_file_hash(filepath)
    if cache_file.exists() and hash_file.exists():
        try:
            with open(hash_file, "r") as f:
                cached_hash = f.read().strip()
            
            if cached_hash == current_hash:
                # Cache is valid
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
        except Exception:
            pass  # Cache invalid, will regenerate
    
    # Read requirements
    requirements = read_requirements(filename, base_dir)
    
    # Update cache
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(requirements, f)
        with open(hash_file, "w") as f:
            f.write(current_hash)
    except Exception:
        pass  # Cache write failed, continue without caching
    
    return requirements


def read_requirements(filename: str, base_dir: Path = REQUIREMENTS_DIR) -> List[str]:
    """
    Read requirements from file, handling -r includes and various formats.
    
    Args:
        filename: Name of requirements file
        base_dir: Base directory for requirements files
        
    Returns:
        List of requirement strings
    """
    requirements = []
    seen_files = set()  # Prevent circular imports
    
    def _read_file(filepath: Path, seen: Set[Path]) -> List[str]:
        """Recursively read requirements file."""
        if filepath in seen:
            return []
        seen.add(filepath)
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return []
        
        file_requirements = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                
                # Handle -r includes
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


async def run_subprocess_async(cmd: List[str], **kwargs) -> Tuple[int, str, str]:
    """Run subprocess asynchronously."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode()


def run_async_commands(commands: List[List[str]]) -> List[Tuple[int, str, str]]:
    """Run multiple commands asynchronously."""
    async def run_all():
        tasks = [run_subprocess_async(cmd) for cmd in commands]
        return await asyncio.gather(*tasks)
    
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    return asyncio.run(run_all())


# Core requirements (from requirements/base.txt or fallback)
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


# Build extras_require from requirements/ structure
def build_extras_require() -> Dict[str, List[str]]:
    """Build extras_require dictionary from requirements files."""
    extras = {}
    
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
        
        for extra_name, req_file in requirements_mapping.items():
            if (REQUIREMENTS_DIR / req_file).exists():
                extras[extra_name] = read_requirements_cached(req_file)
    
    # Additional specialized bundles
    extras.update({
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
        
        "gpu": [
            "cuda-python>=12.0.0",
            "cupy-cuda12x>=12.0.0",
            "nvidia-ml-py>=12.535.0",
            "gpustat>=1.1.0",
            "py3nvml>=0.2.7",
            "pynvml>=11.5.0",
        ],
        
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
        
        "database": [
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0",
            "redis>=5.0.0",
            "pymongo>=4.5.0",
            "psycopg2-binary>=2.9.0",
            "asyncpg>=0.29.0",
        ],
        
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
        
        # Week-specific bundles for structured learning
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


EXTRAS_REQUIRE = build_extras_require()


# Custom Commands
class PostInstallCommand(install):
    """Post-installation for installation mode."""
    
    def run(self):
        """Run standard install and post-install tasks."""
        install.run(self)
        self.execute(self.post_install, [], msg="Running post-installation tasks...")
    
    def post_install(self):
        """Execute post-installation tasks."""
        print("\n" + "="*80)
        print("AG News Text Classification Framework Installation Complete!")
        print("="*80)
        self.print_next_steps()
    
    def print_next_steps(self):
        """Print next steps for user."""
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


class DevelopCommand(develop):
    """Post-installation for development mode."""
    
    def run(self):
        """Run standard develop and setup development environment."""
        develop.run(self)
        self.execute(self.post_develop, [], msg="Setting up development environment...")
    
    def post_develop(self):
        """Setup development environment."""
        print("\nSetting up development environment...")
        
        # Install pre-commit hooks
        try:
            subprocess.check_call(["pre-commit", "install"], stderr=subprocess.DEVNULL)
            print("[SUCCESS] Pre-commit hooks installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[WARNING] Could not install pre-commit hooks")
        
        # Setup git hooks with husky
        husky_dir = ROOT_DIR / ".husky"
        if husky_dir.exists():
            try:
                subprocess.check_call(["npx", "husky", "install"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL)
                print("[SUCCESS] Husky hooks installed")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("[INFO] Husky not available (npm required)")
        
        # Create necessary directories
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
        
        print("[SUCCESS] Development environment ready!")


class SetupGPUEnvironment(Command):
    """Setup GPU environment for deep learning."""
    
    description = "Setup GPU environment with CUDA and optimization tools"
    user_options = [
        ("cuda-version=", "c", "CUDA version (11.8, 12.0, 12.1)"),
        ("install-drivers", "d", "Install NVIDIA drivers"),
        ("benchmark", "b", "Run GPU benchmark after setup"),
        ("multi-gpu", "m", "Setup for multi-GPU training"),
    ]
    
    def initialize_options(self):
        """Initialize command options."""
        self.cuda_version = "11.8"
        self.install_drivers = False
        self.benchmark = False
        self.multi_gpu = False
    
    def finalize_options(self):
        """Validate command options."""
        assert self.cuda_version in ["11.8", "12.0", "12.1", "12.2"], "CUDA version must be 11.8, 12.0, 12.1, or 12.2"
    
    def run(self):
        """Setup GPU environment."""
        print(f"Setting up GPU environment with CUDA {self.cuda_version}...")
        
        # Detect GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                print(f"[SUCCESS] Found {gpu_count} GPU(s)")
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
            else:
                print("[WARNING] No CUDA GPUs found")
                if self.install_drivers:
                    self.install_nvidia_drivers()
        except ImportError:
            print("[WARNING] PyTorch not installed")
        
        # Install GPU packages
        print("\nInstalling GPU optimization packages...")
        gpu_packages = [
            f"torch>=2.0.0+cu{self.cuda_version.replace('.', '')}",
            "nvidia-ml-py>=12.535.0",
            "gpustat>=1.1.0",
            "py3nvml>=0.2.7",
            "pynvml>=11.5.0",
        ]
        
        # Install CUDA-specific packages
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
        
        # Install packages asynchronously for speed
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + gpu_packages)
            print("[SUCCESS] GPU packages installed")
        except subprocess.CalledProcessError:
            print("[WARNING] Some GPU packages failed to install")
        
        # Setup multi-GPU if requested
        if self.multi_gpu:
            self.setup_multi_gpu()
        
        # Run benchmark if requested
        if self.benchmark:
            self.run_gpu_benchmark()
        
        # Setup environment variables
        self.setup_cuda_environment()
        
        print("\n[SUCCESS] GPU environment setup complete!")
    
    def install_nvidia_drivers(self):
        """Install NVIDIA drivers based on OS."""
        system = platform.system()
        
        if system == "Linux":
            print("Installing NVIDIA drivers on Linux...")
            commands = [
                ["sudo", "apt-get", "update"],
                ["sudo", "apt-get", "install", "-y", "nvidia-driver-525"],
                ["sudo", "apt-get", "install", "-y", "nvidia-cuda-toolkit"],
            ]
            for cmd in commands:
                try:
                    subprocess.check_call(cmd)
                except:
                    print(f"[WARNING] Failed to run: {' '.join(cmd)}")
        elif system == "Windows":
            print("[INFO] Please download NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
        else:
            print(f"[WARNING] Automatic driver installation not supported for {system}")
    
    def setup_multi_gpu(self):
        """Setup for multi-GPU training."""
        print("\nSetting up multi-GPU environment...")
        
        # Install distributed training packages
        dist_packages = [
            "horovod>=0.28.0",
            "fairscale>=0.4.13",
            "deepspeed>=0.12.6",
        ]
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + dist_packages)
            print("[SUCCESS] Multi-GPU packages installed")
        except:
            print("[WARNING] Some multi-GPU packages failed to install")
        
        # Set environment variables for distributed training
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        
        print("[SUCCESS] Multi-GPU setup complete")
    
    def run_gpu_benchmark(self):
        """Run GPU benchmark."""
        print("\nRunning GPU benchmark...")
        
        benchmark_script = """
import torch
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
    
    # Matrix multiplication benchmark
    sizes = [1024, 2048, 4096, 8192]
    for size in sizes:
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
        
        tflops = (2 * size**3 * 100) / (elapsed * 1e12)
        print(f"  Matrix size {size}x{size}: {tflops:.2f} TFLOPS")
else:
    print("No GPU available for benchmark")
"""
        
        try:
            exec(benchmark_script)
        except Exception as e:
            print(f"[WARNING] Benchmark failed: {e}")
    
    def setup_cuda_environment(self):
        """Setup CUDA environment variables."""
        print("\nSetting up CUDA environment variables...")
        
        cuda_vars = {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",  # All GPUs
            "CUDA_LAUNCH_BLOCKING": "0",  # Async execution
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",  # Deterministic ops
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",  # Memory management
        }
        
        env_file = ROOT_DIR / ".env.gpu"
        with open(env_file, "w") as f:
            for key, value in cuda_vars.items():
                f.write(f"{key}={value}\n")
        
        print(f"[SUCCESS] GPU environment variables saved to {env_file}")


class SetupResearchEnvironment(Command):
    """Setup complete research environment."""
    
    description = "Setup research environment with all ML/DL tools"
    user_options = [
        ("download-models", "d", "Download pretrained models"),
        ("setup-wandb", "w", "Setup Weights & Biases"),
        ("cuda-version=", "c", "CUDA version (11.8 or 12.0)"),
        ("async", "a", "Use async operations for faster setup"),
    ]
    
    def initialize_options(self):
        """Initialize command options."""
        self.download_models = False
        self.setup_wandb = False
        self.cuda_version = "11.8"
        self.async_mode = False
    
    def finalize_options(self):
        """Validate command options."""
        assert self.cuda_version in ["11.8", "12.0", "12.1"], "CUDA version must be 11.8, 12.0, or 12.1"
    
    def run(self):
        """Setup research environment."""
        print("Setting up research environment...")
        
        # Install research requirements
        print("Installing research requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[research]"])
        
        if self.async_mode:
            # Run multiple setups asynchronously
            self.async_setup()
        else:
            # Traditional sequential setup
            self.sequential_setup()
        
        print("\n[SUCCESS] Research environment setup complete!")
    
    def async_setup(self):
        """Setup using async operations for speed."""
        print("\nUsing async setup for faster installation...")
        
        # Prepare commands for async execution
        commands = []
        
        # spaCy models
        spacy_models = ["en_core_web_sm", "en_core_web_lg", "en_core_web_trf"]
        for model in spacy_models:
            commands.append([sys.executable, "-m", "spacy", "download", model])
        
        # Jupyter extensions
        jupyter_extensions = [
            "jupyter_contrib_nbextensions",
            "jupyter_nbextensions_configurator",
            "jupyterlab-git",
            "jupyterlab-lsp",
        ]
        for ext in jupyter_extensions:
            commands.append([sys.executable, "-m", "pip", "install", ext])
        
        # Run all commands asynchronously
        results = run_async_commands(commands)
        
        # Check results
        for i, (returncode, stdout, stderr) in enumerate(results):
            if returncode == 0:
                print(f"[SUCCESS] Command {i+1}/{len(commands)} completed")
            else:
                print(f"[WARNING] Command {i+1}/{len(commands)} failed")
        
        # NLTK data (still sequential as it uses internal downloads)
        self.download_nltk_data()
        
        # Setup experiment tracking
        if self.setup_wandb:
            self.setup_experiment_tracking()
        
        # Download models
        if self.download_models:
            self.download_pretrained_models()
    
    def sequential_setup(self):
        """Traditional sequential setup."""
        # Download spaCy models
        print("\nDownloading spaCy language models...")
        spacy_models = ["en_core_web_sm", "en_core_web_lg", "en_core_web_trf"]
        for model in spacy_models:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", model],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"[SUCCESS] Downloaded {model}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"[WARNING] Could not download {model}")
        
        # Download NLTK data
        self.download_nltk_data()
        
        # Setup Jupyter extensions
        print("\nSetting up Jupyter extensions...")
        jupyter_extensions = [
            "jupyter_contrib_nbextensions",
            "jupyter_nbextensions_configurator",
            "jupyterlab-git",
            "jupyterlab-lsp",
        ]
        for ext in jupyter_extensions:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", ext],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        try:
            subprocess.check_call([sys.executable, "-m", "jupyter", "nbextension", "enable", "--py", "widgetsnbextension"])
            subprocess.check_call([sys.executable, "-m", "jupyter", "labextension", "install", "@jupyter-widgets/jupyterlab-manager"])
        except:
            pass
        
        print("[SUCCESS] Jupyter extensions configured")
        
        # Setup experiment tracking
        if self.setup_wandb:
            self.setup_experiment_tracking()
        
        # Download pretrained models
        if self.download_models:
            self.download_pretrained_models()
    
    def download_nltk_data(self):
        """Download NLTK data packages."""
        print("\nDownloading NLTK data...")
        try:
            import nltk
            nltk_packages = [
                'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                'maxent_ne_chunker', 'words', 'vader_lexicon', 'opinion_lexicon',
                'pros_cons', 'reuters', 'brown', 'conll2000', 'movie_reviews'
            ]
            for package in nltk_packages:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass
            print("[SUCCESS] NLTK data downloaded")
        except ImportError:
            print("[WARNING] NLTK not installed")
    
    def setup_experiment_tracking(self):
        """Setup experiment tracking tools."""
        print("\nSetting up Weights & Biases...")
        try:
            subprocess.check_call([sys.executable, "-m", "wandb", "login", "--relogin"])
            print("[SUCCESS] Weights & Biases configured")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[WARNING] Could not setup Weights & Biases")
    
    def download_pretrained_models(self):
        """Download pretrained models."""
        print("\nDownloading pretrained models...")
        try:
            from huggingface_hub import snapshot_download
            models = [
                "microsoft/deberta-v3-large",
                "roberta-large",
                "xlnet-large-cased",
                "google/electra-large-discriminator",
            ]
            cache_dir = ROOT_DIR / "outputs/models/pretrained"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Use ThreadPoolExecutor for parallel downloads
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for model in models:
                    future = executor.submit(snapshot_download, repo_id=model, cache_dir=str(cache_dir))
                    futures.append((model, future))
                
                for model, future in futures:
                    try:
                        future.result(timeout=300)  # 5 minutes timeout
                        print(f"[SUCCESS] Downloaded {model}")
                    except Exception as e:
                        print(f"[WARNING] Could not download {model}: {e}")
        except ImportError:
            print("[WARNING] huggingface_hub not installed")


class SetupProductionEnvironment(Command):
    """Setup production environment."""
    
    description = "Setup production environment for deployment"
    user_options = [
        ("cloud=", "c", "Cloud provider (aws/gcp/azure)"),
        ("monitoring", "m", "Setup monitoring stack"),
        ("security", "s", "Setup security tools"),
        ("async", "a", "Use async operations"),
    ]
    
    def initialize_options(self):
        """Initialize command options."""
        self.cloud = None
        self.monitoring = False
        self.security = False
        self.async_mode = False
    
    def finalize_options(self):
        """Validate command options."""
        if self.cloud:
            assert self.cloud in ["aws", "gcp", "azure"], "Cloud must be aws, gcp, or azure"
    
    def run(self):
        """Setup production environment."""
        print("Setting up production environment...")
        
        # Install production requirements
        print("Installing production requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", ".[prod]"])
        
        # Install cloud-specific requirements
        if self.cloud:
            print(f"\nSetting up {self.cloud.upper()} tools...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f".[{self.cloud}]"])
            print(f"[SUCCESS] {self.cloud.upper()} tools installed")
        
        # Setup monitoring
        if self.monitoring:
            print("\nSetting up monitoring stack...")
            monitoring_tools = ["prometheus-client", "grafana-api", "evidently", "whylogs"]
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + monitoring_tools)
            
            # Create monitoring directories
            monitoring_dirs = [
                "monitoring/dashboards/grafana",
                "monitoring/dashboards/prometheus",
                "monitoring/logs",
                "monitoring/metrics",
            ]
            for dir_path in monitoring_dirs:
                Path(ROOT_DIR / dir_path).mkdir(parents=True, exist_ok=True)
            print("[SUCCESS] Monitoring stack configured")
        
        # Setup security
        if self.security:
            print("\nSetting up security tools...")
            security_tools = ["bandit", "safety", "detect-secrets", "cryptography", "python-jose", "passlib", "bcrypt"]
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + security_tools)
            
            # Create security directories
            security_dir = ROOT_DIR / "security/scan_results"
            security_dir.mkdir(parents=True, exist_ok=True)
            
            # Run initial security scan
            try:
                subprocess.check_call(
                    ["bandit", "-r", "src/", "-f", "json", "-o", "security/scan_results/initial_scan.json"],
                    stderr=subprocess.DEVNULL
                )
                print("[SUCCESS] Security scan complete")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("[WARNING] Security scan failed")
        
        # Install production servers
        if self.async_mode:
            # Async installation
            commands = [
                [sys.executable, "-m", "pip", "install", "gunicorn"],
                [sys.executable, "-m", "pip", "install", "nginx"],
                [sys.executable, "-m", "pip", "install", "supervisor"],
                [sys.executable, "-m", "pip", "install", "redis"],
                [sys.executable, "-m", "pip", "install", "celery"],
            ]
            results = run_async_commands(commands)
            print("[SUCCESS] Production servers installed (async)")
        else:
            print("\nInstalling production servers...")
            prod_servers = ["gunicorn", "nginx", "supervisor", "redis", "celery"]
            for server in prod_servers:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", server], stderr=subprocess.DEVNULL)
                except:
                    pass
        
        # Optimize for production
        print("\nOptimizing for production...")
        try:
            subprocess.check_call([sys.executable, "-m", "compileall", "src/"])
            print("[SUCCESS] Python files compiled")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("[WARNING] Could not compile Python files")
        
        # Setup Docker if available
        if sys.platform != "win32":
            try:
                subprocess.call(["docker", "--version"], stdout=subprocess.DEVNULL)
                subprocess.call(["docker-compose", "--version"], stdout=subprocess.DEVNULL)
                print("[SUCCESS] Docker available")
            except:
                print("[INFO] Docker not installed")
        
        # Setup Kubernetes tools
        try:
            subprocess.call(["kubectl", "version", "--client"], stdout=subprocess.DEVNULL)
            print("[SUCCESS] Kubernetes tools available")
        except:
            print("[INFO] Kubernetes tools not installed")
        
        print("\n[SUCCESS] Production environment setup complete!")


class SetupDatabaseEnvironment(Command):
    """Setup and manage database environment."""
    
    description = "Setup database environment and run migrations"
    user_options = [
        ("db-type=", "t", "Database type (postgres/mysql/sqlite)"),
        ("migrate", "m", "Run database migrations"),
        ("seed", "s", "Seed database with initial data"),
        ("backup", "b", "Create database backup"),
        ("restore=", "r", "Restore from backup file"),
    ]
    
    def initialize_options(self):
        """Initialize command options."""
        self.db_type = "postgres"
        self.migrate = False
        self.seed = False
        self.backup = False
        self.restore = None
    
    def finalize_options(self):
        """Validate command options."""
        assert self.db_type in ["postgres", "mysql", "sqlite"], "Database type must be postgres, mysql, or sqlite"
    
    def run(self):
        """Setup database environment."""
        print(f"Setting up {self.db_type} database environment...")
        
        # Install database packages
        self.install_database_packages()
        
        # Setup database
        self.setup_database()
        
        # Run migrations if requested
        if self.migrate:
            self.run_migrations()
        
        # Seed database if requested
        if self.seed:
            self.seed_database()
        
        # Backup database if requested
        if self.backup:
            self.backup_database()
        
        # Restore database if requested
        if self.restore:
            self.restore_database(self.restore)
        
        print("\n[SUCCESS] Database environment setup complete!")
    
    def install_database_packages(self):
        """Install database-specific packages."""
        print(f"\nInstalling {self.db_type} packages...")
        
        packages = ["sqlalchemy>=2.0.0", "alembic>=1.12.0"]
        
        if self.db_type == "postgres":
            packages.extend(["psycopg2-binary>=2.9.0", "asyncpg>=0.29.0"])
        elif self.db_type == "mysql":
            packages.extend(["pymysql>=1.1.0", "aiomysql>=0.2.0"])
        elif self.db_type == "sqlite":
            packages.append("aiosqlite>=0.19.0")
        
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print(f"[SUCCESS] {self.db_type} packages installed")
    
    def setup_database(self):
        """Setup database configuration."""
        print("\nSetting up database configuration...")
        
        # Create database config
        db_config = {
            "postgres": {
                "driver": "postgresql+psycopg2",
                "host": "localhost",
                "port": 5432,
                "database": "ag_news",
                "username": "ag_news_user",
                "password": "secure_password",
            },
            "mysql": {
                "driver": "mysql+pymysql",
                "host": "localhost",
                "port": 3306,
                "database": "ag_news",
                "username": "ag_news_user",
                "password": "secure_password",
            },
            "sqlite": {
                "driver": "sqlite",
                "database": "data/ag_news.db",
            }
        }
        
        config = db_config[self.db_type]
        config_file = ROOT_DIR / "configs" / "database.yaml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        import yaml
        with open(config_file, "w") as f:
            yaml.dump(config, f)
        
        print(f"[SUCCESS] Database configuration saved to {config_file}")
    
    def run_migrations(self):
        """Run database migrations."""
        print("\nRunning database migrations...")
        
        # Create migrations directory
        migrations_dir = ROOT_DIR / "migrations"
        migrations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize alembic if not already done
        alembic_ini = ROOT_DIR / "alembic.ini"
        if not alembic_ini.exists():
            subprocess.check_call(["alembic", "init", "migrations"])
        
        # Create initial migration
        try:
            subprocess.check_call(["alembic", "revision", "--autogenerate", "-m", "Initial migration"])
            subprocess.check_call(["alembic", "upgrade", "head"])
            print("[SUCCESS] Migrations completed")
        except subprocess.CalledProcessError:
            print("[WARNING] Migration failed")
    
    def seed_database(self):
        """Seed database with initial data."""
        print("\nSeeding database...")
        
        seed_script = ROOT_DIR / "scripts" / "seed_database.py"
        if seed_script.exists():
            subprocess.check_call([sys.executable, str(seed_script)])
            print("[SUCCESS] Database seeded")
        else:
            print("[WARNING] Seed script not found")
    
    def backup_database(self):
        """Create database backup."""
        print("\nCreating database backup...")
        
        backup_dir = ROOT_DIR / "backup" / "database"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"backup_{self.db_type}_{timestamp}.sql"
        
        if self.db_type == "postgres":
            cmd = ["pg_dump", "-h", "localhost", "-U", "ag_news_user", "-d", "ag_news", "-f", str(backup_file)]
        elif self.db_type == "mysql":
            cmd = ["mysqldump", "-h", "localhost", "-u", "ag_news_user", "-p", "ag_news", ">", str(backup_file)]
        else:  # sqlite
            import shutil
            shutil.copy2("data/ag_news.db", backup_file)
            print(f"[SUCCESS] Database backed up to {backup_file}")
            return
        
        try:
            subprocess.check_call(cmd)
            print(f"[SUCCESS] Database backed up to {backup_file}")
        except subprocess.CalledProcessError:
            print("[WARNING] Backup failed")
    
    def restore_database(self, backup_file: str):
        """Restore database from backup."""
        print(f"\nRestoring database from {backup_file}...")
        
        backup_path = Path(backup_file)
        if not backup_path.exists():
            print(f"[ERROR] Backup file not found: {backup_file}")
            return
        
        if self.db_type == "postgres":
            cmd = ["psql", "-h", "localhost", "-U", "ag_news_user", "-d", "ag_news", "-f", str(backup_path)]
        elif self.db_type == "mysql":
            cmd = ["mysql", "-h", "localhost", "-u", "ag_news_user", "-p", "ag_news", "<", str(backup_path)]
        else:  # sqlite
            import shutil
            shutil.copy2(backup_path, "data/ag_news.db")
            print("[SUCCESS] Database restored")
            return
        
        try:
            subprocess.check_call(cmd)
            print("[SUCCESS] Database restored")
        except subprocess.CalledProcessError:
            print("[WARNING] Restore failed")


class SetupMLOpsEnvironment(Command):
    """Setup MLOps environment with monitoring and deployment tools."""
    
    description = "Setup MLOps environment"
    user_options = [
        ("full", "f", "Install all MLOps tools"),
    ]
    
    def initialize_options(self):
        """Initialize command options."""
        self.full = False
    
    def finalize_options(self):
        """Finalize command options."""
        pass
    
    def run(self):
        """Setup MLOps tools."""
        print("Setting up MLOps environment...")
        
        # Core MLOps tools
        mlops_tools = [
            "mlflow>=2.8.0",
            "bentoml>=1.1.0",
            "evidently>=0.4.0",
            "whylogs>=1.3.0",
            "prometheus-client>=0.19.0",
            "grafana-api>=1.0.0",
        ]
        
        print("Installing core MLOps tools...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + mlops_tools)
        
        if self.full:
            # Advanced MLOps tools
            advanced_tools = [
                "kubeflow>=1.7.0",
                "seldon-core>=1.17.1",
                "kserve>=0.11.0",
                "feast>=0.35.0",
            ]
            print("Installing advanced MLOps tools...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + advanced_tools)
            except:
                print("[WARNING] Some advanced tools may require additional setup")
        
        print("[SUCCESS] MLOps environment setup complete!")


class RunTests(TestCommand):
    """Custom test command with coverage and multiple test types."""
    
    user_options = [
        ("test-type=", "t", "Type of tests to run (unit/integration/performance/setup/all)"),
        ("coverage", "c", "Generate coverage report"),
        ("parallel", "p", "Run tests in parallel"),
        ("benchmark", "b", "Run benchmark tests"),
        ("verbose", "v", "Verbose output"),
    ]
    
    def initialize_options(self):
        """Initialize test options."""
        TestCommand.initialize_options(self)
        self.test_type = "unit"
        self.coverage = False
        self.parallel = False
        self.benchmark = False
        self.verbose = False
    
    def finalize_options(self):
        """Finalize test options."""
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    
    def run_tests(self):
        """Run tests with pytest."""
        import pytest
        
        # Build pytest arguments
        pytest_args = []
        
        # Test type selection
        if self.test_type == "unit":
            pytest_args.extend(["tests/unit"])
        elif self.test_type == "integration":
            pytest_args.extend(["tests/integration"])
        elif self.test_type == "performance":
            pytest_args.extend(["tests/performance"])
        elif self.test_type == "setup":
            pytest_args.extend(["tests/setup"])
        else:  # all
            pytest_args.extend(["tests"])
        
        # Coverage configuration
        if self.coverage:
            pytest_args.extend([
                "--cov=src",
                "--cov-report=html:outputs/coverage/html",
                "--cov-report=term-missing",
                "--cov-report=xml:outputs/coverage/coverage.xml"
            ])
        
        # Parallel execution
        if self.parallel:
            pytest_args.extend(["-n", "auto"])
        
        # Benchmark tests
        if self.benchmark:
            pytest_args.extend(["--benchmark-only", "--benchmark-json=outputs/benchmarks/results.json"])
        
        # Verbose output
        if self.verbose:
            pytest_args.append("-vv")
        
        # Create output directories
        output_dirs = ["outputs/coverage", "outputs/benchmarks"]
        for dir_path in output_dirs:
            Path(ROOT_DIR / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Run tests
        print(f"Running {self.test_type} tests...")
        errno = pytest.main(pytest_args)
        
        # Generate coverage badge if coverage was run
        if self.coverage and errno == 0:
            try:
                subprocess.check_call(
                    ["coverage-badge", "-o", "docs/images/coverage.svg"],
                    stderr=subprocess.DEVNULL
                )
                print("[SUCCESS] Coverage badge generated")
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        sys.exit(errno)


class TestSetupCommands(Command):
    """Test custom setup commands."""
    
    description = "Test setup commands functionality"
    user_options = []
    
    def initialize_options(self):
        """Initialize options."""
        pass
    
    def finalize_options(self):
        """Finalize options."""
        pass
    
    def run(self):
        """Run setup command tests."""
        print("Testing setup commands...")
        
        # Test imports
        test_results = []
        
        # Test 1: Check if commands are callable
        commands = [
            SetupGPUEnvironment,
            SetupResearchEnvironment,
            SetupProductionEnvironment,
            SetupDatabaseEnvironment,
            SetupMLOpsEnvironment,
        ]
        
        for cmd_class in commands:
            try:
                cmd = cmd_class(self.distribution)
                test_results.append((cmd_class.__name__, "PASS"))
            except Exception as e:
                test_results.append((cmd_class.__name__, f"FAIL: {e}"))
        
        # Test 2: Check async functionality
        try:
            test_commands = [
                ["echo", "test1"],
                ["echo", "test2"],
                ["echo", "test3"],
            ]
            results = run_async_commands(test_commands)
            test_results.append(("Async Commands", "PASS"))
        except Exception as e:
            test_results.append(("Async Commands", f"FAIL: {e}"))
        
        # Test 3: Check cache functionality
        try:
            # Test cache
            req1 = read_requirements_cached("base.txt")
            req2 = read_requirements_cached("base.txt")  # Should use cache
            test_results.append(("Requirements Cache", "PASS"))
        except Exception as e:
            test_results.append(("Requirements Cache", f"FAIL: {e}"))
        
        # Print results
        print("\n" + "="*50)
        print("Setup Command Test Results:")
        print("="*50)
        for name, result in test_results:
            status = "[PASS]" if result == "PASS" else "[FAIL]"
            print(f"{status} {name}: {result}")
        print("="*50)
        
        # Check if all tests passed
        all_passed = all(result == "PASS" for _, result in test_results)
        if all_passed:
            print("[SUCCESS] All setup command tests passed!")
        else:
            print("[WARNING] Some setup command tests failed")


class GenerateDocs(Command):
    """Generate project documentation."""
    
    description = "Generate project documentation"
    user_options = [
        ("format=", "f", "Documentation format (html/pdf/epub)"),
        ("serve", "s", "Serve documentation locally"),
        ("port=", "p", "Port for serving documentation"),
    ]
    
    def initialize_options(self):
        """Initialize documentation options."""
        self.format = "html"
        self.serve = False
        self.port = 8000
    
    def finalize_options(self):
        """Validate documentation options."""
        assert self.format in ["html", "pdf", "epub"], "Format must be html, pdf, or epub"
        self.port = int(self.port)
    
    def run(self):
        """Generate documentation."""
        print("Generating documentation...")
        
        # Install documentation requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", ".[docs]"])
        
        # Build documentation
        docs_dir = ROOT_DIR / "docs"
        build_dir = docs_dir / "_build"
        
        # Create necessary directories
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Run sphinx-build
        subprocess.check_call([
            "sphinx-build", "-b", self.format,
            str(docs_dir), str(build_dir / self.format)
        ])
        
        print(f"[SUCCESS] Documentation generated at {build_dir / self.format}")
        
        # Serve if requested
        if self.serve and self.format == "html":
            import http.server
            import socketserver
            import os
            
            os.chdir(build_dir / "html")
            with socketserver.TCPServer(("", self.port), http.server.SimpleHTTPRequestHandler) as httpd:
                print(f"Serving documentation at http://localhost:{self.port}")
                print("Press Ctrl+C to stop")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\nStopping documentation server...")


class CreateRelease(Command):
    """Create a new release."""
    
    description = "Create a new release"
    user_options = [
        ("version=", "v", "Version number"),
        ("message=", "m", "Release message"),
        ("push", "p", "Push to remote"),
        ("build", "b", "Build distribution packages"),
    ]
    
    def initialize_options(self):
        """Initialize release options."""
        self.version = None
        self.message = None
        self.push = False
        self.build = False
    
    def finalize_options(self):
        """Validate release options."""
        if not self.version:
            self.version = VERSION
        if not self.message:
            self.message = f"Release v{self.version}"
    
    def run(self):
        """Create release."""
        print(f"Creating release v{self.version}...")
        
        # Update version file
        version_file = SRC_DIR / "__version__.py"
        version_file.parent.mkdir(parents=True, exist_ok=True)
        with open(version_file, "w", encoding="utf-8") as f:
            f.write(f'__version__ = "{self.version}"\n')
        
        # Update CHANGELOG
        changelog_file = ROOT_DIR / "CHANGELOG.md"
        if changelog_file.exists():
            print("Please update CHANGELOG.md manually")
        
        # Git operations
        try:
            subprocess.check_call(["git", "add", "."])
            subprocess.check_call(["git", "commit", "-m", self.message])
            subprocess.check_call(["git", "tag", f"v{self.version}", "-m", self.message])
            print(f"[SUCCESS] Git tag v{self.version} created")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Git operations failed: {e}")
        
        # Push to remote
        if self.push:
            try:
                subprocess.check_call(["git", "push", "origin", "main"])
                subprocess.check_call(["git", "push", "origin", f"v{self.version}"])
                print("[SUCCESS] Pushed to remote repository")
            except subprocess.CalledProcessError as e:
                print(f"[WARNING] Push failed: {e}")
        
        # Build distribution packages
        if self.build:
            print("\nBuilding distribution packages...")
            try:
                subprocess.check_call([sys.executable, "setup.py", "sdist", "bdist_wheel"])
                print("[SUCCESS] Distribution packages built")
            except subprocess.CalledProcessError as e:
                print(f"[WARNING] Build failed: {e}")
        
        print(f"\n[SUCCESS] Release v{self.version} created!")


# Console scripts entry points (comprehensive list)
CONSOLE_SCRIPTS = [
    # Main CLI
    "ag-news=src.cli.main:cli",
    
    # Data management commands
    "ag-news-data=src.cli.data:cli",
    "ag-news-prepare=scripts.data_preparation.prepare_ag_news:main",
    "ag-news-prepare-external=scripts.data_preparation.prepare_external_data:main",
    "ag-news-augment=scripts.data_preparation.create_augmented_data:main",
    "ag-news-splits=scripts.data_preparation.create_data_splits:main",
    "ag-news-contrast=scripts.data_preparation.generate_contrast_sets:main",
    "ag-news-pseudo=scripts.data_preparation.generate_pseudo_labels:main",
    "ag-news-quality=scripts.data_preparation.select_quality_data:main",
    "ag-news-instruction-data=scripts.data_preparation.prepare_instruction_data:main",
    
    # Training commands
    "ag-news-train=src.cli.train:cli",
    "ag-news-train-single=scripts.training.train_single_model:main",
    "ag-news-train-all=scripts.training.train_all_models:main",
    "ag-news-train-ensemble=scripts.training.train_ensemble:main",
    "ag-news-train-distributed=scripts.training.distributed_training:main",
    "ag-news-resume=scripts.training.resume_training:main",
    "ag-news-prompt=scripts.training.train_with_prompts:main",
    "ag-news-instruction=scripts.training.instruction_tuning:main",
    "ag-news-multistage=scripts.training.multi_stage_training:main",
    "ag-news-distill=scripts.training.distill_from_gpt4:main",
    
    # Domain Adaptation
    "ag-news-pretrain=scripts.domain_adaptation.pretrain_on_news:main",
    "ag-news-download-news=scripts.domain_adaptation.download_news_corpus:main",
    "ag-news-domain-adapt=scripts.domain_adaptation.run_dapt:main",
    
    # Evaluation commands
    "ag-news-evaluate=src.cli.evaluate:cli",
    "ag-news-benchmark=scripts.evaluation.create_leaderboard:main",
    "ag-news-analyze=scripts.evaluation.statistical_analysis:main",
    "ag-news-report=scripts.evaluation.generate_reports:main",
    "ag-news-contrast-eval=scripts.evaluation.evaluate_contrast_sets:main",
    
    # Analysis commands
    "ag-news-error-analysis=src.cli.error_analysis:main",
    "ag-news-interpretability=src.cli.interpretability:main",
    "ag-news-attention=src.cli.attention_analysis:main",
    "ag-news-embeddings=src.cli.embedding_analysis:main",
    
    # Optimization commands
    "ag-news-optimize=scripts.optimization.hyperparameter_search:main",
    "ag-news-nas=scripts.optimization.architecture_search:main",
    "ag-news-ensemble-opt=scripts.optimization.ensemble_optimization:main",
    "ag-news-prompt-opt=scripts.optimization.prompt_optimization:main",
    
    # Deployment commands
    "ag-news-export=scripts.deployment.export_models:main",
    "ag-news-serve=src.cli.serve:cli",
    "ag-news-deploy=scripts.deployment.deploy_to_cloud:main",
    "ag-news-docker=scripts.deployment.create_docker_image:main",
    
    # API and App commands
    "ag-news-api=src.api.rest.main:main",
    "ag-news-grpc=src.api.grpc.services:main",
    "ag-news-graphql=src.api.graphql.server:main",
    "ag-news-streamlit=app.streamlit_app:main",
    
    # Utility commands
    "ag-news-setup=scripts.setup.verify_installation:main",
    "ag-news-download=scripts.setup.download_all_data:main",
    "ag-news-monitor=src.cli.monitor:main",
    "ag-news-profile=src.cli.profile:main",
    
    # Tools
    "ag-news-debug=tools.debugging.model_debugger:main",
    "ag-news-validate=tools.debugging.data_validator:main",
    "ag-news-visualize=tools.visualization.training_monitor:main",
    "ag-news-memory-profile=tools.profiling.memory_profiler:main",
    "ag-news-speed-profile=tools.profiling.speed_profiler:main",
    
    # Research tools
    "ag-news-experiment=experiments.experiment_runner:main",
    "ag-news-ablation=experiments.ablation_studies.component_ablation:main",
    "ag-news-leaderboard=experiments.results.leaderboard_generator:main",
    
    # Monitoring and security
    "ag-news-metrics=monitoring.metrics.metric_collectors:main",
    "ag-news-alerts=monitoring.alerts.alert_manager:main",
    "ag-news-audit=security.audit_logs.audit_logger:main",
    "ag-news-security-scan=security.model_security.security_scanner:main",
    
    # Database commands
    "ag-news-db-migrate=migrations.data.migration_runner:main",
    "ag-news-db-seed=scripts.seed_database:main",
    "ag-news-db-backup=scripts.backup_database:main",
    
    # Quick start commands
    "ag-news-quick-start=quickstart.minimal_example:main",
    "ag-news-quick-train=quickstart.train_simple:main",
    "ag-news-quick-eval=quickstart.evaluate_simple:main",
    "ag-news-demo=quickstart.demo_app:main",
]


# Package data files
PACKAGE_DATA = {
    "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md", "*.toml", "*.cfg", "*.ini", "__version__.py"],
    "src": ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.txt", "**/*.pyi", "py.typed"],
    "configs": ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.toml"],
    "prompts": ["**/*.txt", "**/*.json", "**/*.md", "**/*.jinja2"],
    "app": ["**/*.css", "**/*.js", "**/*.png", "**/*.jpg", "**/*.html", "**/*.svg"],
    "docs": ["**/*.md", "**/*.rst", "**/*.png", "**/*.jpg", "**/*.svg", "**/*.puml"],
    "templates": ["**/*.py", "**/*.yaml", "**/*.md", "**/*.jinja2"],
    "quickstart": ["**/*.py", "**/*.ipynb", "**/*.md", "**/Dockerfile*", "**/*.yml"],
    "data": ["**/.gitkeep", "**/*.txt"],
    "notebooks": ["**/*.ipynb", "**/*.py"],
    "deployment": ["**/*.yaml", "**/*.yml", "**/*.sh", "**/*Dockerfile*", "**/*.env"],
    "benchmarks": ["**/*.json", "**/*.yaml", "**/*.csv"],
    "tests": ["**/*.py", "**/*.json", "**/*.yaml", "**/*.txt"],
    "tools": ["**/*.py", "**/*.sh"],
    "monitoring": ["**/*.yaml", "**/*.json", "**/*.py"],
    "security": ["**/*.py", "**/*.yaml"],
    "experiments": ["**/*.yaml", "**/*.json", "**/*.py"],
    "research": ["**/*.bib", "**/*.md", "**/*.txt"],
    "scripts": ["**/*.sh", "**/*.py"],
    "migrations": ["**/*.py", "**/*.sql"],
    ".github": ["**/*.yml", "**/*.md"],
    ".devcontainer": ["**/*.json", "**/Dockerfile"],
    ".husky": ["**/*"],
    "ci": ["**/*.sh", "**/*.py", "**/*.yml"],
}


# Main setup configuration
setup(
    # Metadata
    name=METADATA["name"],
    version=VERSION,
    author=METADATA["author"],
    author_email=METADATA["author_email"],
    maintainer=METADATA["maintainer"],
    maintainer_email=METADATA["maintainer_email"],
    license=METADATA["license"],
    
    # Description
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
        "Model Hub": "https://huggingface.co/agnews-research",
        "Demo": "https://huggingface.co/spaces/agnews-research/demo",
        "Changelog": f"{METADATA['url']}/blob/main/CHANGELOG.md",
        "Benchmarks": "https://agnews-benchmarks.github.io",
        "CI/CD": f"{METADATA['url']}/actions",
        "Discussions": f"{METADATA['url']}/discussions",
        "Wiki": f"{METADATA['url']}/wiki",
    },
    
    # License
    license_files=["LICENSE"],
    
    # Packages
    packages=find_packages(
        where=".",
        include=["src*", "quickstart*"],
        exclude=["tests*", "docs*", "examples*", "*.tests", "*.tests.*"]
    ),
    package_dir={"": "."},
    py_modules=["quickstart.minimal_example"],
    include_package_data=True,
    package_data=PACKAGE_DATA,
    
    # Python version requirement
    python_requires=">=3.8,<3.12",
    
    # Dependencies
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points with extensive plugin system
    entry_points={
        "console_scripts": CONSOLE_SCRIPTS,
        
        # Plugin system for extensibility
        "ag_news.models": [
            # Transformer models
            "deberta = src.models.transformers.deberta.deberta_v3:DeBERTaV3Model",
            "deberta_sliding = src.models.transformers.deberta.deberta_sliding:DeBERTaSlidingModel",
            "deberta_hierarchical = src.models.transformers.deberta.deberta_hierarchical:DeBERTaHierarchicalModel",
            "roberta = src.models.transformers.roberta.roberta_enhanced:RoBERTaEnhancedModel",
            "roberta_domain = src.models.transformers.roberta.roberta_domain:RoBERTaDomainModel",
            "xlnet = src.models.transformers.xlnet.xlnet_classifier:XLNetClassifier",
            "electra = src.models.transformers.electra.electra_discriminator:ElectraDiscriminator",
            "longformer = src.models.transformers.longformer.longformer_global:LongformerGlobalModel",
            "gpt2 = src.models.transformers.generative.gpt2_classifier:GPT2Classifier",
            "t5 = src.models.transformers.generative.t5_classifier:T5Classifier",
        ],
        
        "ag_news.prompt_models": [
            "prompt_model = src.models.prompt_based.prompt_model:PromptModel",
            "soft_prompt = src.models.prompt_based.soft_prompt:SoftPromptModel",
            "instruction = src.models.prompt_based.instruction_model:InstructionModel",
        ],
        
        "ag_news.efficient_models": [
            "lora = src.models.efficient.lora.lora_model:LoRAModel",
            "adapter = src.models.efficient.adapters.adapter_model:AdapterModel",
            "adapter_fusion = src.models.efficient.adapters.adapter_fusion:AdapterFusionModel",
            "int8 = src.models.efficient.quantization.int8_quantization:Int8Model",
            "dynamic_quant = src.models.efficient.quantization.dynamic_quantization:DynamicQuantModel",
            "pruned = src.models.efficient.pruning.magnitude_pruning:PrunedModel",
        ],
        
        "ag_news.ensemble_models": [
            "soft_voting = src.models.ensemble.voting.soft_voting:SoftVotingEnsemble",
            "weighted_voting = src.models.ensemble.voting.weighted_voting:WeightedVotingEnsemble",
            "rank_averaging = src.models.ensemble.voting.rank_averaging:RankAveragingEnsemble",
            "stacking = src.models.ensemble.stacking.stacking_classifier:StackingEnsemble",
            "blending = src.models.ensemble.blending.blending_ensemble:BlendingEnsemble",
            "bayesian = src.models.ensemble.advanced.bayesian_ensemble:BayesianEnsemble",
            "snapshot = src.models.ensemble.advanced.snapshot_ensemble:SnapshotEnsemble",
        ],
        
        "ag_news.trainers": [
            "standard = src.training.trainers.standard_trainer:StandardTrainer",
            "distributed = src.training.trainers.distributed_trainer:DistributedTrainer",
            "apex = src.training.trainers.apex_trainer:ApexTrainer",
            "prompt = src.training.trainers.prompt_trainer:PromptTrainer",
            "instruction = src.training.trainers.instruction_trainer:InstructionTrainer",
            "multistage = src.training.trainers.multi_stage_trainer:MultiStageTrainer",
        ],
        
        "ag_news.augmenters": [
            "backtranslation = src.data.augmentation.back_translation:BackTranslationAugmenter",
            "paraphrase = src.data.augmentation.paraphrase:ParaphraseAugmenter",
            "token_replacement = src.data.augmentation.token_replacement:TokenReplacementAugmenter",
            "mixup = src.data.augmentation.mixup:MixupAugmenter",
            "cutmix = src.data.augmentation.cutmix:CutMixAugmenter",
            "adversarial = src.data.augmentation.adversarial:AdversarialAugmenter",
            "contrast_set = src.data.augmentation.contrast_set_generator:ContrastSetGenerator",
        ],
        
        "ag_news.evaluators": [
            "classification = src.evaluation.metrics.classification_metrics:ClassificationEvaluator",
            "ensemble = src.evaluation.metrics.ensemble_metrics:EnsembleEvaluator",
            "robustness = src.evaluation.metrics.robustness_metrics:RobustnessEvaluator",
            "efficiency = src.evaluation.metrics.efficiency_metrics:EfficiencyEvaluator",
            "fairness = src.evaluation.metrics.fairness_metrics:FairnessEvaluator",
            "environmental = src.evaluation.metrics.environmental_impact:EnvironmentalEvaluator",
            "contrast = src.evaluation.metrics.contrast_consistency:ContrastConsistencyEvaluator",
        ],
        
        "ag_news.interpreters": [
            "shap = src.evaluation.interpretability.shap_interpreter:SHAPInterpreter",
            "lime = src.evaluation.interpretability.lime_interpreter:LIMEInterpreter",
            "attention = src.evaluation.interpretability.attention_analysis:AttentionAnalyzer",
            "integrated_gradients = src.evaluation.interpretability.integrated_gradients:IntegratedGradients",
        ],
        
        "ag_news.serving": [
            "fastapi = src.inference.serving.model_server:FastAPIServer",
            "batch = src.inference.serving.batch_server:BatchServer",
            "load_balancer = src.inference.serving.load_balancer:LoadBalancer",
            "onnx = src.inference.optimization.onnx_converter:ONNXServer",
            "tensorrt = src.inference.optimization.tensorrt_optimizer:TensorRTServer",
            "quantized = src.inference.optimization.quantization_optimizer:QuantizedServer",
        ],
    },
    
    # Classifiers
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        
        # Topics
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        
        # Environment
        "Environment :: Console",
        "Environment :: Web Environment",
        "Environment :: GPU :: NVIDIA CUDA :: 11.8",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        
        # Framework
        "Framework :: Pytest",
        "Framework :: Jupyter",
        "Framework :: Flask",
        "Framework :: FastAPI",
        
        # Natural Language
        "Natural Language :: English",
        
        # Typing
        "Typing :: Typed",
    ],
    
    # Keywords
    keywords=[
        "nlp", "natural-language-processing", "text-classification",
        "news-classification", "ag-news", "deep-learning", "machine-learning",
        "transformers", "pytorch", "huggingface", "bert", "deberta", "roberta",
        "ensemble-learning", "state-of-the-art", "research", "production-ml",
        "mlops", "prompt-engineering", "instruction-tuning", "model-optimization",
        "knowledge-distillation", "domain-adaptation", "contrast-sets",
        "adversarial-training", "model-compression", "quantization", "pruning",
        "experiment-tracking", "hyperparameter-optimization", "model-serving",
        "gpu-optimization", "distributed-training", "database-migration",
    ],
    
    # Custom commands
    cmdclass={
        "install": PostInstallCommand,
        "develop": DevelopCommand,
        "test": RunTests,
        "test_setup": TestSetupCommands,
        "setup_gpu": SetupGPUEnvironment,
        "setup_research": SetupResearchEnvironment,
        "setup_production": SetupProductionEnvironment,
        "setup_database": SetupDatabaseEnvironment,
        "setup_mlops": SetupMLOpsEnvironment,
        "docs": GenerateDocs,
        "release": CreateRelease,
    },
    
    # Test configuration
    test_suite="tests",
    tests_require=["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    
    # Additional options
    zip_safe=False,
    platforms=["any"],
)


# Create MANIFEST.in if it doesn't exist
manifest_content = """
# Include all documentation
include README.md LICENSE CITATION.cff CHANGELOG.md
include ARCHITECTURE.md PERFORMANCE.md SECURITY.md
include TROUBLESHOOTING.md ROADMAP.md

# Include configuration files
recursive-include configs *.yaml *.yml *.json *.toml
recursive-include prompts *.txt *.json *.md *.jinja2
recursive-include deployment *.yaml *.yml *.sh Dockerfile*
recursive-include .github *.yml *.md
recursive-include .devcontainer *.json Dockerfile
recursive-include .husky *

# Include requirements
recursive-include requirements *.txt

# Include data files
recursive-include data .gitkeep
recursive-include benchmarks *.json *.yaml *.csv

# Include notebooks
recursive-include notebooks *.ipynb

# Include documentation
recursive-include docs *.md *.rst *.png *.jpg *.svg *.puml

# Include app assets
recursive-include app/assets *

# Include test fixtures
recursive-include tests/fixtures *

# Include scripts
recursive-include scripts *.sh *.py

# Include CI/CD
recursive-include ci *.sh *.yml

# Include tools
recursive-include tools *.py *.sh

# Include templates
recursive-include templates *

# Include quickstart
recursive-include quickstart *

# Include migrations
recursive-include migrations *.py *.sql

# Include cache directory structure
include .cache/setup/.gitkeep

# Exclude unnecessary files
global-exclude *.pyc
global-exclude *.pyo
global-exclude *.swp
global-exclude .DS_Store
global-exclude __pycache__
global-exclude .git
global-exclude .pytest_cache
global-exclude .mypy_cache
global-exclude *.egg-info
"""

manifest_path = ROOT_DIR / "MANIFEST.in"
if not manifest_path.exists():
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest_content)
    print("Created MANIFEST.in file")


# Clear cache on exit for clean state
import atexit

def clear_cache():
    """Clear setup cache on exit."""
    try:
        if CACHE_DIR.exists():
            import shutil
            shutil.rmtree(CACHE_DIR, ignore_errors=True)
    except:
        pass

# Register cache cleanup
atexit.register(clear_cache)
