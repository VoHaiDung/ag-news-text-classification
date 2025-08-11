import os
import sys
import json
import shutil
import subprocess
import asyncio
import aiohttp
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict

from setuptools import setup, find_packages, Command
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.test import test as TestCommand


# Project root directory
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
REQUIREMENTS_DIR = ROOT_DIR / "requirements"

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

# Version tracking
VERSION_FILE = ROOT_DIR / ".versions.json"
COMPATIBILITY_MATRIX = ROOT_DIR / "compatibility_matrix.json"


# Version tracking system
@dataclass
class VersionInfo:
    """Complete version information for all components."""
    framework_version: str
    api_version: str
    db_schema_version: str
    dataset_version: str
    model_version: str
    config_version: str
    compatibility_hash: str
    
    def to_dict(self) -> Dict[str, str]:
        return asdict(self)
    
    def is_compatible_with(self, other: 'VersionInfo') -> bool:
        """Check compatibility with another version."""
        # Check major version compatibility
        def get_major(version: str) -> int:
            return int(version.split(".")[0])
        
        return (
            get_major(self.api_version) == get_major(other.api_version) and
            get_major(self.db_schema_version) == get_major(other.db_schema_version) and
            get_major(self.dataset_version) == get_major(other.dataset_version)
        )


def load_version_info() -> VersionInfo:
    """Load version information from file or create default."""
    if VERSION_FILE.exists():
        with open(VERSION_FILE, "r") as f:
            data = json.load(f)
            return VersionInfo(**data)
    
    # Default versions
    return VersionInfo(
        framework_version="1.0.0",
        api_version="1.0.0",
        db_schema_version="1.0.0",
        dataset_version="1.0.0",
        model_version="1.0.0",
        config_version="1.0.0",
        compatibility_hash=generate_compatibility_hash()
    )


def save_version_info(info: VersionInfo):
    """Save version information to file."""
    with open(VERSION_FILE, "w") as f:
        json.dump(info.to_dict(), f, indent=2)


def generate_compatibility_hash() -> str:
    """Generate hash for compatibility checking."""
    info = load_version_info()
    content = f"{info.api_version}:{info.db_schema_version}:{info.dataset_version}"
    return hashlib.sha256(content.encode()).hexdigest()[:8]


def check_compatibility_matrix() -> Dict[str, Any]:
    """Load and check compatibility matrix."""
    if COMPATIBILITY_MATRIX.exists():
        with open(COMPATIBILITY_MATRIX, "r") as f:
            return json.load(f)
    
    # Default compatibility matrix
    return {
        "dataset_versions": {
            "1.0.0": ["transformers>=4.35.0", "datasets>=2.14.0"],
            "2.0.0": ["transformers>=4.40.0", "datasets>=2.15.0"],
        },
        "api_versions": {
            "1.0.0": ["fastapi>=0.104.0", "pydantic>=2.4.0"],
            "2.0.0": ["fastapi>=0.105.0", "pydantic>=2.5.0"],
        },
        "db_schema_versions": {
            "1.0.0": ["sqlalchemy>=2.0.0", "alembic>=1.12.0"],
            "2.0.0": ["sqlalchemy>=2.1.0", "alembic>=1.13.0"],
        }
    }


# Async operations
async def run_command_async(cmd: List[str], **kwargs) -> Tuple[int, str, str]:
    """Run command asynchronously."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode()


async def download_file_async(session: aiohttp.ClientSession, url: str, dest: Path) -> bool:
    """Download file asynchronously."""
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            content = await response.read()
            
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(content)
            return True
    except Exception as e:
        print(f"[WARNING] Failed to download {url}: {e}")
        return False


async def parallel_pip_install(packages: List[str]) -> List[bool]:
    """Install multiple packages in parallel."""
    tasks = []
    for package in packages:
        cmd = [sys.executable, "-m", "pip", "install", package]
        tasks.append(run_command_async(cmd))
    
    results = await asyncio.gather(*tasks)
    return [r[0] == 0 for r in results]


async def download_models_async(models: List[str], cache_dir: Path) -> List[bool]:
    """Download multiple models asynchronously."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for model in models:
            # This is simplified - actual implementation would use HuggingFace API
            url = f"https://huggingface.co/{model}/resolve/main/config.json"
            dest = cache_dir / model.replace("/", "_") / "config.json"
            tasks.append(download_file_async(session, url, dest))
        
        return await asyncio.gather(*tasks)


# Dataset version management
class DatasetVersionManager:
    """Manage dataset versions and compatibility."""
    
    def __init__(self):
        self.versions_file = ROOT_DIR / "data" / ".dataset_versions.json"
        self.load_versions()
    
    def load_versions(self):
        """Load dataset version information."""
        if self.versions_file.exists():
            with open(self.versions_file, "r") as f:
                self.versions = json.load(f)
        else:
            self.versions = {
                "ag_news": {
                    "current": "1.0.0",
                    "available": ["1.0.0", "2.0.0"],
                    "compatibility": {
                        "1.0.0": {"min_transformers": "4.35.0", "max_transformers": "4.40.0"},
                        "2.0.0": {"min_transformers": "4.40.0", "max_transformers": "5.0.0"},
                    }
                },
                "external_news": {
                    "current": "1.0.0",
                    "available": ["1.0.0"],
                    "compatibility": {
                        "1.0.0": {"min_pandas": "2.0.0", "max_pandas": "2.2.0"},
                    }
                }
            }
    
    def save_versions(self):
        """Save dataset version information."""
        self.versions_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.versions_file, "w") as f:
            json.dump(self.versions, f, indent=2)
    
    def check_compatibility(self, dataset: str, version: str) -> bool:
        """Check if dataset version is compatible with current environment."""
        if dataset not in self.versions:
            return False
        
        if version not in self.versions[dataset]["available"]:
            return False
        
        compat = self.versions[dataset]["compatibility"].get(version, {})
        
        # Check transformers version if required
        if "min_transformers" in compat:
            try:
                import transformers
                current = transformers.__version__
                min_ver = compat["min_transformers"]
                max_ver = compat.get("max_transformers", "999.0.0")
                
                if not (min_ver <= current <= max_ver):
                    print(f"[WARNING] transformers {current} may not be compatible with {dataset} v{version}")
                    return False
            except ImportError:
                return False
        
        return True
    
    def migrate_dataset(self, dataset: str, from_version: str, to_version: str):
        """Migrate dataset from one version to another."""
        print(f"[INFO] Migrating {dataset} from v{from_version} to v{to_version}")
        
        # Migration logic would go here
        migration_scripts = {
            ("ag_news", "1.0.0", "2.0.0"): "scripts/migrations/ag_news_1_to_2.py",
        }
        
        migration_key = (dataset, from_version, to_version)
        if migration_key in migration_scripts:
            script_path = ROOT_DIR / migration_scripts[migration_key]
            if script_path.exists():
                subprocess.check_call([sys.executable, str(script_path)])
                
                # Update current version
                self.versions[dataset]["current"] = to_version
                self.save_versions()
                
                print(f"[SUCCESS] Migration complete")
                return True
        
        print(f"[WARNING] No migration path from v{from_version} to v{to_version}")
        return False


# Database version management
class DatabaseVersionManager:
    """Manage database schema versions."""
    
    def __init__(self):
        self.db_version_file = ROOT_DIR / "migrations" / ".db_version"
        self.current_version = self.get_current_version()
    
    def get_current_version(self) -> str:
        """Get current database schema version."""
        if self.db_version_file.exists():
            with open(self.db_version_file, "r") as f:
                return f.read().strip()
        return "1.0.0"
    
    def set_version(self, version: str):
        """Set database schema version."""
        self.db_version_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_version_file, "w") as f:
            f.write(version)
        self.current_version = version
    
    async def run_migrations_async(self, target_version: Optional[str] = None):
        """Run database migrations asynchronously."""
        print(f"[INFO] Running database migrations from v{self.current_version}")
        
        # Use alembic for migrations
        cmd = ["alembic", "upgrade", target_version or "head"]
        returncode, stdout, stderr = await run_command_async(cmd)
        
        if returncode == 0:
            print("[SUCCESS] Database migrations completed")
            if target_version:
                self.set_version(target_version)
            return True
        else:
            print(f"[ERROR] Migration failed: {stderr}")
            return False
    
    def check_compatibility(self, api_version: str) -> bool:
        """Check if database version is compatible with API version."""
        compatibility = {
            "1.0.0": ["1.0.0", "1.1.0"],  # DB 1.0.0 compatible with API 1.0.0 and 1.1.0
            "2.0.0": ["2.0.0", "2.1.0"],
        }
        
        compatible_apis = compatibility.get(self.current_version, [])
        return api_version in compatible_apis


# Improved version functions
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


# Improved requirements reading with caching
@lru_cache(maxsize=32)
def read_requirements_cached(filename: str) -> List[str]:
    """Read requirements with caching."""
    return read_requirements(filename, REQUIREMENTS_DIR)


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
        "aiohttp>=3.9.0",  # Added for async operations
        "asyncio>=3.4.3",  # Added for async operations
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


# Improved custom commands with async
class PostInstallCommand(install):
    """Post-installation for installation mode."""
    
    def run(self):
        """Run standard install and post-install tasks."""
        install.run(self)
        self.execute(self.post_install, [], msg="Running post-installation tasks...")
    
    def post_install(self):
        """Execute post-installation tasks."""
        # Check version compatibility
        version_info = load_version_info()
        print(f"\n[INFO] Framework version: {version_info.framework_version}")
        print(f"[INFO] API version: {version_info.api_version}")
        print(f"[INFO] Database schema: {version_info.db_schema_version}")
        print(f"[INFO] Dataset version: {version_info.dataset_version}")
        
        # Check compatibility
        compat_matrix = check_compatibility_matrix()
        dataset_version = version_info.dataset_version
        if dataset_version in compat_matrix.get("dataset_versions", {}):
            required_deps = compat_matrix["dataset_versions"][dataset_version]
            print(f"[INFO] Dataset v{dataset_version} requires: {', '.join(required_deps)}")
        
        print("\n" + "="*80)
        print("AG News Text Classification Framework Installation Complete!")
        print("="*80)
        self.print_next_steps()
    
    def print_next_steps(self):
        """Print next steps for user."""
        print("\nQuick Start Guide:")
        print("  1. Verify installation: ag-news --version")
        print("  2. Check compatibility: ag-news check-compatibility")
        print("  3. Setup environment: ag-news setup --env research")
        print("  4. Download data: ag-news data download --version latest")
        print("  5. Train baseline: ag-news train --config configs/models/single/deberta_v3_xlarge.yaml")
        print("  6. Evaluate: ag-news evaluate --model outputs/models/fine_tuned/best_model.pt")
        
        print("\nEnvironment Setup Commands:")
        print("  - Research: pip install -e '.[research]'")
        print("  - Production: pip install '.[prod]'")
        print("  - Development: pip install -e '.[dev]'")
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
            "migrations/data",
            "migrations/models",
        ]
        
        for dir_path in directories_to_create:
            Path(ROOT_DIR / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize version tracking
        version_info = load_version_info()
        save_version_info(version_info)
        
        # Initialize dataset version manager
        dataset_manager = DatasetVersionManager()
        dataset_manager.save_versions()
        
        # Initialize database version manager
        db_manager = DatabaseVersionManager()
        db_manager.set_version(db_manager.current_version)
        
        print("[SUCCESS] Development environment ready!")


class SetupResearchEnvironmentAsync(Command):
    """Setup complete research environment with async operations."""
    
    description = "Setup research environment with all ML/DL tools"
    user_options = [
        ("download-models", "d", "Download pretrained models"),
        ("setup-wandb", "w", "Setup Weights & Biases"),
        ("cuda-version=", "c", "CUDA version (11.8 or 12.0)"),
        ("dataset-version=", "v", "Dataset version to use"),
    ]
    
    def initialize_options(self):
        """Initialize command options."""
        self.download_models = False
        self.setup_wandb = False
        self.cuda_version = "11.8"
        self.dataset_version = "1.0.0"
    
    def finalize_options(self):
        """Validate command options."""
        assert self.cuda_version in ["11.8", "12.0", "12.1"], "CUDA version must be 11.8, 12.0, or 12.1"
    
    def run(self):
        """Setup research environment using async operations."""
        print("Setting up research environment...")
        
        # Run async setup
        asyncio.run(self.async_setup())
        
        print("\n[SUCCESS] Research environment setup complete!")
    
    async def async_setup(self):
        """Async setup operations."""
        # Install research requirements asynchronously
        print("Installing research requirements...")
        packages = read_requirements_cached("research.txt") if REQUIREMENTS_DIR.exists() else []
        
        if packages:
            # Install in batches for better performance
            batch_size = 5
            for i in range(0, len(packages), batch_size):
                batch = packages[i:i+batch_size]
                results = await parallel_pip_install(batch)
                successful = sum(results)
                print(f"[INFO] Installed {successful}/{len(batch)} packages in batch {i//batch_size + 1}")
        
        # Download spaCy models asynchronously
        print("\nDownloading spaCy language models...")
        spacy_models = ["en_core_web_sm", "en_core_web_lg", "en_core_web_trf"]
        spacy_tasks = []
        for model in spacy_models:
            cmd = [sys.executable, "-m", "spacy", "download", model]
            spacy_tasks.append(run_command_async(cmd))
        
        spacy_results = await asyncio.gather(*spacy_tasks)
        for model, (returncode, _, _) in zip(spacy_models, spacy_results):
            if returncode == 0:
                print(f"[SUCCESS] Downloaded {model}")
            else:
                print(f"[WARNING] Could not download {model}")
        
        # Download NLTK data
        print("\nDownloading NLTK data...")
        try:
            import nltk
            nltk_packages = [
                'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                'maxent_ne_chunker', 'words', 'vader_lexicon', 'opinion_lexicon',
                'pros_cons', 'reuters', 'brown', 'conll2000', 'movie_reviews'
            ]
            # NLTK downloads are not async-friendly, so we do them sequentially
            for package in nltk_packages:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass
            print("[SUCCESS] NLTK data downloaded")
        except ImportError:
            print("[WARNING] NLTK not installed")
        
        # Setup Jupyter extensions asynchronously
        print("\nSetting up Jupyter extensions...")
        jupyter_extensions = [
            "jupyter_contrib_nbextensions",
            "jupyter_nbextensions_configurator",
            "jupyterlab-git",
            "jupyterlab-lsp",
        ]
        jupyter_results = await parallel_pip_install(jupyter_extensions)
        if all(jupyter_results):
            print("[SUCCESS] Jupyter extensions configured")
        else:
            print("[WARNING] Some Jupyter extensions failed to install")
        
        # Setup experiment tracking
        if self.setup_wandb:
            print("\nSetting up Weights & Biases...")
            returncode, _, _ = await run_command_async([sys.executable, "-m", "wandb", "login", "--relogin"])
            if returncode == 0:
                print("[SUCCESS] Weights & Biases configured")
            else:
                print("[WARNING] Could not setup Weights & Biases")
        
        # Download pretrained models asynchronously
        if self.download_models:
            print("\nDownloading pretrained models...")
            models = [
                "microsoft/deberta-v3-large",
                "roberta-large",
                "xlnet-large-cased",
                "google/electra-large-discriminator",
            ]
            cache_dir = ROOT_DIR / "outputs/models/pretrained"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            results = await download_models_async(models, cache_dir)
            for model, success in zip(models, results):
                if success:
                    print(f"[SUCCESS] Downloaded {model}")
                else:
                    print(f"[WARNING] Could not download {model}")
        
        # Check dataset compatibility
        dataset_manager = DatasetVersionManager()
        if dataset_manager.check_compatibility("ag_news", self.dataset_version):
            print(f"[SUCCESS] Dataset version {self.dataset_version} is compatible")
        else:
            print(f"[WARNING] Dataset version {self.dataset_version} may have compatibility issues")


class SetupProductionEnvironmentAsync(Command):
    """Setup production environment with async operations."""
    
    description = "Setup production environment for deployment"
    user_options = [
        ("cloud=", "c", "Cloud provider (aws/gcp/azure)"),
        ("monitoring", "m", "Setup monitoring stack"),
        ("security", "s", "Setup security tools"),
        ("migrate-db", "d", "Run database migrations"),
        ("api-version=", "a", "API version to deploy"),
    ]
    
    def initialize_options(self):
        """Initialize command options."""
        self.cloud = None
        self.monitoring = False
        self.security = False
        self.migrate_db = False
        self.api_version = "1.0.0"
    
    def finalize_options(self):
        """Validate command options."""
        if self.cloud:
            assert self.cloud in ["aws", "gcp", "azure"], "Cloud must be aws, gcp, or azure"
    
    def run(self):
        """Setup production environment."""
        print("Setting up production environment...")
        
        # Run async setup
        asyncio.run(self.async_setup())
        
        print("\n[SUCCESS] Production environment setup complete!")
    
    async def async_setup(self):
        """Async production setup."""
        # Install production requirements
        print("Installing production requirements...")
        packages = read_requirements_cached("prod.txt") if REQUIREMENTS_DIR.exists() else []
        
        if packages:
            results = await parallel_pip_install(packages[:10])  # First 10 packages
            print(f"[INFO] Installed {sum(results)}/{len(results)} core production packages")
        
        # Install cloud-specific requirements
        if self.cloud:
            print(f"\nSetting up {self.cloud.upper()} tools...")
            cloud_packages = EXTRAS_REQUIRE.get(self.cloud, [])
            if cloud_packages:
                results = await parallel_pip_install(cloud_packages)
                if all(results):
                    print(f"[SUCCESS] {self.cloud.upper()} tools installed")
        
        # Setup monitoring asynchronously
        if self.monitoring:
            print("\nSetting up monitoring stack...")
            monitoring_tools = ["prometheus-client", "grafana-api", "evidently", "whylogs"]
            results = await parallel_pip_install(monitoring_tools)
            
            # Create monitoring directories
            monitoring_dirs = [
                "monitoring/dashboards/grafana",
                "monitoring/dashboards/prometheus",
                "monitoring/logs",
                "monitoring/metrics",
            ]
            for dir_path in monitoring_dirs:
                Path(ROOT_DIR / dir_path).mkdir(parents=True, exist_ok=True)
            
            if all(results):
                print("[SUCCESS] Monitoring stack configured")
        
        # Setup security asynchronously
        if self.security:
            print("\nSetting up security tools...")
            security_tools = ["bandit", "safety", "detect-secrets", "cryptography"]
            results = await parallel_pip_install(security_tools)
            
            # Create security directories
            security_dir = ROOT_DIR / "security/scan_results"
            security_dir.mkdir(parents=True, exist_ok=True)
            
            # Run initial security scan asynchronously
            returncode, stdout, stderr = await run_command_async(
                ["bandit", "-r", "src/", "-f", "json", "-o", "security/scan_results/initial_scan.json"]
            )
            if returncode == 0:
                print("[SUCCESS] Security scan complete")
            else:
                print("[WARNING] Security scan failed")
        
        # Run database migrations if requested
        if self.migrate_db:
            print("\nRunning database migrations...")
            db_manager = DatabaseVersionManager()
            
            # Check API compatibility
            if db_manager.check_compatibility(self.api_version):
                success = await db_manager.run_migrations_async()
                if success:
                    print("[SUCCESS] Database migrations completed")
            else:
                print(f"[WARNING] Database schema may not be compatible with API v{self.api_version}")
        
        # Update version info
        version_info = load_version_info()
        version_info.api_version = self.api_version
        save_version_info(version_info)
        
        # Optimize for production
        print("\nOptimizing for production...")
        returncode, _, _ = await run_command_async([sys.executable, "-m", "compileall", "src/"])
        if returncode == 0:
            print("[SUCCESS] Python files compiled")


class CheckCompatibility(Command):
    """Check system compatibility."""
    
    description = "Check version compatibility across all components"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        """Check compatibility."""
        print("Checking system compatibility...")
        print("="*80)
        
        # Load version info
        version_info = load_version_info()
        
        print("Current Versions:")
        print(f"  Framework: {version_info.framework_version}")
        print(f"  API: {version_info.api_version}")
        print(f"  Database Schema: {version_info.db_schema_version}")
        print(f"  Dataset: {version_info.dataset_version}")
        print(f"  Model: {version_info.model_version}")
        print(f"  Config: {version_info.config_version}")
        print(f"  Compatibility Hash: {version_info.compatibility_hash}")
        
        print("\nCompatibility Checks:")
        
        # Check dataset compatibility
        dataset_manager = DatasetVersionManager()
        for dataset in ["ag_news", "external_news"]:
            current = dataset_manager.versions.get(dataset, {}).get("current", "unknown")
            compatible = dataset_manager.check_compatibility(dataset, current)
            status = "[OK]" if compatible else "[WARNING]"
            print(f"  {status} Dataset '{dataset}' v{current}")
        
        # Check database compatibility
        db_manager = DatabaseVersionManager()
        db_compatible = db_manager.check_compatibility(version_info.api_version)
        status = "[OK]" if db_compatible else "[WARNING]"
        print(f"  {status} Database schema v{db_manager.current_version} with API v{version_info.api_version}")
        
        # Check Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        python_ok = sys.version_info >= (3, 8) and sys.version_info < (3, 12)
        status = "[OK]" if python_ok else "[WARNING]"
        print(f"  {status} Python {python_version}")
        
        # Check key dependencies
        try:
            import torch
            torch_version = torch.__version__
            torch_ok = torch_version >= "2.0.0" and torch_version < "2.2.0"
            status = "[OK]" if torch_ok else "[WARNING]"
            print(f"  {status} PyTorch {torch_version}")
        except ImportError:
            print("  [ERROR] PyTorch not installed")
        
        try:
            import transformers
            trans_version = transformers.__version__
            trans_ok = trans_version >= "4.35.0" and trans_version < "5.0.0"
            status = "[OK]" if trans_ok else "[WARNING]"
            print(f"  {status} Transformers {trans_version}")
        except ImportError:
            print("  [ERROR] Transformers not installed")
        
        print("="*80)
        
        # Generate new compatibility hash
        new_hash = generate_compatibility_hash()
        if new_hash != version_info.compatibility_hash:
            print(f"\n[INFO] Compatibility hash updated: {version_info.compatibility_hash} -> {new_hash}")
            version_info.compatibility_hash = new_hash
            save_version_info(version_info)


class MigrateDataset(Command):
    """Migrate dataset to different version."""
    
    description = "Migrate dataset to a different version"
    user_options = [
        ("dataset=", "d", "Dataset name"),
        ("from-version=", "f", "Source version"),
        ("to-version=", "t", "Target version"),
    ]
    
    def initialize_options(self):
        self.dataset = "ag_news"
        self.from_version = None
        self.to_version = None
    
    def finalize_options(self):
        if not self.from_version or not self.to_version:
            raise ValueError("Both from-version and to-version must be specified")
    
    def run(self):
        """Run dataset migration."""
        print(f"Migrating {self.dataset} from v{self.from_version} to v{self.to_version}...")
        
        dataset_manager = DatasetVersionManager()
        success = dataset_manager.migrate_dataset(
            self.dataset,
            self.from_version,
            self.to_version
        )
        
        if success:
            print("[SUCCESS] Migration completed successfully")
            
            # Update version info
            version_info = load_version_info()
            version_info.dataset_version = self.to_version
            save_version_info(version_info)
        else:
            print("[ERROR] Migration failed")


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
        ("test-type=", "t", "Type of tests to run (unit/integration/performance/all)"),
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
        ("update-all", "u", "Update all component versions"),
    ]
    
    def initialize_options(self):
        """Initialize release options."""
        self.version = None
        self.message = None
        self.push = False
        self.build = False
        self.update_all = False
    
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
        
        # Update all component versions if requested
        if self.update_all:
            version_info = load_version_info()
            version_info.framework_version = self.version
            version_info.api_version = self.version
            version_info.model_version = self.version
            version_info.config_version = self.version
            version_info.compatibility_hash = generate_compatibility_hash()
            save_version_info(version_info)
            print("[SUCCESS] All component versions updated")
        
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


# Console scripts entry points
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
    
    # Version and compatibility commands
    "ag-news-check-compatibility=src.cli.compatibility:check",
    "ag-news-migrate-dataset=src.cli.migrations:migrate_dataset",
    "ag-news-migrate-db=src.cli.migrations:migrate_database",
    "ag-news-version=src.cli.version:show",
    
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
    
    # Quick start commands
    "ag-news-quick-start=quickstart.minimal_example:main",
    "ag-news-quick-train=quickstart.train_simple:main",
    "ag-news-quick-eval=quickstart.evaluate_simple:main",
    "ag-news-demo=quickstart.demo_app:main",
]


# Package data files
PACKAGE_DATA = {
    "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md", "*.toml", "*.cfg", "*.ini", "__version__.py", ".versions.json", "compatibility_matrix.json"],
    "src": ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.txt", "**/*.pyi", "py.typed"],
    "configs": ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.toml"],
    "prompts": ["**/*.txt", "**/*.json", "**/*.md", "**/*.jinja2"],
    "app": ["**/*.css", "**/*.js", "**/*.png", "**/*.jpg", "**/*.html", "**/*.svg"],
    "docs": ["**/*.md", "**/*.rst", "**/*.png", "**/*.jpg", "**/*.svg", "**/*.puml"],
    "templates": ["**/*.py", "**/*.yaml", "**/*.md", "**/*.jinja2"],
    "quickstart": ["**/*.py", "**/*.ipynb", "**/*.md", "**/Dockerfile*", "**/*.yml"],
    "data": ["**/.gitkeep", "**/.dataset_versions.json"],
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
    "migrations": ["**/*.py", "**/*.sql", "**/.db_version"],
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
        "version-tracking", "compatibility-management", "dataset-versioning",
    ],
    
    # Custom commands
    cmdclass={
        "install": PostInstallCommand,
        "develop": DevelopCommand,
        "test": RunTests,
        "setup_research": SetupResearchEnvironmentAsync,
        "setup_production": SetupProductionEnvironmentAsync,
        "setup_mlops": SetupMLOpsEnvironment,
        "check_compatibility": CheckCompatibility,
        "migrate_dataset": MigrateDataset,
        "docs": GenerateDocs,
        "release": CreateRelease,
    },
    
    # Test configuration
    test_suite="tests",
    tests_require=["pytest>=7.4.0", "pytest-cov>=4.1.0", "pytest-asyncio>=0.21.0"],
    
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

# Include version tracking files
include .versions.json
include compatibility_matrix.json

# Include data files
recursive-include data .gitkeep .dataset_versions.json
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
recursive-include migrations *.py *.sql .db_version

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
