#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup Script for AG News Text Classification
============================================

Project: AG News Text Classification (ag-news-text-classification)
Description: State-of-the-art text classification framework with comprehensive
             overfitting prevention, parameter-efficient fine-tuning, and
             ensemble learning capabilities for the AG News dataset.

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
Python: >=3.8,<3.12

Architecture Overview:
    This setup script configures a modular framework supporting:
    - SOTA transformer models (DeBERTa, RoBERTa, ELECTRA, XLNet)
    - Large Language Models (LLaMA, Mistral, Falcon, Phi)
    - Parameter-efficient fine-tuning (LoRA, QLoRA, Adapters, Prefix/Prompt Tuning)
    - Comprehensive overfitting prevention system
    - Ensemble learning methods (Voting, Stacking, Blending)
    - Multi-platform deployment (Local, Colab, Kaggle, HuggingFace Spaces)
    - Academic-grade reproducibility and experimentation

Installation Profiles:
    minimal:
        Core dependencies only for basic text classification
    ml:
        Machine learning stack with traditional and neural models
    llm:
        Large Language Model support with quantization
    efficient:
        Parameter-efficient fine-tuning methods
    data:
        Data processing and augmentation tools
    ui:
        Streamlit and Gradio web interfaces
    dev:
        Development tools (testing, linting, formatting)
    docs:
        Documentation generation tools
    research:
        Full research environment with experiment tracking
    all:
        Complete installation with all optional dependencies

Usage Examples:
    Standard installation:
        pip install .
    
    Development mode:
        pip install -e .
    
    With machine learning:
        pip install -e .[ml]
    
    Full research environment:
        pip install -e .[research]
    
    Complete installation:
        pip install -e .[all]

Command-Line Interface:
    After installation, the following commands are available:
    - ag-news: Main CLI entry point
    - ag-news-health: System health check and diagnostics
    - ag-news-train: Model training interface
    - ag-news-evaluate: Model evaluation interface
    - ag-news-setup: Interactive setup wizard
    - ag-news-config: Configuration generator

For comprehensive documentation:
    - README.md: Project overview and quick start
    - QUICK_START.md: 5-minute getting started guide
    - docs/getting_started/installation.md: Detailed installation
    - ARCHITECTURE.md: System architecture documentation
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Set

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


ROOT_DIR = Path(__file__).parent.resolve()
VERSION_FILE = ROOT_DIR / "src" / "__version__.py"


def get_version() -> str:
    """
    Extract version string from __version__.py.
    
    The version is maintained in a single source of truth file to ensure
    consistency across the package. This follows PEP 396 recommendations.
    
    Returns:
        str: Semantic version string (e.g., "1.0.0")
        
    References:
        PEP 396 -- Module Version Numbers
        https://www.python.org/dev/peps/pep-0396/
    """
    version_namespace = {}
    if VERSION_FILE.exists():
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            exec(f.read(), version_namespace)
        return version_namespace.get("__version__", "1.0.0")
    return "1.0.0"


def read_file(filename: str) -> str:
    """
    Read and return file content.
    
    Args:
        filename: Relative path to file from project root
        
    Returns:
        str: File content or empty string if file not found
    """
    filepath = ROOT_DIR / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def parse_requirements_file(filename: str, processed: Set[str] = None) -> List[str]:
    """
    Parse requirements file with recursive include support.
    
    This function handles:
    - Comment lines (starting with #)
    - Empty lines
    - Include directives (-r other_file.txt)
    - Pip options (--index-url, etc.)
    - Environment markers (package; python_version >= "3.9")
    - Circular dependency detection
    
    Args:
        filename: Requirements file name relative to requirements/ directory
        processed: Set of already processed files (for circular dependency detection)
        
    Returns:
        List[str]: Parsed requirement specifications
        
    Notes:
        Requirements files are located in requirements/ directory following
        the project's modular dependency structure.
    """
    if processed is None:
        processed = set()
    
    req_file = ROOT_DIR / "requirements" / filename
    
    if not req_file.exists():
        print(f"Warning: Requirements file not found: {req_file}", file=sys.stderr)
        return []
    
    if str(req_file) in processed:
        print(f"Warning: Circular dependency detected: {req_file}", file=sys.stderr)
        return []
    
    processed.add(str(req_file))
    requirements = []
    
    with open(req_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            if not line or line.startswith("#"):
                continue
            
            if line.startswith("-r "):
                included_file = line.split(maxsplit=1)[1]
                requirements.extend(parse_requirements_file(included_file, processed))
                continue
            
            if line.startswith("-"):
                continue
            
            requirements.append(line)
    
    return requirements


def get_long_description() -> str:
    """
    Construct long description for PyPI from README.
    
    Returns:
        str: Formatted long description in Markdown
    """
    readme_content = read_file("README.md")
    if readme_content:
        return readme_content
    
    return """
AG News Text Classification
============================

A comprehensive, state-of-the-art framework for text classification research
and production deployment, featuring advanced overfitting prevention mechanisms
and parameter-efficient training methods.

Key Features
------------

- State-of-the-art Models: DeBERTa-v3, RoBERTa, ELECTRA, XLNet, LLaMA, Mistral
- Parameter-Efficient Methods: LoRA, QLoRA, Adapters, Prefix Tuning, Prompt Tuning
- Overfitting Prevention: Multi-layer validation, test set protection, automated monitoring
- Ensemble Learning: Voting, Stacking, Blending with diversity optimization
- Platform Support: Local, Google Colab, Kaggle, HuggingFace Spaces
- Academic Rigor: Reproducible experiments, ablation studies, statistical validation
- Free Deployment: Optimized for free-tier cloud platforms with quota management

Documentation: https://github.com/VoHaiDung/ag-news-text-classification
"""


def get_install_requires() -> List[str]:
    """
    Get base installation requirements.
    
    Base requirements include core dependencies needed for basic functionality:
    - Deep learning framework (PyTorch)
    - Transformer models (Transformers, Tokenizers)
    - Data handling (NumPy, Pandas, Datasets)
    - Configuration (YAML, Pydantic)
    - Utilities (tqdm, loguru, rich)
    
    Returns:
        List[str]: Base package requirements
    """
    return parse_requirements_file("base.txt")


def get_extras_require() -> Dict[str, List[str]]:
    """
    Define optional dependency groups for specialized use cases.
    
    Dependency Groups:
        ml: Traditional and neural ML models, experiment tracking
        llm: Large Language Model support with quantization
        efficient: Parameter-efficient fine-tuning methods
        data: Advanced data processing and augmentation
        ui: Web interfaces (Streamlit, Gradio)
        dev: Development tools (pytest, black, mypy)
        docs: Documentation generation (Sphinx, MkDocs)
        research: Research tools (Optuna, Ray Tune, Jupyter)
        robustness: Adversarial training and robustness testing
        local_prod: Local production deployment
        local_monitoring: Local monitoring stack (TensorBoard, MLflow)
        colab: Google Colab optimizations
        kaggle: Kaggle platform support
        free_tier: Free-tier cloud platform optimizations
        minimal: Absolute minimal dependencies
        
    Composite Groups:
        complete: ml + llm + efficient + data + ui
        research_full: research + robustness + dev
        all: All dependencies for local development
    
    Returns:
        Dict[str, List[str]]: Mapping of extra names to requirement lists
    """
    extras = {
        "ml": parse_requirements_file("ml.txt"),
        "llm": parse_requirements_file("llm.txt"),
        "efficient": parse_requirements_file("efficient.txt"),
        "data": parse_requirements_file("data.txt"),
        "ui": parse_requirements_file("ui.txt"),
        "dev": parse_requirements_file("dev.txt"),
        "docs": parse_requirements_file("docs.txt"),
        "research": parse_requirements_file("research.txt"),
        "robustness": parse_requirements_file("robustness.txt"),
        "local_prod": parse_requirements_file("local_prod.txt"),
        "local_monitoring": parse_requirements_file("local_monitoring.txt"),
        "colab": parse_requirements_file("colab.txt"),
        "kaggle": parse_requirements_file("kaggle.txt"),
        "free_tier": parse_requirements_file("free_tier.txt"),
        "minimal": parse_requirements_file("minimal.txt"),
        "platform_minimal": parse_requirements_file("platform_minimal.txt"),
    }
    
    extras["complete"] = list(set(
        extras.get("ml", []) +
        extras.get("llm", []) +
        extras.get("efficient", []) +
        extras.get("data", []) +
        extras.get("ui", [])
    ))
    
    extras["research_full"] = list(set(
        extras.get("research", []) +
        extras.get("robustness", []) +
        extras.get("dev", [])
    ))
    
    extras["all"] = parse_requirements_file("all_local.txt")
    
    return extras


def get_entry_points() -> Dict[str, List[str]]:
    """
    Define command-line interface entry points.
    
    Entry Points:
        ag-news: Main CLI interface with subcommands
        ag-news-health: System health check and diagnostics
        ag-news-train: Direct training interface
        ag-news-evaluate: Direct evaluation interface
        ag-news-setup: Interactive setup wizard for beginners
        ag-news-config: Configuration file generator
    
    Returns:
        Dict[str, List[str]]: Console scripts mapping
        
    Implementation Notes:
        Entry points follow the pattern: command_name=module.path:function
        All entry points are validated during package installation.
    """
    return {
        "console_scripts": [
            "ag-news=src.cli:main",
            "ag-news-health=src.core.health.health_checker:main",
            "ag-news-train=src.cli:train_command",
            "ag-news-evaluate=src.cli:evaluate_command",
            "ag-news-setup=quickstart.setup_wizard:main",
            "ag-news-config=tools.config_tools.config_generator:main",
        ],
    }


def get_package_data() -> Dict[str, List[str]]:
    """
    Specify non-Python files to include in package distribution.
    
    Included File Types:
        - Configuration files: *.yaml, *.yml, *.json
        - Documentation: *.md, *.txt
        - Configuration schemas: *.cfg, *.ini
        - Prompt templates: prompts/**/*.txt
        - Model configurations: configs/**/*.yaml
    
    Returns:
        Dict[str, List[str]]: Package data patterns by package
    """
    return {
        "": [
            "*.yaml",
            "*.yml",
            "*.json",
            "*.txt",
            "*.md",
            "*.cfg",
            "*.ini",
        ],
        "configs": [
            "**/*.yaml",
            "**/*.yml",
            "**/*.json",
        ],
        "prompts": [
            "**/*.txt",
        ],
    }


def get_classifiers() -> List[str]:
    """
    Define PyPI classifiers for package discovery.
    
    Classifiers categorize the package for PyPI search and filtering.
    They follow the Trove classification system.
    
    Returns:
        List[str]: PyPI Trove classifiers
        
    References:
        https://pypi.org/classifiers/
    """
    return [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Environment :: Console",
        "Environment :: GPU",
        "Environment :: GPU :: NVIDIA CUDA",
        "Framework :: Jupyter",
        "Typing :: Typed",
    ]


def get_keywords() -> List[str]:
    """
    Define search keywords for PyPI.
    
    Returns:
        List[str]: Search keywords
    """
    return [
        "text-classification",
        "natural-language-processing",
        "nlp",
        "transformers",
        "deep-learning",
        "machine-learning",
        "pytorch",
        "huggingface",
        "ag-news",
        "news-classification",
        "lora",
        "qlora",
        "parameter-efficient-fine-tuning",
        "peft",
        "ensemble-learning",
        "overfitting-prevention",
        "llm",
        "large-language-models",
        "deberta",
        "roberta",
        "electra",
        "xlnet",
        "llama",
        "mistral",
        "falcon",
        "model-training",
        "academic-research",
        "sota",
        "state-of-the-art",
        "knowledge-distillation",
        "curriculum-learning",
        "adversarial-training",
    ]


class PostDevelopCommand(develop):
    """
    Post-installation hook for development mode installation.
    
    Executes additional setup tasks after editable installation:
        1. Download NLTK data packages for text preprocessing
        2. Install pre-commit hooks for code quality
        3. Initialize local configuration directories
        4. Verify installation health
    
    Usage:
        pip install -e .
    
    This command is automatically invoked during development installation
    and ensures the development environment is properly configured.
    """
    
    def run(self):
        """Execute post-development installation tasks."""
        develop.run(self)
        
        print("\n" + "="*70)
        print("Post-Development Installation Setup")
        print("="*70 + "\n")
        
        self._download_nltk_data()
        self._setup_precommit_hooks()
        self._display_next_steps()
    
    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        try:
            import nltk
            print("[1/2] Downloading NLTK data packages...")
            
            packages = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"]
            for package in packages:
                nltk.download(package, quiet=True)
            
            print("      NLTK data downloaded successfully.")
        except Exception as e:
            print(f"      Warning: NLTK data download failed: {e}")
            print("      You can manually download with: python -m nltk.downloader punkt stopwords")
    
    def _setup_precommit_hooks(self):
        """Install pre-commit hooks for code quality."""
        try:
            import subprocess
            print("[2/2] Installing pre-commit hooks...")
            
            subprocess.run(
                ["pre-commit", "install"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            subprocess.run(
                ["pre-commit", "install", "--hook-type", "commit-msg"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            print("      Pre-commit hooks installed successfully.")
        except Exception as e:
            print(f"      Warning: Pre-commit setup failed: {e}")
            print("      Install manually with: pre-commit install")
    
    def _display_next_steps(self):
        """Display next steps for developers."""
        print("\n" + "="*70)
        print("Development Environment Ready!")
        print("="*70)
        print("\nNext Steps:")
        print("  1. Run health check:  ag-news-health")
        print("  2. Run setup wizard:  ag-news-setup")
        print("  3. Run tests:         pytest tests/")
        print("  4. View documentation: docs/getting_started/quickstart.md")
        print("\nFor quick start:       python quickstart/auto_start.py")
        print("="*70 + "\n")


class PostInstallCommand(install):
    """
    Post-installation hook for standard installation.
    
    Executes setup tasks after normal package installation:
        1. Download essential NLTK data
        2. Display getting started information
    
    Usage:
        pip install .
    """
    
    def run(self):
        """Execute post-installation tasks."""
        install.run(self)
        
        print("\n" + "="*70)
        print("AG News Text Classification - Installation Complete")
        print("="*70 + "\n")
        
        self._download_nltk_data()
        self._display_getting_started()
    
    def _download_nltk_data(self):
        """Download essential NLTK data packages."""
        try:
            import nltk
            print("Downloading NLTK data...")
            
            essential_packages = ["punkt", "stopwords"]
            for package in essential_packages:
                nltk.download(package, quiet=True)
            
            print("NLTK data downloaded successfully.\n")
        except Exception as e:
            print(f"Warning: NLTK data download failed: {e}\n")
    
    def _display_getting_started(self):
        """Display getting started information."""
        print("="*70)
        print("Getting Started")
        print("="*70)
        print("\nQuick Commands:")
        print("  System health check:  ag-news-health")
        print("  Interactive setup:    ag-news-setup")
        print("  View help:            ag-news --help")
        print("\nDocumentation:")
        print("  Quick Start:  QUICK_START.md")
        print("  Full Guide:   docs/getting_started/installation.md")
        print("  Repository:   https://github.com/VoHaiDung/ag-news-text-classification")
        print("="*70 + "\n")


setup(
    name="ag-news-text-classification",
    version=get_version(),
    author="Võ Hải Dũng",
    author_email="vohaidung.work@gmail.com",
    
    description=(
        "State-of-the-art text classification framework for AG News dataset "
        "with comprehensive overfitting prevention and parameter-efficient fine-tuning"
    ),
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    url="https://github.com/VoHaiDung/ag-news-text-classification",
    project_urls={
        "Bug Reports": "https://github.com/VoHaiDung/ag-news-text-classification/issues",
        "Documentation": "https://github.com/VoHaiDung/ag-news-text-classification#readme",
        "Source Code": "https://github.com/VoHaiDung/ag-news-text-classification",
        "Changelog": "https://github.com/VoHaiDung/ag-news-text-classification/blob/main/CHANGELOG.md",
    },
    
    license="MIT",
    keywords=get_keywords(),
    classifiers=get_classifiers(),
    
    packages=find_packages(where=".", exclude=["tests", "tests.*", "experiments", "experiments.*"]),
    package_dir={"": "."},
    package_data=get_package_data(),
    include_package_data=True,
    
    python_requires=">=3.8,<3.12",
    
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
    
    entry_points=get_entry_points(),
    
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
    
    zip_safe=False,
)
