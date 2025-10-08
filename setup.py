#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup Script for AG News Text Classification
============================================

Project: AG News Text Classification (ag-news-text-classification)
Description: State-of-the-art text classification framework with comprehensive
             overfitting prevention, parameter-efficient fine-tuning, and
             ensemble learning capabilities.

Author: Võ Hải Dũng
License: MIT
Python: >=3.8,<3.12

This setup script provides:
- Standard pip installation
- Development installation with editable mode
- Multiple installation profiles (minimal, ml, llm, research, etc.)
- Command-line interface registration
- Package data inclusion
- Automated dependency resolution

Installation Examples:
    Minimal installation:
        pip install .
    
    Development installation:
        pip install -e .
    
    Full ML stack:
        pip install -e .[ml]
    
    Research environment:
        pip install -e .[research,dev,docs]
    
    Complete local environment:
        pip install -e .[all]

For detailed installation instructions, see:
- README.md
- docs/getting_started/installation.md
- QUICK_START.md
"""

import os
import sys
from pathlib import Path
from typing import List, Dict

from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install


# Project root directory
ROOT_DIR = Path(__file__).parent.resolve()

# Version management
VERSION_FILE = ROOT_DIR / "src" / "__version__.py"


def get_version() -> str:
    """
    Extract version from __version__.py file.
    
    Returns:
        str: Project version string
    """
    version_dict = {}
    if VERSION_FILE.exists():
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            exec(f.read(), version_dict)
        return version_dict.get("__version__", "1.0.0")
    return "1.0.0"


def read_file(filename: str) -> str:
    """
    Read content from a file.
    
    Args:
        filename: Name of file to read
        
    Returns:
        str: File content
    """
    filepath = ROOT_DIR / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def read_requirements(filename: str) -> List[str]:
    """
    Read requirements from requirements file.
    
    Handles:
    - Comments (lines starting with #)
    - Empty lines
    - Include directives (-r another_file.txt)
    - Environment markers (package; python_version >= "3.9")
    
    Args:
        filename: Requirements file name (e.g., "base.txt")
        
    Returns:
        List[str]: List of requirement strings
    """
    req_file = ROOT_DIR / "requirements" / filename
    requirements = []
    
    if not req_file.exists():
        print(f"Warning: Requirements file not found: {req_file}")
        return requirements
    
    with open(req_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            
            # Handle include directives
            if line.startswith("-r "):
                included_file = line.split()[1]
                requirements.extend(read_requirements(included_file))
                continue
            
            # Skip other pip options
            if line.startswith("-"):
                continue
            
            requirements.append(line)
    
    return requirements


def get_long_description() -> str:
    """
    Get long description from README.md.
    
    Returns:
        str: Long description for PyPI
    """
    readme = read_file("README.md")
    if readme:
        return readme
    
    # Fallback description
    return """
AG News Text Classification Framework
======================================

A state-of-the-art text classification framework featuring:

- SOTA models (DeBERTa, RoBERTa, ELECTRA, LLaMA, Mistral)
- Parameter-efficient fine-tuning (LoRA, QLoRA, Adapters)
- Comprehensive overfitting prevention system
- Ensemble learning methods
- Local deployment and monitoring
- Multi-IDE support
- Academic-grade reproducibility

See GitHub repository for full documentation.
"""


def get_install_requires() -> List[str]:
    """
    Get base installation requirements.
    
    Returns:
        List[str]: Base dependencies
    """
    return read_requirements("base.txt")


def get_extras_require() -> Dict[str, List[str]]:
    """
    Get optional dependency groups.
    
    Returns:
        Dict[str, List[str]]: Mapping of extra names to requirements
    """
    extras = {
        # Core extras
        "ml": read_requirements("ml.txt"),
        "llm": read_requirements("llm.txt"),
        "efficient": read_requirements("efficient.txt"),
        "data": read_requirements("data.txt"),
        "ui": read_requirements("ui.txt"),
        
        # Development and testing
        "dev": read_requirements("dev.txt"),
        "test": [
            "pytest>=7.4.0,<8.3.0",
            "pytest-cov>=4.1.0,<5.1.0",
            "pytest-xdist>=3.5.0,<3.7.0",
            "pytest-mock>=3.12.0,<3.15.0",
            "hypothesis>=6.92.0,<6.109.0",
        ],
        
        # Documentation
        "docs": read_requirements("docs.txt"),
        
        # Research and experimentation
        "research": read_requirements("research.txt"),
        "robustness": read_requirements("robustness.txt"),
        
        # Deployment
        "local_prod": read_requirements("local_prod.txt"),
        "local_monitoring": read_requirements("local_monitoring.txt"),
        
        # Cloud platforms
        "colab": read_requirements("colab.txt"),
        "kaggle": read_requirements("kaggle.txt"),
        "free_tier": read_requirements("free_tier.txt"),
        
        # Minimal installation
        "minimal": read_requirements("minimal.txt"),
    }
    
    # Convenience combinations
    extras["complete"] = list(set(
        extras["ml"] + 
        extras["llm"] + 
        extras["efficient"] +
        extras["data"] +
        extras["ui"]
    ))
    
    extras["research_full"] = list(set(
        extras["research"] + 
        extras["robustness"] + 
        extras["dev"]
    ))
    
    extras["all"] = read_requirements("all_local.txt")
    
    return extras


def get_entry_points() -> Dict[str, List[str]]:
    """
    Define command-line interface entry points.
    
    Returns:
        Dict[str, List[str]]: Console scripts mapping
    """
    return {
        "console_scripts": [
            # Main CLI
            "ag-news=src.cli:main",
            
            # Health checks
            "ag-news-health=src.core.health.health_checker:main",
            
            # Training commands
            "ag-news-train=src.cli:train_command",
            "ag-news-evaluate=src.cli:evaluate_command",
            
            # Model management
            "ag-news-models=src.cli:models_command",
            
            # Configuration tools
            "ag-news-config=tools.config_tools.config_generator:main",
            
            # Setup wizard
            "ag-news-setup=quickstart.setup_wizard:main",
        ],
    }


def get_package_data() -> Dict[str, List[str]]:
    """
    Define package data to include.
    
    Returns:
        Dict[str, List[str]]: Package data patterns
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
    Define package classifiers for PyPI.
    
    Returns:
        List[str]: Trove classifiers
    """
    return [
        # Development status
        "Development Status :: 4 - Beta",
        
        # Intended audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming language
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        
        # Topics
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # Natural Language
        "Natural Language :: English",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        
        # Environment
        "Environment :: Console",
        "Environment :: GPU",
        "Environment :: GPU :: NVIDIA CUDA",
        
        # Framework
        "Framework :: Jupyter",
        
        # Typing
        "Typing :: Typed",
    ]


def get_keywords() -> List[str]:
    """
    Define package keywords for search.
    
    Returns:
        List[str]: Keywords
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
        "llama",
        "mistral",
        "model-training",
        "academic-research",
        "sota",
        "state-of-the-art",
    ]


class PostDevelopCommand(develop):
    """
    Post-installation for development mode.
    
    Executes additional setup steps after installing in development mode:
    - Download NLTK data
    - Setup pre-commit hooks
    - Initialize configuration
    """
    
    def run(self):
        """Execute post-development installation tasks."""
        develop.run(self)
        
        print("\nPost-installation setup for development mode...")
        
        # Download NLTK data
        try:
            import nltk
            print("Downloading NLTK data...")
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            print("NLTK data downloaded successfully.")
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
        
        # Setup pre-commit hooks
        try:
            import subprocess
            print("Setting up pre-commit hooks...")
            subprocess.run(["pre-commit", "install"], check=True)
            subprocess.run(["pre-commit", "install", "--hook-type", "commit-msg"], check=True)
            print("Pre-commit hooks installed successfully.")
        except Exception as e:
            print(f"Warning: Could not setup pre-commit hooks: {e}")
        
        print("\nDevelopment setup complete!")
        print("Run 'ag-news-health' to verify installation.")


class PostInstallCommand(install):
    """
    Post-installation for normal installation.
    
    Executes setup steps after normal installation:
    - Download NLTK data
    - Display next steps
    """
    
    def run(self):
        """Execute post-installation tasks."""
        install.run(self)
        
        print("\nPost-installation setup...")
        
        # Download NLTK data
        try:
            import nltk
            print("Downloading NLTK data...")
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            print("NLTK data downloaded successfully.")
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
        
        print("\nInstallation complete!")
        print("\nNext steps:")
        print("1. Run health check: ag-news-health")
        print("2. See quick start: ag-news-setup")
        print("3. Read documentation: docs/getting_started/quickstart.md")


# Main setup configuration
setup(
    # Basic metadata
    name="ag-news-text-classification",
    version=get_version(),
    author="Võ Hải Dũng",
    author_email="vohaidung.work@gmail.com",
    description="State-of-the-art text classification framework for AG News dataset",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/VoHaiDung/ag-news-text-classification",
    project_urls={
        "Bug Reports": "https://github.com/VoHaiDung/ag-news-text-classification/issues",
        "Documentation": "https://ag-news-text-classification.readthedocs.io/",
        "Source Code": "https://github.com/VoHaiDung/ag-news-text-classification",
        "Changelog": "https://github.com/VoHaiDung/ag-news-text-classification/blob/main/CHANGELOG.md",
    },
    license="MIT",
    keywords=get_keywords(),
    classifiers=get_classifiers(),
    
    # Package configuration
    packages=find_packages(where="src", exclude=["tests", "tests.*"]),
    package_dir={"": "src"},
    package_data=get_package_data(),
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.8,<3.12",
    
    # Dependencies
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
    
    # Entry points
    entry_points=get_entry_points(),
    
    # Additional files to include
    data_files=[
        ("", ["README.md", "LICENSE", "CHANGELOG.md"]),
    ],
    
    # Custom commands
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
    
    # Zip safety
    zip_safe=False,
)
