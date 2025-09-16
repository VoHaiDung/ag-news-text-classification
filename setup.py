#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for AG News Text Classification Framework
=======================================================

Following Python packaging best practices from:
- Python Packaging Authority (PyPA): https://www.pypa.io/
- setuptools documentation: https://setuptools.pypa.io/

Author: Võ Hải Dũng
License: MIT
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Project root directory
ROOT_DIR = Path(__file__).parent.resolve()

# Python version check
if sys.version_info < (3, 8):
    sys.exit("Python 3.8+ is required")

def read_file(filename):
    """Read file content."""
    filepath = ROOT_DIR / filename
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def read_requirements(filename):
    """Read requirements from file."""
    filepath = ROOT_DIR / "requirements" / filename
    if not filepath.exists():
        return []
    
    requirements = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-r"):
                requirements.append(line)
    return requirements

# Package metadata
NAME = "ag-news-classification"
VERSION = "1.0.0"
DESCRIPTION = "State-of-the-art text classification framework for AG News dataset"
LONG_DESCRIPTION = read_file("README.md")
AUTHOR = "Võ Hải Dũng"
AUTHOR_EMAIL = "contact@example.com"
URL = "https://github.com/yourusername/ag-news-classification"
LICENSE = "MIT"

# Package requirements
INSTALL_REQUIRES = read_requirements("base.txt")
EXTRAS_REQUIRE = {
    "dev": read_requirements("dev.txt"),
    "test": read_requirements("test.txt"),
    "docs": [
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
    ],
    "ml": [
        "optuna>=3.3.0",
        "ray[tune]>=2.6.0",
        "wandb>=0.15.0",
    ],
    "prod": [
        "fastapi>=0.103.0",
        "uvicorn>=0.23.0",
        "gunicorn>=21.2.0",
        "pydantic>=2.4.0",
    ],
}

# All extras
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

# Package configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    
    # Package discovery
    packages=find_packages(
        include=["src", "src.*"],
        exclude=["tests", "tests.*", "docs", "docs.*"]
    ),
    package_dir={"": "."},
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "ag-news-train=src.cli.train:main",
            "ag-news-evaluate=src.cli.evaluate:main",
            "ag-news-predict=src.cli.predict:main",
            "ag-news-serve=src.cli.serve:main",
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    
    # Keywords for search
    keywords="nlp text-classification ag-news transformers deep-learning machine-learning",
    
    # Project URLs
    project_urls={
        "Documentation": f"{URL}/docs",
        "Source": URL,
        "Tracker": f"{URL}/issues",
        "Changelog": f"{URL}/blob/main/CHANGELOG.md",
    },
    
    # Zip safe
    zip_safe=False,
)
