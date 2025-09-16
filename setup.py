#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for AG News Classification
========================================

Simplified setup for easy installation
Author: Võ Hải Dũng
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    req_file = Path("requirements") / filename
    if req_file.exists():
        with open(req_file) as f:
            return [line.strip() for line in f 
                   if line.strip() and not line.startswith("#") 
                   and not line.startswith("-r")]
    return []

# Basic setup
setup(
    name="ag-news-classification",
    version="1.0.0",
    author="Võ Hải Dũng",
    description="AG News Text Classification Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ag-news-classification",
    
    # Packages
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=read_requirements("base.txt"),
    
    # Optional dependencies
    extras_require={
        "dev": read_requirements("dev.txt"),
        "ml": read_requirements("ml.txt"),
        "research": read_requirements("research.txt"),
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
