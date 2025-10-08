# ============================================================================
# Package Manifest for AG News Text Classification
# ============================================================================
# Project: AG News Text Classification (ag-news-text-classification)
# Description: Specification of files to include in source distribution
# Author: Võ Hải Dũng
# Email: vohaidung.work@gmail.com
# License: MIT
# ============================================================================
# This file specifies which files should be included in the source distribution
# when the package is built with setuptools.
#
# Syntax:
#   include <pattern>           - Include files matching pattern
#   recursive-include <dir> <pattern> - Include recursively
#   exclude <pattern>           - Exclude files matching pattern
#   global-exclude <pattern>    - Exclude globally
#   graft <dir>                 - Include everything in directory
#   prune <dir>                 - Exclude everything in directory
#
# For more information:
# - https://packaging.python.org/en/latest/guides/using-manifest-in/
# - https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html
# ============================================================================

# ============================================================================
# Project Documentation
# ============================================================================

# Root documentation files
include README.md
include LICENSE
include CHANGELOG.md
include CITATION.cff
include AUTHORS.md
include CONTRIBUTING.md

# Specialized guides
include ARCHITECTURE.md
include PERFORMANCE.md
include SECURITY.md
include TROUBLESHOOTING.md
include SOTA_MODELS_GUIDE.md
include OVERFITTING_PREVENTION.md
include ROADMAP.md
include FREE_DEPLOYMENT_GUIDE.md
include IDE_SETUP_GUIDE.md
include LOCAL_MONITORING_GUIDE.md
include QUICK_START.md
include HEALTH_CHECK.md

# ============================================================================
# Build and Package Configuration
# ============================================================================

# Setup files
include setup.py
include setup.cfg
include MANIFEST.in
include pyproject.toml
include poetry.lock

# Makefile for automation
include Makefile

# Installation scripts
include install.sh

# ============================================================================
# Environment Configuration
# ============================================================================

# Environment templates
include .env.example
include .env.test
include .env.local

# Git configuration
include .gitignore
include .gitattributes
include .dockerignore

# Editor configuration
include .editorconfig

# ============================================================================
# Code Quality Configuration
# ============================================================================

# Pre-commit hooks
include .pre-commit-config.yaml

# Linting configuration
include .flake8
include .pylintrc

# Commit message linting
include commitlint.config.js

# ============================================================================
# Requirements Files
# ============================================================================

# Include all requirements files
include requirements/*.txt
include requirements/lock/*.lock
include requirements/lock/README.md

# ============================================================================
# Configuration Files
# ============================================================================

# All configuration files
recursive-include configs *.yaml
recursive-include configs *.yml
recursive-include configs *.json
recursive-include configs *.py
recursive-include configs *.toml
recursive-include configs *.j2

# Include README files in configs
recursive-include configs README.md
recursive-include configs SELECTION_GUIDE.md
recursive-include configs ENSEMBLE_SELECTION_GUIDE.yaml

# ============================================================================
# Source Code
# ============================================================================

# Python source files
recursive-include src *.py
recursive-include src README.md

# Include package data
recursive-include src *.yaml
recursive-include src *.json
recursive-include src *.txt

# Version file
include src/__version__.py

# ============================================================================
# Scripts
# ============================================================================

# All scripts
recursive-include scripts *.py
recursive-include scripts *.sh
recursive-include scripts *.bash
recursive-include scripts README.md

# Make scripts executable
graft scripts/setup
graft scripts/data_preparation
graft scripts/training
graft scripts/evaluation
graft scripts/optimization
graft scripts/deployment
graft scripts/overfitting_prevention
graft scripts/ide
graft scripts/local
graft scripts/ci

# ============================================================================
# Prompts
# ============================================================================

# Prompt templates for LLM
recursive-include prompts *.txt
recursive-include prompts *.md
recursive-include prompts README.md

# ============================================================================
# Tools
# ============================================================================

# Development tools
recursive-include tools *.py
recursive-include tools README.md

graft tools/profiling
graft tools/debugging
graft tools/visualization
graft tools/config_tools
graft tools/ide_tools
graft tools/compatibility
graft tools/automation
graft tools/cli_helpers

# ============================================================================
# Images and Static Assets
# ============================================================================

# Documentation images
recursive-include images *.png
recursive-include images *.jpg
recursive-include images *.jpeg
recursive-include images *.svg
recursive-include images *.pdf

# ============================================================================
# IDE Configuration
# ============================================================================

# IDE setup files
graft .ide

# VSCode
recursive-include .ide/vscode *.json

# PyCharm
recursive-include .ide/pycharm *.xml
include .ide/pycharm/README_PYCHARM.md
include .ide/pycharm/settings.zip

# Jupyter
recursive-include .ide/jupyter *.py
recursive-include .ide/jupyter *.json
recursive-include .ide/jupyter *.css
recursive-include .ide/jupyter *.js

# Vim
include .ide/vim/.vimrc
include .ide/vim/coc-settings.json
include .ide/vim/README_VIM.md
recursive-include .ide/vim/ultisnips *.snippets

# Neovim
include .ide/neovim/init.lua
include .ide/neovim/coc-settings.json
include .ide/neovim/README_NEOVIM.md
recursive-include .ide/neovim/lua *.lua

# Sublime Text
recursive-include .ide/sublime *.sublime-project
recursive-include .ide/sublime *.sublime-workspace
recursive-include .ide/sublime *.sublime-settings
recursive-include .ide/sublime *.sublime-snippet
recursive-include .ide/sublime *.sublime-build
include .ide/sublime/README_SUBLIME.md

# Cloud IDEs
recursive-include .ide/cloud_ides *.yml
recursive-include .ide/cloud_ides *.yaml
recursive-include .ide/cloud_ides *.json
recursive-include .ide/cloud_ides *.py
recursive-include .ide/cloud_ides Dockerfile

# Source of truth
include .ide/SOURCE_OF_TRUTH.yaml

# ============================================================================
# Dev Container
# ============================================================================

recursive-include .devcontainer *.json
recursive-include .devcontainer Dockerfile

# ============================================================================
# Husky Git Hooks
# ============================================================================

recursive-include .husky *

# ============================================================================
# Templates
# ============================================================================

# Project templates
recursive-include templates *.py
recursive-include templates *.yaml
recursive-include templates *.json
recursive-include templates *.xml
recursive-include templates *.ipynb
recursive-include templates README*.md

# ============================================================================
# Quickstart
# ============================================================================

# Quick start files
recursive-include quickstart *.py
recursive-include quickstart *.ipynb
recursive-include quickstart *.sh
recursive-include quickstart *.md
recursive-include quickstart *.yaml
recursive-include quickstart Dockerfile
recursive-include quickstart docker-compose*.yml

# ============================================================================
# Monitoring
# ============================================================================

# Monitoring configuration
recursive-include monitoring *.yaml
recursive-include monitoring *.yml
recursive-include monitoring *.json
recursive-include monitoring *.py
recursive-include monitoring *.sh
recursive-include monitoring README.md
recursive-include monitoring docker-compose*.yml

# ============================================================================
# Security
# ============================================================================

# Security templates and scripts
recursive-include security *.py
recursive-include security *.yaml
recursive-include security *.md

# ============================================================================
# Migrations
# ============================================================================

# Migration scripts
recursive-include migrations *.py
recursive-include migrations README.md

# ============================================================================
# Cache Configuration
# ============================================================================

# Cache schema
recursive-include cache *.sql
recursive-include cache *.py

# ============================================================================
# Backup Configuration
# ============================================================================

# Backup strategies and scripts
recursive-include backup *.yaml
recursive-include backup *.sh
recursive-include backup *.md

# ============================================================================
# Deployment
# ============================================================================

# Deployment configurations
recursive-include deployment *.yaml
recursive-include deployment *.yml
recursive-include deployment *.conf
recursive-include deployment *.service
recursive-include deployment *.sh
recursive-include deployment Dockerfile*
recursive-include deployment docker-compose*.yml
recursive-include deployment .dockerignore
recursive-include deployment README.md
recursive-include deployment requirements.txt

# ============================================================================
# App
# ============================================================================

# Application files
recursive-include app *.py
recursive-include app *.yaml
recursive-include app *.css
recursive-include app *.js
recursive-include app *.png
recursive-include app *.jpg
recursive-include app *.svg

# ============================================================================
# Tests
# ============================================================================

# Test files
recursive-include tests *.py
recursive-include tests *.yaml
recursive-include tests *.json
recursive-include tests conftest.py
recursive-include tests README.md

# Test fixtures
recursive-include tests/fixtures *.json
recursive-include tests/fixtures *.yaml
recursive-include tests/fixtures *.txt

# ============================================================================
# Benchmarks
# ============================================================================

# Benchmark results
recursive-include benchmarks *.json
recursive-include benchmarks *.yaml
recursive-include benchmarks README.md

# ============================================================================
# Documentation
# ============================================================================

# Documentation source
recursive-include docs *.md
recursive-include docs *.rst
recursive-include docs *.py
recursive-include docs *.txt
recursive-include docs conf.py
recursive-include docs Makefile
recursive-include docs make.bat
recursive-include docs requirements.txt

# MkDocs
include mkdocs.yml
include docs.yml

# Sphinx
recursive-include docs/_static *
recursive-include docs/_templates *

# Cheatsheets
recursive-include docs/cheatsheets *.pdf

# ============================================================================
# Notebooks
# ============================================================================

# Jupyter notebooks
recursive-include notebooks *.ipynb
recursive-include notebooks *.py
recursive-include notebooks *.md
recursive-include notebooks README.md

# ============================================================================
# Plugins
# ============================================================================

# Plugin system
recursive-include plugins *.py
recursive-include plugins README.md

# ============================================================================
# Data (Sample/Test Data Only)
# ============================================================================

# Include sample and test data
recursive-include data/test_samples *.json
recursive-include data/test_samples *.csv
recursive-include data/test_samples *.txt

# Include metadata
recursive-include data/metadata *.json
recursive-include data/metadata README.md

# Keep directory structure
include data/raw/.gitkeep
include data/processed/.gitkeep
include data/augmented/.gitkeep
include data/external/.gitkeep
include data/cache/.gitkeep

# ============================================================================
# Outputs (Structure Only)
# ============================================================================

# Keep directory structure
include outputs/.gitkeep
include outputs/models/.gitkeep
include outputs/results/.gitkeep
include outputs/logs/.gitkeep
include outputs/analysis/.gitkeep

# ============================================================================
# GitHub Workflows
# ============================================================================

# CI/CD workflows
recursive-include .github *.yml
recursive-include .github *.yaml
recursive-include .github *.md

# ============================================================================
# Exclusions
# ============================================================================

# Exclude Python cache
global-exclude __pycache__
global-exclude *.py[cod]
global-exclude *$py.class
global-exclude *.so

# Exclude tests artifacts
global-exclude .pytest_cache
global-exclude .coverage
global-exclude htmlcov
global-exclude .hypothesis

# Exclude Jupyter checkpoints
global-exclude .ipynb_checkpoints

# Exclude IDE specific
global-exclude .idea
global-exclude .vscode
global-exclude *.swp
global-exclude *.swo
global-exclude *~

# Exclude OS files
global-exclude .DS_Store
global-exclude Thumbs.db

# Exclude model weights and large files
global-exclude *.pt
global-exclude *.pth
global-exclude *.ckpt
global-exclude *.safetensors
global-exclude *.bin
global-exclude *.h5
global-exclude *.onnx
global-exclude *.pkl
global-exclude *.pickle

# Exclude logs
global-exclude *.log

# Exclude environment files
global-exclude .env
exclude .env.local
exclude .env.prod

# Exclude actual data
prune data/raw/ag_news
prune data/processed/train
prune data/processed/validation
prune data/processed/test
prune data/augmented/back_translated
prune data/augmented/paraphrased
prune data/augmented/synthetic
prune data/external/news_corpus

# Exclude outputs
prune outputs/models/checkpoints
prune outputs/models/pretrained
prune outputs/models/fine_tuned
prune outputs/results/experiments
prune outputs/logs/training

# Exclude cache
prune cache/local
prune .cache

# Exclude build artifacts
prune build
prune dist
prune *.egg-info

# Exclude node modules
prune node_modules

# Exclude experiment results
prune experiments/results

# Exclude wandb
prune wandb

# Exclude mlruns
prune mlruns

# ============================================================================
# Notes
# ============================================================================
# This MANIFEST.in is comprehensive for the AG News Text Classification project.
#
# Included:
# - All documentation and guides
# - All configuration files
# - All source code
# - All scripts and tools
# - IDE configurations for 10 IDEs
# - Sample and test data
# - Templates and examples
# - Deployment configurations
# - Security templates
#
# Excluded:
# - Large model weights (download separately)
# - Actual training data (download via scripts)
# - Build artifacts
# - Cache files
# - IDE-specific files
# - OS-specific files
# - Environment secrets
#
# Size estimate:
# - Source distribution: 50-100 MB
# - Without model weights and data: manageable size
# - All configs and code included
#
# Users can:
# - Install package: pip install ag-news-text-classification
# - Download data: python scripts/setup/download_all_data.py
# - Download models: python scripts/setup/download_pretrained_models.py
#
# Build source distribution:
#   python setup.py sdist
#
# Build wheel:
#   python setup.py bdist_wheel
#
# Check MANIFEST:
#   python setup.py sdist --dry-run --verbose
# ============================================================================
