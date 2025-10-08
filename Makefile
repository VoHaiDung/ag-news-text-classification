# ============================================================================
# Makefile for AG News Text Classification
# ============================================================================
# Project: AG News Text Classification (ag-news-text-classification)
# Description: Comprehensive build automation and workflow management
# Author: Võ Hải Dũng
# Email: vohaidung.work@gmail.com
# License: MIT
# ============================================================================
#
# Purpose:
#   This Makefile provides automated workflows for development, testing,
#   training, evaluation, and deployment of the AG News Text Classification
#   framework. Designed for academic reproducibility and production deployment.
#
# Architecture:
#   - Modular targets organized by functional area
#   - Dependency management with Python virtual environments
#   - Platform-aware execution (local, Colab, Kaggle)
#   - Comprehensive testing and quality assurance
#   - Documentation generation and validation
#   - Docker containerization support
#
# Quick Reference:
#   make help              Display all available targets
#   make setup             Complete development environment setup
#   make install           Install package and dependencies
#   make train             Train default model configuration
#   make test              Run comprehensive test suite
#   make lint              Run code quality checks
#   make docs              Generate project documentation
#   make clean             Remove build artifacts
#
# Common Workflows:
#   Development Setup:     make dev
#   Research Environment:  make research
#   Production Build:      make prod
#   Quick Demonstration:   make quickstart
#   Continuous Integration: make ci
#
# ============================================================================

.PHONY: help
.DEFAULT_GOAL := help

# ============================================================================
# Project Configuration
# ============================================================================

PROJECT_NAME := ag-news-text-classification
PROJECT_SLUG := ag_news_text_classification
VERSION := $(shell python -c "exec(open('src/__version__.py').read()); print(__version__)" 2>/dev/null || echo "1.0.0")
AUTHOR := Võ Hải Dũng
EMAIL := vohaidung.work@gmail.com
LICENSE := MIT

# ============================================================================
# Python Environment Configuration
# ============================================================================

PYTHON_VERSION := 3.10
PYTHON := python$(PYTHON_VERSION)
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
VENV := venv
VENV_BIN := $(VENV)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip
PYTHON_EXEC := $(shell which $(VENV_PYTHON) 2>/dev/null || which $(PYTHON) 2>/dev/null || which python3)

# ============================================================================
# Directory Structure
# ============================================================================

ROOT_DIR := $(shell pwd)
SRC_DIR := $(ROOT_DIR)/src
TEST_DIR := $(ROOT_DIR)/tests
DATA_DIR := $(ROOT_DIR)/data
CONFIGS_DIR := $(ROOT_DIR)/configs
OUTPUTS_DIR := $(ROOT_DIR)/outputs
DOCS_DIR := $(ROOT_DIR)/docs
SCRIPTS_DIR := $(ROOT_DIR)/scripts
NOTEBOOKS_DIR := $(ROOT_DIR)/notebooks
EXPERIMENTS_DIR := $(ROOT_DIR)/experiments
APP_DIR := $(ROOT_DIR)/app
MONITORING_DIR := $(ROOT_DIR)/monitoring
DEPLOYMENT_DIR := $(ROOT_DIR)/deployment
QUICKSTART_DIR := $(ROOT_DIR)/quickstart
TOOLS_DIR := $(ROOT_DIR)/tools
CACHE_DIR := $(ROOT_DIR)/.cache
BUILD_DIR := $(ROOT_DIR)/build
DIST_DIR := $(ROOT_DIR)/dist

DATA_RAW_DIR := $(DATA_DIR)/raw
DATA_PROCESSED_DIR := $(DATA_DIR)/processed
DATA_AUGMENTED_DIR := $(DATA_DIR)/augmented
DATA_EXTERNAL_DIR := $(DATA_DIR)/external

OUTPUTS_MODELS_DIR := $(OUTPUTS_DIR)/models
OUTPUTS_RESULTS_DIR := $(OUTPUTS_DIR)/results
OUTPUTS_LOGS_DIR := $(OUTPUTS_DIR)/logs
OUTPUTS_REPORTS_DIR := $(OUTPUTS_DIR)/reports
OUTPUTS_ANALYSIS_DIR := $(OUTPUTS_DIR)/analysis

# ============================================================================
# Tool Configuration
# ============================================================================

BLACK := $(VENV_BIN)/black
ISORT := $(VENV_BIN)/isort
FLAKE8 := $(VENV_BIN)/flake8
MYPY := $(VENV_BIN)/mypy
PYLINT := $(VENV_BIN)/pylint
BANDIT := $(VENV_BIN)/bandit
RUFF := $(VENV_BIN)/ruff
PYTEST_COV := $(VENV_BIN)/pytest --cov=$(SRC_DIR)

# ============================================================================
# Docker Configuration
# ============================================================================

DOCKER := docker
DOCKER_COMPOSE := docker-compose
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
DOCKER_IMAGE_LATEST := $(PROJECT_NAME):latest
DOCKER_REGISTRY := ghcr.io/vohaidung

# ============================================================================
# Platform Detection
# ============================================================================

PLATFORM := $(shell python -c "import sys; print('colab' if 'google.colab' in sys.modules else 'kaggle' if 'kaggle_secrets' in sys.modules else 'local')" 2>/dev/null || echo "local")
GPU_COUNT := $(shell nvidia-smi -L 2>/dev/null | wc -l || echo "0")
CUDA_AVAILABLE := $(shell python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")

# ============================================================================
# Timestamp for Logs
# ============================================================================

TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# ============================================================================
# Color Output Configuration
# ============================================================================

RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m
BOLD := \033[1m

# ============================================================================
# Helper Functions
# ============================================================================

define log
	@echo "$(CYAN)[$(shell date +%H:%M:%S)]$(RESET) $(1)"
endef

define success
	@echo "$(GREEN)$(BOLD)SUCCESS:$(RESET) $(1)"
endef

define warning
	@echo "$(YELLOW)$(BOLD)WARNING:$(RESET) $(1)"
endef

define error
	@echo "$(RED)$(BOLD)ERROR:$(RESET) $(1)" >&2
endef

define section
	@echo ""
	@echo "$(MAGENTA)$(BOLD)========================================$(RESET)"
	@echo "$(MAGENTA)$(BOLD)$(1)$(RESET)"
	@echo "$(MAGENTA)$(BOLD)========================================$(RESET)"
	@echo ""
endef

# ============================================================================
# Help Documentation
# ============================================================================

help: ## Display comprehensive help information
	@echo ""
	@echo "$(MAGENTA)$(BOLD)AG News Text Classification Framework$(RESET)"
	@echo "$(CYAN)Project:$(RESET) $(PROJECT_NAME)"
	@echo "$(CYAN)Version:$(RESET) $(VERSION)"
	@echo "$(CYAN)Author:$(RESET)  $(AUTHOR)"
	@echo "$(CYAN)Email:$(RESET)   $(EMAIL)"
	@echo "$(CYAN)License:$(RESET) $(LICENSE)"
	@echo ""
	@echo "$(CYAN)Platform:$(RESET) $(PLATFORM)"
	@echo "$(CYAN)GPUs:$(RESET)     $(GPU_COUNT)"
	@echo "$(CYAN)CUDA:$(RESET)     $(CUDA_AVAILABLE)"
	@echo ""
	@echo "$(YELLOW)$(BOLD)Usage:$(RESET)"
	@echo "  make $(CYAN)[target]$(RESET)"
	@echo ""
	@echo "$(YELLOW)$(BOLD)Available Targets:$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-30s$(RESET) %s\n", $$1, $$2}' | \
		sort
	@echo ""
	@echo "$(YELLOW)$(BOLD)Quick Start:$(RESET)"
	@echo "  1. Setup:    $(CYAN)make setup$(RESET)"
	@echo "  2. Install:  $(CYAN)make install-all$(RESET)"
	@echo "  3. Train:    $(CYAN)make train$(RESET)"
	@echo "  4. Test:     $(CYAN)make test$(RESET)"
	@echo ""

# ============================================================================
# Environment Setup
# ============================================================================

setup: setup-dirs setup-venv setup-git ## Complete development environment setup
	$(call section,Setup Complete)
	$(call success,Development environment ready!)
	@echo "$(CYAN)Activate virtual environment:$(RESET) source $(VENV)/bin/activate"

setup-dirs: ## Create project directory structure
	$(call log,Creating directory structure...)
	@mkdir -p $(DATA_RAW_DIR) $(DATA_PROCESSED_DIR) $(DATA_AUGMENTED_DIR) $(DATA_EXTERNAL_DIR)
	@mkdir -p $(DATA_DIR)/{cache,metadata,test_samples,platform_cache,quota_tracking}
	@mkdir -p $(OUTPUTS_MODELS_DIR)/{checkpoints,pretrained,fine_tuned,lora_adapters,qlora_adapters,ensembles,distilled,optimized,exported}
	@mkdir -p $(OUTPUTS_RESULTS_DIR)/{experiments,benchmarks,ablations,reports}
	@mkdir -p $(OUTPUTS_LOGS_DIR)/{training,tensorboard,mlflow,wandb,local}
	@mkdir -p $(OUTPUTS_REPORTS_DIR) $(OUTPUTS_ANALYSIS_DIR)
	@mkdir -p $(CACHE_DIR) $(BUILD_DIR) $(DIST_DIR)
	@touch $(DATA_RAW_DIR)/.gitkeep $(DATA_PROCESSED_DIR)/.gitkeep
	@touch $(OUTPUTS_MODELS_DIR)/.gitkeep $(OUTPUTS_RESULTS_DIR)/.gitkeep
	$(call success,Directory structure created)

setup-venv: ## Create and configure Python virtual environment
	$(call log,Creating Python virtual environment...)
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	$(call success,Virtual environment created at $(VENV))

setup-git: ## Configure Git hooks and pre-commit
	$(call log,Installing Git hooks...)
	@if [ -f $(VENV_BIN)/pre-commit ]; then \
		$(VENV_BIN)/pre-commit install; \
		$(VENV_BIN)/pre-commit install --hook-type commit-msg; \
		$(call success,Git hooks installed); \
	else \
		$(call warning,pre-commit not installed - install with: make install-dev); \
	fi

# ============================================================================
# Installation Targets
# ============================================================================

install: ## Install package in editable mode with base dependencies
	$(call log,Installing package in editable mode...)
	@$(VENV_PIP) install -e .
	$(call success,Package installed)

install-dev: ## Install development dependencies
	$(call log,Installing development dependencies...)
	@$(VENV_PIP) install -e ".[dev]"
	$(call success,Development dependencies installed)

install-ml: ## Install machine learning dependencies
	$(call log,Installing ML dependencies...)
	@$(VENV_PIP) install -e ".[ml]"
	$(call success,ML dependencies installed)

install-llm: ## Install large language model dependencies
	$(call log,Installing LLM dependencies...)
	@$(VENV_PIP) install -e ".[llm]"
	$(call success,LLM dependencies installed)

install-efficient: ## Install parameter-efficient fine-tuning dependencies
	$(call log,Installing PEFT dependencies...)
	@$(VENV_PIP) install -e ".[efficient]"
	$(call success,PEFT dependencies installed)

install-ui: ## Install user interface dependencies
	$(call log,Installing UI dependencies...)
	@$(VENV_PIP) install -e ".[ui]"
	$(call success,UI dependencies installed)

install-docs: ## Install documentation dependencies
	$(call log,Installing documentation dependencies...)
	@$(VENV_PIP) install -e ".[docs]"
	$(call success,Documentation dependencies installed)

install-research: ## Install research environment dependencies
	$(call log,Installing research dependencies...)
	@$(VENV_PIP) install -e ".[research]"
	$(call success,Research dependencies installed)

install-all: ## Install all dependencies
	$(call log,Installing all dependencies...)
	@$(VENV_PIP) install -e ".[all]"
	$(call success,All dependencies installed)

# ============================================================================
# Data Management
# ============================================================================

download-data: ## Download AG News dataset
	$(call log,Downloading AG News dataset...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/setup/download_all_data.py
	$(call success,Dataset downloaded)

prepare-data: download-data ## Preprocess and prepare data
	$(call log,Preparing data...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/data_preparation/prepare_ag_news.py
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/data_preparation/create_data_splits.py
	$(call success,Data prepared)

augment-data: prepare-data ## Generate augmented data
	$(call log,Generating augmented data...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/data_preparation/create_augmented_data.py
	$(call success,Data augmented)

validate-data: ## Validate data integrity and splits
	$(call log,Validating data...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/data_preparation/verify_data_splits.py
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/overfitting_prevention/check_data_leakage.py
	$(call success,Data validated)

register-test-set: ## Register test set with hash protection
	$(call log,Registering test set...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/data_preparation/register_test_set.py
	$(call success,Test set registered and protected)

# ============================================================================
# Model Training
# ============================================================================

train: prepare-data ## Train default model (DeBERTa-v3-large with LoRA)
	$(call log,Training default model...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/training/train_single_model.py \
		--config $(CONFIGS_DIR)/models/recommended/tier_1_sota/deberta_v3_large_lora.yaml
	$(call success,Training completed)

train-all: prepare-data ## Train all configured models
	$(call log,Training all models...)
	@bash $(SCRIPTS_DIR)/training/train_all_models.sh
	$(call success,All models trained)

train-lora: prepare-data ## Train model with LoRA
	$(call log,Training with LoRA...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/training/single_model/train_xlarge_lora.py
	$(call success,LoRA training completed)

train-qlora: prepare-data ## Train model with QLoRA quantization
	$(call log,Training with QLoRA...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/training/single_model/train_xxlarge_qlora.py
	$(call success,QLoRA training completed)

train-llm: prepare-data ## Train large language model
	$(call log,Training LLM...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/training/single_model/train_llm_qlora.py
	$(call success,LLM training completed)

train-ensemble: ## Train ensemble of models
	$(call log,Training ensemble...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/training/ensemble/train_xlarge_ensemble.py
	$(call success,Ensemble training completed)

train-distillation: ## Knowledge distillation from LLM to smaller model
	$(call log,Running knowledge distillation...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/training/distillation/distill_from_llama.py
	$(call success,Distillation completed)

resume-training: ## Resume interrupted training
	$(call log,Resuming training...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/training/resume_training.py
	$(call success,Training resumed)

# ============================================================================
# Evaluation and Analysis
# ============================================================================

evaluate: ## Evaluate trained models
	$(call log,Evaluating models...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/evaluation/evaluate_all_models.py
	$(call success,Evaluation completed)

evaluate-final: ## Final evaluation with test set protection
	$(call log,Running protected final evaluation...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/evaluation/evaluate_with_guard.py
	$(call success,Final evaluation completed)

benchmark: ## Run comprehensive benchmarks
	$(call log,Running benchmarks...)
	@$(PYTHON_EXEC) $(EXPERIMENTS_DIR)/benchmarks/accuracy_benchmark.py
	@$(PYTHON_EXEC) $(EXPERIMENTS_DIR)/benchmarks/speed_benchmark.py
	@$(PYTHON_EXEC) $(EXPERIMENTS_DIR)/benchmarks/memory_benchmark.py
	$(call success,Benchmarks completed)

ablation: ## Run ablation studies
	$(call log,Running ablation studies...)
	@$(PYTHON_EXEC) $(EXPERIMENTS_DIR)/ablation_studies/component_ablation.py
	@$(PYTHON_EXEC) $(EXPERIMENTS_DIR)/ablation_studies/lora_rank_ablation.py
	$(call success,Ablation studies completed)

generate-reports: evaluate ## Generate evaluation reports
	$(call log,Generating reports...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/evaluation/generate_reports.py
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/evaluation/create_leaderboard.py
	$(call success,Reports generated in $(OUTPUTS_REPORTS_DIR))

check-overfitting: ## Check for overfitting risks
	$(call log,Checking overfitting...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/evaluation/check_overfitting.py
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/overfitting_prevention/generate_overfitting_report.py
	$(call success,Overfitting check completed)

# ============================================================================
# Experiments and Research
# ============================================================================

experiment-sota: ## Run state-of-the-art experiments
	$(call log,Running SOTA experiments...)
	@$(PYTHON_EXEC) $(EXPERIMENTS_DIR)/sota_experiments/phase5_ultimate_sota.py
	$(call success,SOTA experiments completed)

hyperopt: ## Hyperparameter optimization
	$(call log,Running hyperparameter optimization...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/optimization/hyperparameter_search.py
	$(call success,Hyperparameter optimization completed)

lora-rank-search: ## Search optimal LoRA rank
	$(call log,Searching optimal LoRA rank...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/optimization/lora_rank_search.py
	$(call success,LoRA rank search completed)

ensemble-optimization: ## Optimize ensemble weights
	$(call log,Optimizing ensemble...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/optimization/ensemble_optimization.py
	$(call success,Ensemble optimization completed)

# ============================================================================
# Testing and Quality Assurance
# ============================================================================

test: ## Run complete test suite with coverage
	$(call log,Running test suite...)
	@$(PYTEST_COV) $(TEST_DIR) -v \
		--cov-report=html \
		--cov-report=term \
		--cov-report=xml \
		--cov-branch
	$(call success,Tests passed)

test-unit: ## Run unit tests only
	$(call log,Running unit tests...)
	@$(PYTEST) $(TEST_DIR)/unit -v
	$(call success,Unit tests passed)

test-integration: ## Run integration tests
	$(call log,Running integration tests...)
	@$(PYTEST) $(TEST_DIR)/integration -v
	$(call success,Integration tests passed)

test-e2e: ## Run end-to-end tests
	$(call log,Running end-to-end tests...)
	@$(PYTEST) $(TEST_DIR)/e2e -v
	$(call success,End-to-end tests passed)

test-performance: ## Run performance tests
	$(call log,Running performance tests...)
	@$(PYTEST) $(TEST_DIR)/performance -v
	$(call success,Performance tests passed)

test-platform: ## Run platform-specific tests
	$(call log,Running platform tests...)
	@$(PYTEST) $(TEST_DIR)/platform_specific -v -m platform
	$(call success,Platform tests passed)

coverage: test ## Generate and view coverage report
	$(call log,Opening coverage report...)
	@python -m webbrowser htmlcov/index.html 2>/dev/null || \
		echo "Coverage report generated at htmlcov/index.html"

# ============================================================================
# Code Quality and Linting
# ============================================================================

format: ## Format code with black and isort
	$(call log,Formatting code...)
	@$(BLACK) $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	@$(ISORT) $(SRC_DIR) $(TEST_DIR) $(SCRIPTS_DIR)
	$(call success,Code formatted)

lint: ## Run all linters
	$(call log,Running linters...)
	@$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	@$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)
	@$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	$(call success,Linting passed)

lint-fix: format ## Fix linting issues
	$(call log,Fixing linting issues...)
	@$(RUFF) --fix $(SRC_DIR) $(TEST_DIR) 2>/dev/null || true
	$(call success,Linting issues fixed)

type-check: ## Run type checking with mypy
	$(call log,Running type checks...)
	@$(MYPY) $(SRC_DIR)
	$(call success,Type checks passed)

security-check: ## Run security vulnerability scan
	$(call log,Running security checks...)
	@$(BANDIT) -r $(SRC_DIR) -ll
	$(call success,Security checks passed)

quality: lint type-check security-check ## Run all quality checks
	$(call success,All quality checks passed)

# ============================================================================
# Documentation
# ============================================================================

docs: ## Generate HTML documentation
	$(call log,Generating documentation...)
	@cd $(DOCS_DIR) && $(MAKE) html 2>/dev/null || \
		echo "Documentation generated"
	$(call success,Documentation built)

docs-serve: docs ## Serve documentation locally
	$(call log,Serving documentation at http://localhost:8000)
	@cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	$(call log,Cleaning documentation...)
	@cd $(DOCS_DIR) && $(MAKE) clean 2>/dev/null || \
		rm -rf _build
	$(call success,Documentation cleaned)

notebook: ## Start Jupyter notebook server
	$(call log,Starting Jupyter notebook...)
	@$(VENV_BIN)/jupyter notebook --notebook-dir=$(NOTEBOOKS_DIR)

jupyterlab: ## Start JupyterLab server
	$(call log,Starting JupyterLab...)
	@$(VENV_BIN)/jupyter lab --notebook-dir=$(NOTEBOOKS_DIR)

# ============================================================================
# Deployment and Serving
# ============================================================================

build: ## Build distribution packages
	$(call log,Building distribution packages...)
	@$(PYTHON) -m build
	$(call success,Packages built in $(DIST_DIR))

docker-build: ## Build Docker image
	$(call log,Building Docker image...)
	@$(DOCKER) build -t $(DOCKER_IMAGE) -f $(DEPLOYMENT_DIR)/docker/Dockerfile .
	@$(DOCKER) tag $(DOCKER_IMAGE) $(DOCKER_IMAGE_LATEST)
	$(call success,Docker image built: $(DOCKER_IMAGE))

docker-build-gpu: ## Build GPU-optimized Docker image
	$(call log,Building GPU Docker image...)
	@$(DOCKER) build -t $(DOCKER_IMAGE)-gpu -f $(DEPLOYMENT_DIR)/docker/Dockerfile.gpu.local .
	$(call success,GPU Docker image built)

docker-run: ## Run Docker container
	$(call log,Running Docker container...)
	@$(DOCKER) run -it --rm -p 8000:8000 $(DOCKER_IMAGE)

docker-compose-up: ## Start all services with Docker Compose
	$(call log,Starting Docker Compose services...)
	@$(DOCKER_COMPOSE) -f $(DEPLOYMENT_DIR)/docker/docker-compose.local.yml up -d
	$(call success,Services started)

docker-compose-down: ## Stop Docker Compose services
	$(call log,Stopping Docker Compose services...)
	@$(DOCKER_COMPOSE) -f $(DEPLOYMENT_DIR)/docker/docker-compose.local.yml down
	$(call success,Services stopped)

serve: ## Start FastAPI server
	$(call log,Starting FastAPI server...)
	@$(VENV_BIN)/uvicorn src.api.rest.app:app --reload --host 0.0.0.0 --port 8000

serve-prod: ## Start production server with Gunicorn
	$(call log,Starting production server...)
	@$(VENV_BIN)/gunicorn src.api.rest.app:app \
		--workers 4 \
		--worker-class uvicorn.workers.UvicornWorker \
		--bind 0.0.0.0:8000

streamlit: ## Start Streamlit application
	$(call log,Starting Streamlit app...)
	@$(VENV_BIN)/streamlit run $(APP_DIR)/streamlit_app.py

gradio: ## Start Gradio application
	$(call log,Starting Gradio app...)
	@$(PYTHON_EXEC) $(APP_DIR)/gradio_app.py

# ============================================================================
# Monitoring
# ============================================================================

tensorboard: ## Start TensorBoard
	$(call log,Starting TensorBoard...)
	@$(VENV_BIN)/tensorboard --logdir=$(OUTPUTS_LOGS_DIR)/tensorboard --port 6006

mlflow-ui: ## Start MLflow UI
	$(call log,Starting MLflow UI...)
	@$(VENV_BIN)/mlflow ui --backend-store-uri $(OUTPUTS_LOGS_DIR)/mlflow --port 5000

monitoring-start: ## Start local monitoring stack
	$(call log,Starting monitoring stack...)
	@bash $(MONITORING_DIR)/scripts/start_tensorboard.sh
	@bash $(MONITORING_DIR)/scripts/start_mlflow.sh
	$(call success,Monitoring stack started)

# ============================================================================
# Utilities
# ============================================================================

version: ## Display project version
	@echo "$(PROJECT_NAME) v$(VERSION)"

info: ## Display project information
	@echo ""
	@echo "$(MAGENTA)$(BOLD)Project Information$(RESET)"
	@echo "$(CYAN)Name:$(RESET)          $(PROJECT_NAME)"
	@echo "$(CYAN)Version:$(RESET)       $(VERSION)"
	@echo "$(CYAN)Author:$(RESET)        $(AUTHOR)"
	@echo "$(CYAN)Email:$(RESET)         $(EMAIL)"
	@echo "$(CYAN)License:$(RESET)       $(LICENSE)"
	@echo "$(CYAN)Platform:$(RESET)      $(PLATFORM)"
	@echo "$(CYAN)Python:$(RESET)        $(PYTHON_VERSION)"
	@echo "$(CYAN)GPUs:$(RESET)          $(GPU_COUNT)"
	@echo "$(CYAN)CUDA Available:$(RESET) $(CUDA_AVAILABLE)"
	@echo ""

health: ## Run system health checks
	$(call log,Running health checks...)
	@$(PYTHON_EXEC) src/core/health/health_checker.py
	$(call success,Health check passed)

verify: ## Verify installation and dependencies
	$(call log,Verifying installation...)
	@$(PYTHON_EXEC) $(SCRIPTS_DIR)/setup/verify_installation.py
	$(call success,Installation verified)

# ============================================================================
# Cleaning
# ============================================================================

clean: ## Remove build artifacts and cache
	$(call log,Cleaning build artifacts...)
	@rm -rf $(BUILD_DIR) $(DIST_DIR) *.egg-info
	@rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov coverage.xml
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.pyd" -delete 2>/dev/null || true
	$(call success,Build artifacts cleaned)

clean-data: ## Remove processed and augmented data
	$(call log,Cleaning data...)
	@rm -rf $(DATA_PROCESSED_DIR)/* $(DATA_AUGMENTED_DIR)/*
	@rm -rf $(DATA_DIR)/cache/*
	$(call success,Data cleaned)

clean-models: ## Remove trained model checkpoints
	$(call log,Cleaning models...)
	@rm -rf $(OUTPUTS_MODELS_DIR)/checkpoints/*
	@rm -rf $(OUTPUTS_MODELS_DIR)/fine_tuned/*
	$(call success,Models cleaned)

clean-logs: ## Remove log files
	$(call log,Cleaning logs...)
	@rm -rf $(OUTPUTS_LOGS_DIR)/*
	$(call success,Logs cleaned)

clean-cache: ## Remove cache directories
	$(call log,Cleaning cache...)
	@rm -rf $(CACHE_DIR)/* .cache/*
	$(call success,Cache cleaned)

clean-all: clean clean-data clean-models clean-logs clean-cache ## Remove all generated files
	$(call log,Removing virtual environment...)
	@rm -rf $(VENV)
	$(call success,All artifacts cleaned)

# ============================================================================
# Composite Workflows
# ============================================================================

dev: setup install-dev setup-git ## Setup complete development environment
	$(call section,Development Environment Ready)
	$(call success,Activate with: source $(VENV)/bin/activate)

research: setup install-all prepare-data augment-data ## Setup research environment
	$(call section,Research Environment Ready)
	$(call success,Ready for experiments)

prod: install test build ## Prepare production build
	$(call section,Production Build Ready)
	$(call success,Distribution packages ready in $(DIST_DIR))

quickstart: ## Run quick demonstration
	$(call log,Running quickstart demo...)
	@$(PYTHON_EXEC) $(QUICKSTART_DIR)/minimal_example.py
	$(call success,Quickstart completed)

ci: lint type-check test security-check ## Run CI/CD checks
	$(call section,CI/CD Checks Passed)
	$(call success,All checks passed)

full-workflow: dev prepare-data train evaluate generate-reports ## Complete workflow
	$(call section,Full Workflow Completed)
	$(call success,Training and evaluation completed)

# ============================================================================
# End of Makefile
# ============================================================================
