# ============================================================================
# Makefile for AG News Text Classification
# ============================================================================
# Project: AG News Text Classification (ag-news-text-classification)
# Description: Build automation and development workflow management
# Author: Võ Hải Dũng
# Email: vohaidung.work@gmail.com
# License: MIT
# ============================================================================
# This Makefile provides comprehensive automation for:
# - Development environment setup
# - Dependency installation
# - Data preparation and augmentation
# - Model training and evaluation
# - Experiment tracking and benchmarking
# - Testing and quality assurance
# - Documentation generation
# - Deployment and serving
# - Monitoring and profiling
#
# Quick Start:
#   make help           Show all available targets
#   make setup          Complete development setup
#   make train          Train default model
#   make test           Run all tests
#   make docs           Generate documentation
#
# Common Workflows:
#   Development:  make dev
#   Research:     make research
#   Production:   make prod
#   Quick Demo:   make quickstart
# ============================================================================

.PHONY: help
.DEFAULT_GOAL := help

# ============================================================================
# Configuration Variables
# ============================================================================

# Project metadata
PROJECT_NAME := ag-news-text-classification
PROJECT_SLUG := ag_news_text_classification
VERSION := $(shell python -c "exec(open('src/__version__.py').read()); print(__version__)" 2>/dev/null || echo "1.0.0")
AUTHOR := Võ Hải Dũng
EMAIL := vohaidung.work@gmail.com

# Python configuration
PYTHON_VERSION := 3.10
PYTHON := python$(PYTHON_VERSION)
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
VENV := venv
VENV_BIN := $(VENV)/bin
VENV_PYTHON := $(VENV_BIN)/python
VENV_PIP := $(VENV_BIN)/pip

# Directory structure
ROOT_DIR := $(shell pwd)
SRC_DIR := $(ROOT_DIR)/src
TEST_DIR := $(ROOT_DIR)/tests
DATA_DIR := $(ROOT_DIR)/data
CONFIGS_DIR := $(ROOT_DIR)/configs
OUTPUTS_DIR := $(ROOT_DIR)/outputs
DOCS_DIR := $(ROOT_DIR)/docs
SCRIPTS_DIR := $(ROOT_DIR)/scripts
NOTEBOOKS_DIR := $(ROOT_DIR)/notebooks
CACHE_DIR := $(ROOT_DIR)/.cache
BUILD_DIR := $(ROOT_DIR)/build
DIST_DIR := $(ROOT_DIR)/dist

# Output directories
MODELS_DIR := $(OUTPUTS_DIR)/models
RESULTS_DIR := $(OUTPUTS_DIR)/results
LOGS_DIR := $(OUTPUTS_DIR)/logs
REPORTS_DIR := $(OUTPUTS_DIR)/reports

# Data directories
RAW_DATA_DIR := $(DATA_DIR)/raw
PROCESSED_DATA_DIR := $(DATA_DIR)/processed
AUGMENTED_DATA_DIR := $(DATA_DIR)/augmented

# Tool configurations
BLACK := $(VENV_BIN)/black
ISORT := $(VENV_BIN)/isort
FLAKE8 := $(VENV_BIN)/flake8
MYPY := $(VENV_BIN)/mypy
PYLINT := $(VENV_BIN)/pylint
BANDIT := $(VENV_BIN)/bandit

# Docker configuration
DOCKER := docker
DOCKER_COMPOSE := docker-compose
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
DOCKER_REGISTRY := docker.io/$(AUTHOR)

# CUDA configuration
CUDA_VERSION := 11.8
GPU_COUNT := $(shell nvidia-smi -L 2>/dev/null | wc -l)

# Timestamp
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

# ============================================================================
# Color Output
# ============================================================================

# ANSI color codes
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
RESET := \033[0m

# ============================================================================
# Helper Functions
# ============================================================================

# Print colored messages
define log
	@echo "$(CYAN)[$(shell date +%H:%M:%S)]$(RESET) $(1)"
endef

define success
	@echo "$(GREEN)SUCCESS:$(RESET) $(1)"
endef

define warning
	@echo "$(YELLOW)WARNING:$(RESET) $(1)"
endef

define error
	@echo "$(RED)ERROR:$(RESET) $(1)"
endef

# ============================================================================
# Help Target
# ============================================================================

help:
	@echo ""
	@echo "$(MAGENTA)AG News Text Classification Framework$(RESET)"
	@echo "$(CYAN)Version:$(RESET) $(VERSION)"
	@echo "$(CYAN)Author:$(RESET) $(AUTHOR)"
	@echo "$(CYAN)Email:$(RESET) $(EMAIL)"
	@echo ""
	@echo "$(YELLOW)Usage:$(RESET)"
	@echo "  make [target]"
	@echo ""
	@echo "$(YELLOW)Available targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-25s$(RESET) %s\n", $$1, $$2}' | \
		sort
	@echo ""

# ============================================================================
# Setup and Installation
# ============================================================================

.PHONY: setup
setup: setup-dirs setup-venv install-dev setup-git download-data ## Complete development setup
	$(call success,"Development environment ready!")

.PHONY: setup-dirs
setup-dirs: ## Create project directory structure
	$(call log,"Creating directory structure...")
	@mkdir -p $(DATA_DIR)/{raw,processed,augmented,external,cache}
	@mkdir -p $(OUTPUTS_DIR)/{models,results,logs,reports}
	@mkdir -p $(CACHE_DIR)
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(DIST_DIR)
	$(call success,"Directories created")

.PHONY: setup-venv
setup-venv: ## Create Python virtual environment
	$(call log,"Creating virtual environment...")
	@$(PYTHON) -m venv $(VENV)
	@$(VENV_PIP) install --upgrade pip setuptools wheel
	$(call success,"Virtual environment created at $(VENV)")

.PHONY: setup-git
setup-git: ## Setup Git hooks
	$(call log,"Installing Git hooks...")
	@$(VENV_BIN)/pre-commit install 2>/dev/null || $(call warning,"pre-commit not installed")
	@$(VENV_BIN)/pre-commit install --hook-type commit-msg 2>/dev/null || true
	$(call success,"Git hooks installed")

.PHONY: install
install: ## Install package in editable mode
	$(call log,"Installing package...")
	@$(VENV_PIP) install -e .
	$(call success,"Package installed")

.PHONY: install-dev
install-dev: ## Install development dependencies
	$(call log,"Installing development dependencies...")
	@$(VENV_PIP) install -e ".[dev]"
	$(call success,"Development dependencies installed")

.PHONY: install-ml
install-ml: ## Install ML dependencies
	$(call log,"Installing ML dependencies...")
	@$(VENV_PIP) install -e ".[ml]"
	$(call success,"ML dependencies installed")

.PHONY: install-all
install-all: ## Install all dependencies
	$(call log,"Installing all dependencies...")
	@$(VENV_PIP) install -e ".[all]"
	$(call success,"All dependencies installed")

# ============================================================================
# Data Management
# ============================================================================

.PHONY: download-data
download-data: ## Download AG News dataset
	$(call log,"Downloading AG News dataset...")
	@$(VENV_PYTHON) scripts/setup/download_all_data.py
	$(call success,"Data downloaded")

.PHONY: prepare-data
prepare-data: download-data ## Prepare and preprocess data
	$(call log,"Preparing data...")
	@$(VENV_PYTHON) scripts/data_preparation/prepare_ag_news.py
	$(call success,"Data prepared")

.PHONY: augment-data
augment-data: prepare-data ## Generate augmented data
	$(call log,"Generating augmented data...")
	@$(VENV_PYTHON) scripts/data_preparation/create_augmented_data.py
	$(call success,"Data augmented")

.PHONY: validate-data
validate-data: ## Validate data integrity
	$(call log,"Validating data...")
	@$(VENV_PYTHON) scripts/data_preparation/verify_data_splits.py
	$(call success,"Data validated")

# ============================================================================
# Training
# ============================================================================

.PHONY: train
train: prepare-data ## Train default model (DeBERTa-v3-large)
	$(call log,"Training DeBERTa-v3-large model...")
	@$(VENV_PYTHON) scripts/training/train_single_model.py \
		--config configs/models/recommended/tier_1_sota/deberta_v3_large_lora.yaml
	$(call success,"Training completed")

.PHONY: train-all
train-all: prepare-data ## Train all models
	$(call log,"Training all models...")
	@bash scripts/training/train_all_models.sh
	$(call success,"All models trained")

.PHONY: train-ensemble
train-ensemble: train-all ## Train ensemble models
	$(call log,"Training ensemble...")
	@$(VENV_PYTHON) scripts/training/ensemble/train_xlarge_ensemble.py
	$(call success,"Ensemble trained")

.PHONY: train-lora
train-lora: prepare-data ## Train with LoRA
	$(call log,"Training with LoRA...")
	@$(VENV_PYTHON) scripts/training/single_model/train_xlarge_lora.py
	$(call success,"LoRA training completed")

.PHONY: train-qlora
train-qlora: prepare-data ## Train with QLoRA
	$(call log,"Training with QLoRA...")
	@$(VENV_PYTHON) scripts/training/single_model/train_llm_qlora.py
	$(call success,"QLoRA training completed")

# ============================================================================
# Evaluation
# ============================================================================

.PHONY: evaluate
evaluate: ## Evaluate trained models
	$(call log,"Evaluating models...")
	@$(VENV_PYTHON) scripts/evaluation/evaluate_all_models.py
	$(call success,"Evaluation completed")

.PHONY: evaluate-final
evaluate-final: ## Final evaluation with guards
	$(call log,"Running final evaluation...")
	@$(VENV_PYTHON) scripts/evaluation/evaluate_with_guard.py
	$(call success,"Final evaluation completed")

.PHONY: benchmark
benchmark: ## Run benchmarks
	$(call log,"Running benchmarks...")
	@$(VENV_PYTHON) experiments/benchmarks/accuracy_benchmark.py
	@$(VENV_PYTHON) experiments/benchmarks/speed_benchmark.py
	$(call success,"Benchmarks completed")

.PHONY: report
report: evaluate ## Generate evaluation reports
	$(call log,"Generating reports...")
	@$(VENV_PYTHON) scripts/evaluation/generate_reports.py
	$(call success,"Reports generated in $(REPORTS_DIR)")

# ============================================================================
# Experiments
# ============================================================================

.PHONY: experiment
experiment: ## Run SOTA experiments
	$(call log,"Running SOTA experiments...")
	@$(VENV_PYTHON) experiments/sota_experiments/phase5_ultimate_sota.py
	$(call success,"Experiments completed")

.PHONY: ablation
ablation: ## Run ablation studies
	$(call log,"Running ablation studies...")
	@$(VENV_PYTHON) experiments/ablation_studies/component_ablation.py
	$(call success,"Ablation studies completed")

.PHONY: hyperopt
hyperopt: ## Hyperparameter optimization
	$(call log,"Running hyperparameter optimization...")
	@$(VENV_PYTHON) scripts/optimization/hyperparameter_search.py
	$(call success,"Optimization completed")

# ============================================================================
# Testing
# ============================================================================

.PHONY: test
test: ## Run all tests
	$(call log,"Running tests...")
	@$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term
	$(call success,"Tests passed")

.PHONY: test-unit
test-unit: ## Run unit tests
	$(call log,"Running unit tests...")
	@$(PYTEST) tests/unit/ -v
	$(call success,"Unit tests passed")

.PHONY: test-integration
test-integration: ## Run integration tests
	$(call log,"Running integration tests...")
	@$(PYTEST) tests/integration/ -v
	$(call success,"Integration tests passed")

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	$(call log,"Running E2E tests...")
	@$(PYTEST) tests/e2e/ -v
	$(call success,"E2E tests passed")

.PHONY: coverage
coverage: test ## Generate coverage report
	@echo "Opening coverage report..."
	@python -m webbrowser htmlcov/index.html 2>/dev/null || true

# ============================================================================
# Code Quality
# ============================================================================

.PHONY: format
format: ## Format code with black and isort
	$(call log,"Formatting code...")
	@$(BLACK) $(SRC_DIR) $(TEST_DIR)
	@$(ISORT) $(SRC_DIR) $(TEST_DIR)
	$(call success,"Code formatted")

.PHONY: lint
lint: ## Run all linters
	$(call log,"Running linters...")
	@$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	@$(ISORT) --check-only $(SRC_DIR) $(TEST_DIR)
	@$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	@$(MYPY) $(SRC_DIR)
	$(call success,"Linting passed")

.PHONY: type-check
type-check: ## Run type checking
	$(call log,"Running type check...")
	@$(MYPY) $(SRC_DIR)
	$(call success,"Type check passed")

.PHONY: security
security: ## Run security checks
	$(call log,"Running security checks...")
	@$(BANDIT) -r $(SRC_DIR)
	$(call success,"Security check passed")

# ============================================================================
# Documentation
# ============================================================================

.PHONY: docs
docs: ## Generate documentation
	$(call log,"Generating documentation...")
	@cd $(DOCS_DIR) && make html
	$(call success,"Documentation generated")

.PHONY: docs-serve
docs-serve: docs ## Serve documentation locally
	$(call log,"Serving documentation...")
	@cd $(DOCS_DIR)/_build/html && python -m http.server 8000

.PHONY: notebook
notebook: ## Start Jupyter notebook
	$(call log,"Starting Jupyter notebook...")
	@$(VENV_BIN)/jupyter notebook --notebook-dir=$(NOTEBOOKS_DIR)

# ============================================================================
# Deployment
# ============================================================================

.PHONY: build
build: ## Build distribution packages
	$(call log,"Building package...")
	@$(VENV_PYTHON) -m build
	$(call success,"Package built in $(DIST_DIR)")

.PHONY: docker-build
docker-build: ## Build Docker image
	$(call log,"Building Docker image...")
	@$(DOCKER) build -t $(DOCKER_IMAGE) .
	$(call success,"Docker image built: $(DOCKER_IMAGE)")

.PHONY: docker-run
docker-run: ## Run Docker container
	$(call log,"Running Docker container...")
	@$(DOCKER) run -it --rm -p 8000:8000 $(DOCKER_IMAGE)

.PHONY: serve
serve: ## Start API server
	$(call log,"Starting API server...")
	@$(VENV_BIN)/uvicorn src.api.rest.app:app --reload --host 0.0.0.0 --port 8000

.PHONY: streamlit
streamlit: ## Start Streamlit app
	$(call log,"Starting Streamlit app...")
	@$(VENV_BIN)/streamlit run app/streamlit_app.py

# ============================================================================
# Cleaning
# ============================================================================

.PHONY: clean
clean: ## Remove build artifacts
	$(call log,"Cleaning build artifacts...")
	@rm -rf $(BUILD_DIR) $(DIST_DIR) *.egg-info
	@rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	$(call success,"Cleaned")

.PHONY: clean-data
clean-data: ## Remove processed data
	$(call log,"Cleaning data...")
	@rm -rf $(PROCESSED_DATA_DIR)/* $(AUGMENTED_DATA_DIR)/*
	$(call success,"Data cleaned")

.PHONY: clean-models
clean-models: ## Remove trained models
	$(call log,"Cleaning models...")
	@rm -rf $(MODELS_DIR)/*
	$(call success,"Models cleaned")

.PHONY: clean-all
clean-all: clean clean-data clean-models ## Remove all generated files
	$(call log,"Cleaning everything...")
	@rm -rf $(VENV)
	$(call success,"Everything cleaned")

# ============================================================================
# Utility Targets
# ============================================================================

.PHONY: version
version: ## Show project version
	@echo "$(PROJECT_NAME) v$(VERSION)"

.PHONY: info
info: ## Show project information
	@echo ""
	@echo "$(MAGENTA)Project Information$(RESET)"
	@echo "$(CYAN)Name:$(RESET)         $(PROJECT_NAME)"
	@echo "$(CYAN)Version:$(RESET)      $(VERSION)"
	@echo "$(CYAN)Author:$(RESET)       $(AUTHOR)"
	@echo "$(CYAN)Email:$(RESET)        $(EMAIL)"
	@echo "$(CYAN)Python:$(RESET)       $(PYTHON_VERSION)"
	@echo "$(CYAN)GPU Count:$(RESET)    $(GPU_COUNT)"
	@echo "$(CYAN)CUDA Version:$(RESET) $(CUDA_VERSION)"
	@echo ""

.PHONY: health
health: ## Run health checks
	$(call log,"Running health checks...")
	@$(VENV_PYTHON) src/core/health/health_checker.py
	$(call success,"Health check passed")

# ============================================================================
# Workflow Targets
# ============================================================================

.PHONY: dev
dev: setup install-dev setup-git ## Setup development environment
	$(call success,"Development environment ready!")
	@echo "Activate with: source $(VENV)/bin/activate"

.PHONY: research
research: setup install-all prepare-data augment-data ## Setup research environment
	$(call success,"Research environment ready!")

.PHONY: prod
prod: install test build ## Prepare for production
	$(call success,"Production build ready!")

.PHONY: quickstart
quickstart: ## Quick start demo
	$(call log,"Running quick start...")
	@$(VENV_PYTHON) quickstart/minimal_example.py
	$(call success,"Quick start completed!")

.PHONY: ci
ci: lint test security ## Run CI checks
	$(call success,"CI checks passed!")

# ============================================================================
# End of Makefile
# ============================================================================
