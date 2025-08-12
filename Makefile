SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:
.SUFFIXES:
.SECONDARY:
.PHONY: all

MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --warn-undefined-variables

ifndef SEQUENTIAL
	MAKEFLAGS += -j$(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
endif

PROJECT_NAME := ag-news-text-classification
VERSION := $(shell python -c "exec(open('src/__version__.py').read()); print(__version__)" 2>/dev/null || echo "1.0.0")
PYTHON_VERSION := 3.10
VENV_NAME := venv
CURRENT_DATE := $(shell date +%Y%m%d_%H%M%S)
TIMESTAMP := $(shell date +%s)

ROOT_DIR := $(shell pwd)
SRC_DIR := $(ROOT_DIR)/src
DATA_DIR := $(ROOT_DIR)/data
CONFIGS_DIR := $(ROOT_DIR)/configs
NOTEBOOKS_DIR := $(ROOT_DIR)/notebooks
TESTS_DIR := $(ROOT_DIR)/tests
DOCS_DIR := $(ROOT_DIR)/docs
OUTPUTS_DIR := $(ROOT_DIR)/outputs
SCRIPTS_DIR := $(ROOT_DIR)/scripts
DEPLOYMENT_DIR := $(ROOT_DIR)/deployment
CACHE_DIR := $(ROOT_DIR)/.cache
BUILD_DIR := $(ROOT_DIR)/build
DIST_DIR := $(ROOT_DIR)/dist
BACKUP_DIR := $(ROOT_DIR)/.backups

PYTHON := python$(PYTHON_VERSION)
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy
PYLINT := $(PYTHON) -m pylint
BANDIT := $(PYTHON) -m bandit
SAFETY := $(PYTHON) -m safety
COVERAGE := $(PYTHON) -m coverage
SPHINX := $(PYTHON) -m sphinx
PRE_COMMIT := pre-commit

DOCKER := docker
DOCKER_COMPOSE := docker-compose
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
DOCKER_REGISTRY := agnews-research
DOCKER_TAG := $(DOCKER_REGISTRY)/$(DOCKER_IMAGE)

CUDA_VERSION ?= 11.8
GPU_COUNT := $(shell nvidia-smi -L 2>/dev/null | wc -l || echo 0)
CUDA_VISIBLE_DEVICES ?= 0,1,2,3,4,5,6,7

CLOUD_PROVIDER ?= aws
AWS_REGION ?= us-west-2
GCP_PROJECT ?= ag-news-research
AZURE_SUBSCRIPTION ?= ag-news-sub

ENV_FILE := .env
ENV_TEST_FILE := .env.test
ENV_PROD_FILE := .env.prod

RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
MAGENTA := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

LOG_DIR := $(OUTPUTS_DIR)/logs/make
LOG_FILE := $(LOG_DIR)/make_$(CURRENT_DATE).log
METRICS_FILE := $(CACHE_DIR)/make_metrics.json

BENCHMARK_DIR := $(OUTPUTS_DIR)/benchmarks
PROFILE_DIR := $(OUTPUTS_DIR)/profiles

CONFIG_FILE := $(CONFIGS_DIR)/build_config.yaml
SHARED_CONFIG := $(shell test -f $(CONFIG_FILE) && cat $(CONFIG_FILE) || echo "{}")

-include .env.local
-include .make.cache

define log
	@echo -e "$(CYAN)[$(shell date +%H:%M:%S)]$(RESET) $(1)" | tee -a $(LOG_FILE)
endef

define success
	@echo -e "$(GREEN)✓$(RESET) $(1)" | tee -a $(LOG_FILE)
endef

define warning
	@echo -e "$(YELLOW)⚠$(RESET) $(1)" | tee -a $(LOG_FILE)
endef

define error
	@echo -e "$(RED)✗$(RESET) $(1)" | tee -a $(LOG_FILE)
endef

define header
	@echo -e "\n$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@echo -e "$(BLUE)▶$(RESET) $(WHITE)$(1)$(RESET)"
	@echo -e "$(BLUE)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
endef

define check_command
	@command -v $(1) >/dev/null 2>&1 || { \
		echo -e "$(RED)Error: $(1) is not installed$(RESET)"; \
		exit 1; \
	}
endef

define create_dir
	@mkdir -p $(1) 2>/dev/null || true
endef

define with_progress
	@echo -n "$(1): "
	@for i in {1..10}; do \
		echo -n "▓"; \
		sleep 0.1; \
	done
	@echo " Done!"
endef

define save_metric
	@echo '{"target": "$(1)", "duration": $$(($(shell date +%s) - $(TIMESTAMP))), "timestamp": "$(CURRENT_DATE)"}' >> $(METRICS_FILE)
endef

define validate_env
	@$(PYTHON) -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" || { \
		$(call error,"Python 3.8+ required"); \
		exit 1; \
	}
	@test -d "$(SRC_DIR)" || { \
		$(call error,"Source directory not found"); \
		exit 1; \
	}
endef

define use_entry_point
	@ag-news-$(1) $(2)
endef

define parallel_exec
	@echo "$(1)" | xargs -P $(shell nproc) -I {} bash -c '{}'
endef

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	OS := linux
	OPEN_CMD := xdg-open
endif
ifeq ($(UNAME_S),Darwin)
	OS := macos
	OPEN_CMD := open
endif
ifeq ($(findstring MINGW,$(UNAME_S)),MINGW)
	OS := windows
	OPEN_CMD := start
endif

IN_CONTAINER := $(shell [ -f /.dockerenv ] && echo 1 || echo 0)
CI := $(shell [ ! -z "$${CI}" ] && echo 1 || echo 0)
HAS_GPU := $(shell [ $(GPU_COUNT) -gt 0 ] && echo 1 || echo 0)

ENTRY_POINTS := train evaluate data optimize serve monitor analyze experiment

.PHONY: all
all: validate clean install test build deploy
	$(call save_metric,"all")

.PHONY: help
help:
	@echo -e "\n$(MAGENTA)AG News Text Classification Framework - Makefile$(RESET)"
	@echo -e "$(CYAN)Version:$(RESET) $(VERSION)"
	@echo -e "$(CYAN)Python:$(RESET) $(PYTHON_VERSION)"
	@echo -e "$(CYAN)OS:$(RESET) $(OS)"
	@echo -e "$(CYAN)GPU Available:$(RESET) $(HAS_GPU) ($(GPU_COUNT) GPUs)"
	@echo -e "\n$(YELLOW)Usage:$(RESET)"
	@echo -e "  make [target] [VARIABLE=value ...]\n"
	@echo -e "$(YELLOW)Available targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-30s$(RESET) %s\n", $$1, $$2}' | \
		sort
	@echo -e "\n$(YELLOW)Entry Point Commands:$(RESET)"
	@for ep in $(ENTRY_POINTS); do \
		echo -e "  $(GREEN)ag-news-$$ep$(RESET) - Available as entry point"; \
	done
	@echo -e "\n$(YELLOW)Quick Commands:$(RESET)"
	@echo -e "  $(GREEN)make dev$(RESET)      - Development setup"
	@echo -e "  $(GREEN)make prod$(RESET)     - Production setup"
	@echo -e "  $(GREEN)make research$(RESET) - Research pipeline"
	@echo -e "  $(GREEN)make quickstart$(RESET) - Quick demo"

.PHONY: validate
validate:
	$(call header,"Validating Environment")
	$(call validate_env)
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || $(call warning,"PyTorch not installed")
	@$(PYTHON) -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || $(call warning,"Transformers not installed")
	$(call success,"Environment validated")

.PHONY: setup
setup: validate setup-dirs setup-venv install-base setup-git download-resources
	$(call success,"Setup complete!")
	$(call save_metric,"setup")

.PHONY: setup-dirs
setup-dirs:
	$(call log,"Creating directory structure...")
	@$(parallel_exec,"mkdir -p $(LOG_DIR) $(OUTPUTS_DIR)/models/checkpoints $(OUTPUTS_DIR)/models/pretrained $(OUTPUTS_DIR)/models/fine_tuned $(OUTPUTS_DIR)/models/ensembles $(OUTPUTS_DIR)/models/exported $(OUTPUTS_DIR)/results/experiments $(OUTPUTS_DIR)/analysis $(DATA_DIR)/raw/ag_news $(DATA_DIR)/processed/train $(DATA_DIR)/processed/validation $(DATA_DIR)/processed/test $(DATA_DIR)/augmented $(DATA_DIR)/external $(DATA_DIR)/cache $(BENCHMARK_DIR) $(PROFILE_DIR) $(CACHE_DIR)/setup $(BACKUP_DIR)")
	$(call success,"Directories created")

.PHONY: setup-venv
setup-venv:
	$(call log,"Setting up Python virtual environment...")
	@if [ ! -d "$(VENV_NAME)" ]; then \
		$(PYTHON) -m venv $(VENV_NAME); \
		$(call success,"Virtual environment created"); \
	else \
		$(call warning,"Virtual environment already exists"); \
	fi
	@echo "Activate with: source $(VENV_NAME)/bin/activate"

.PHONY: setup-gpu
setup-gpu:
	$(call header,"Setting up GPU Environment")
	@$(PYTHON) setup.py setup_gpu --cuda-version=$(CUDA_VERSION) --benchmark || $(call use_entry_point,setup --gpu)
	$(call save_metric,"setup-gpu")

.PHONY: setup-research
setup-research:
	$(call header,"Setting up Research Environment")
	@$(PIP) install -e ".[research]"
	@$(PYTHON) setup.py setup_research --download-models --setup-wandb --async
	$(call save_metric,"setup-research")

.PHONY: setup-production
setup-production:
	$(call header,"Setting up Production Environment")
	@$(PIP) install ".[prod]"
	@$(PYTHON) setup.py setup_production --cloud=$(CLOUD_PROVIDER) --monitoring --security
	$(call save_metric,"setup-production")

.PHONY: setup-mlops
setup-mlops:
	$(call header,"Setting up MLOps Environment")
	@$(PYTHON) setup.py setup_mlops --full
	$(call save_metric,"setup-mlops")

.PHONY: setup-database
setup-database:
	$(call header,"Setting up Database Environment")
	@$(PYTHON) setup.py setup_database --db-type=postgres --migrate --seed
	$(call save_metric,"setup-database")

.PHONY: setup-git
setup-git:
	$(call log,"Setting up git hooks...")
	@$(PRE_COMMIT) install 2>/dev/null || $(call warning,"Pre-commit not installed")
	@if [ -d ".husky" ]; then \
		npx husky install 2>/dev/null || true; \
	fi
	$(call success,"Git hooks installed")

.PHONY: install
install: install-base install-dev install-research
	$(call save_metric,"install")

.PHONY: install-base
install-base:
	$(call log,"Installing base dependencies...")
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -e .
	$(call success,"Base dependencies installed")

.PHONY: install-dev
install-dev:
	$(call log,"Installing development dependencies...")
	@$(PIP) install -e ".[dev]"
	$(call success,"Development dependencies installed")

.PHONY: install-research
install-research:
	$(call log,"Installing research dependencies...")
	@$(PIP) install -e ".[research]"
	$(call success,"Research dependencies installed")

.PHONY: install-prod
install-prod:
	$(call log,"Installing production dependencies...")
	@$(PIP) install ".[prod]"
	$(call success,"Production dependencies installed")

.PHONY: install-minimal
install-minimal:
	$(call log,"Installing minimal dependencies...")
	@$(PIP) install -e ".[minimal]"

.PHONY: install-all
install-all:
	$(call log,"Installing all dependencies...")
	@$(PIP) install -e ".[all]"
	$(call save_metric,"install-all")

.PHONY: install-week1
install-week1:
	@$(PIP) install -e ".[week1]"
	$(call success,"Week 1 dependencies installed")

.PHONY: install-week2-3
install-week2-3:
	@$(PIP) install -e ".[week2-3]"
	$(call success,"Week 2-3 dependencies installed")

.PHONY: install-week4-5
install-week4-5:
	@$(PIP) install -e ".[week4-5]"
	$(call success,"Week 4-5 dependencies installed")

.PHONY: install-week6-7
install-week6-7:
	@$(PIP) install -e ".[week6-7]"
	$(call success,"Week 6-7 dependencies installed")

.PHONY: install-week8-9
install-week8-9:
	@$(PIP) install -e ".[week8-9]"
	$(call success,"Week 8-9 dependencies installed")

.PHONY: install-week10
install-week10:
	@$(PIP) install -e ".[week10]"
	$(call success,"Week 10 dependencies installed")

.PHONY: data
data: data-download data-prepare data-augment data-validate
	$(call save_metric,"data")

.PHONY: data-download
data-download:
	$(call log,"Downloading AG News dataset...")
	@$(call use_entry_point,download) || $(PYTHON) scripts/setup/download_all_data.py
	$(call success,"Data downloaded")

.PHONY: data-prepare
data-prepare: data-download
	$(call log,"Preparing data...")
	@$(call use_entry_point,prepare) || $(PYTHON) scripts/data_preparation/prepare_ag_news.py
	$(call success,"Data prepared")

.PHONY: data-augment
data-augment: data-prepare
	$(call log,"Generating augmented data...")
	@$(call use_entry_point,augment) || $(PYTHON) scripts/data_preparation/create_augmented_data.py
	$(call success,"Data augmented")

.PHONY: data-contrast
data-contrast: data-prepare
	$(call log,"Generating contrast sets...")
	@$(call use_entry_point,contrast) || $(PYTHON) scripts/data_preparation/generate_contrast_sets.py

.PHONY: data-pseudo
data-pseudo: data-prepare
	$(call log,"Generating pseudo labels...")
	@$(call use_entry_point,pseudo) || $(PYTHON) scripts/data_preparation/generate_pseudo_labels.py

.PHONY: data-external
data-external:
	$(call log,"Preparing external data...")
	@$(call use_entry_point,prepare-external) || $(PYTHON) scripts/data_preparation/prepare_external_data.py

.PHONY: data-instruction
data-instruction: data-prepare
	$(call log,"Preparing instruction data...")
	@$(call use_entry_point,instruction-data) || $(PYTHON) scripts/data_preparation/prepare_instruction_data.py

.PHONY: data-quality
data-quality: data-prepare
	$(call log,"Selecting quality data...")
	@$(call use_entry_point,quality) || $(PYTHON) scripts/data_preparation/select_quality_data.py

.PHONY: data-splits
data-splits: data-prepare
	$(call log,"Creating data splits...")
	@$(call use_entry_point,splits) || $(PYTHON) scripts/data_preparation/create_data_splits.py

.PHONY: data-validate
data-validate: data-prepare
	$(call log,"Validating data...")
	@$(call use_entry_point,validate) || $(PYTHON) tools/debugging/data_validator.py
	$(call success,"Data validated")

.PHONY: data-stats
data-stats:
	$(call log,"Computing data statistics...")
	@$(PYTHON) -c "from src.data.datasets import ag_news; ag_news.show_statistics()"

.PHONY: train
train: data-prepare
	$(call header,"Training DeBERTa-v3 Model")
	@$(call use_entry_point,train --config configs/models/single/deberta_v3_xlarge.yaml) || \
		$(PYTHON) scripts/training/train_single_model.py --config configs/models/single/deberta_v3_xlarge.yaml
	$(call save_metric,"train")

.PHONY: train-all
train-all: data-prepare
	$(call header,"Training All Models")
	@$(call use_entry_point,train-all) || bash scripts/training/train_all_models.sh
	$(call save_metric,"train-all")

.PHONY: train-ensemble
train-ensemble: train-all
	$(call header,"Training Ensemble")
	@$(call use_entry_point,train-ensemble --config configs/models/ensemble/voting_ensemble.yaml) || \
		$(PYTHON) scripts/training/train_ensemble.py --config configs/models/ensemble/voting_ensemble.yaml
	$(call save_metric,"train-ensemble")

.PHONY: train-distributed
train-distributed: data-prepare
	$(call header,"Distributed Training")
	@torchrun --nproc_per_node=$(GPU_COUNT) scripts/training/distributed_training.py || \
		$(call use_entry_point,train-distributed)
	$(call save_metric,"train-distributed")

.PHONY: train-prompt
train-prompt: data-instruction
	$(call header,"Prompt-based Training")
	@$(call use_entry_point,prompt) || $(PYTHON) scripts/training/train_with_prompts.py
	$(call save_metric,"train-prompt")

.PHONY: train-instruction
train-instruction: data-instruction
	$(call header,"Instruction Tuning")
	@$(call use_entry_point,instruction) || $(PYTHON) scripts/training/instruction_tuning.py
	$(call save_metric,"train-instruction")

.PHONY: train-multistage
train-multistage: data-prepare
	$(call header,"Multi-stage Training")
	@$(call use_entry_point,multistage) || $(PYTHON) scripts/training/multi_stage_training.py
	$(call save_metric,"train-multistage")

.PHONY: train-distill
train-distill: data-prepare
	$(call header,"Knowledge Distillation from GPT-4")
	@$(call use_entry_point,distill) || $(PYTHON) scripts/training/distill_from_gpt4.py
	$(call save_metric,"train-distill")

.PHONY: train-lora
train-lora: data-prepare
	$(call log,"Training with LoRA...")
	@$(PYTHON) src/models/efficient/lora/lora_model.py train --config configs/training/efficient/lora_peft.yaml

.PHONY: train-curriculum
train-curriculum: data-prepare
	$(call log,"Training with curriculum learning...")
	@$(PYTHON) src/training/strategies/curriculum/curriculum_learning.py

.PHONY: train-adversarial
train-adversarial: data-prepare
	$(call log,"Adversarial training...")
	@$(PYTHON) src/training/strategies/adversarial/freelb.py

.PHONY: train-resume
train-resume:
	$(call log,"Resuming training...")
	@$(call use_entry_point,resume --checkpoint $(OUTPUTS_DIR)/models/checkpoints/latest.pt) || \
		$(PYTHON) scripts/training/resume_training.py --checkpoint $(OUTPUTS_DIR)/models/checkpoints/latest.pt

.PHONY: dapt
dapt: dapt-download dapt-pretrain
	$(call save_metric,"dapt")

.PHONY: dapt-download
dapt-download:
	$(call log,"Downloading news corpus...")
	@$(call use_entry_point,download-news) || $(PYTHON) scripts/domain_adaptation/download_news_corpus.py

.PHONY: dapt-pretrain
dapt-pretrain: dapt-download
	$(call log,"Pretraining on news corpus...")
	@$(call use_entry_point,pretrain) || $(PYTHON) scripts/domain_adaptation/pretrain_on_news.py

.PHONY: evaluate
evaluate:
	$(call header,"Model Evaluation")
	@$(call use_entry_point,evaluate --model $(OUTPUTS_DIR)/models/fine_tuned/best_model.pt) || \
		$(PYTHON) scripts/evaluation/evaluate_all_models.py --model $(OUTPUTS_DIR)/models/fine_tuned/best_model.pt
	$(call save_metric,"evaluate")

.PHONY: evaluate-all
evaluate-all:
	$(call log,"Evaluating all models...")
	@$(call use_entry_point,evaluate) || $(PYTHON) scripts/evaluation/evaluate_all_models.py

.PHONY: evaluate-contrast
evaluate-contrast: data-contrast
	$(call log,"Evaluating on contrast sets...")
	@$(call use_entry_point,contrast-eval) || $(PYTHON) scripts/evaluation/evaluate_contrast_sets.py

.PHONY: benchmark
benchmark:
	$(call header,"Running Benchmarks")
	@$(PYTHON) experiments/benchmarks/speed_benchmark.py
	@$(PYTHON) experiments/benchmarks/memory_benchmark.py
	@$(PYTHON) experiments/benchmarks/accuracy_benchmark.py
	$(call save_metric,"benchmark")

.PHONY: analyze
analyze: analyze-errors analyze-attention analyze-interpretability
	$(call save_metric,"analyze")

.PHONY: analyze-errors
analyze-errors:
	$(call log,"Running error analysis...")
	@$(call use_entry_point,error-analysis) || $(PYTHON) src/evaluation/analysis/error_analysis.py

.PHONY: analyze-attention
analyze-attention:
	$(call log,"Analyzing attention patterns...")
	@$(call use_entry_point,attention) || $(PYTHON) src/evaluation/interpretability/attention_analysis.py

.PHONY: analyze-interpretability
analyze-interpretability:
	$(call log,"Running interpretability analysis...")
	@$(call use_entry_point,interpretability) || $(PYTHON) src/evaluation/interpretability/shap_interpreter.py

.PHONY: leaderboard
leaderboard:
	$(call log,"Generating leaderboard...")
	@$(call use_entry_point,leaderboard) || $(PYTHON) scripts/evaluation/create_leaderboard.py

.PHONY: report
report:
	$(call header,"Generating Report")
	@$(call use_entry_point,report) || $(PYTHON) scripts/evaluation/generate_reports.py
	@$(OPEN_CMD) $(OUTPUTS_DIR)/results/report.html

.PHONY: research
research: experiment-baseline experiment-sota experiment-ablation
	$(call save_metric,"research")

.PHONY: experiment
experiment:
	$(call log,"Running experiment...")
	@$(call use_entry_point,experiment --config configs/experiments/sota_attempts/phase5_bleeding_edge.yaml) || \
		$(PYTHON) experiments/experiment_runner.py --config configs/experiments/sota_attempts/phase5_bleeding_edge.yaml

.PHONY: experiment-baseline
experiment-baseline:
	$(call log,"Running baseline experiments...")
	@$(PYTHON) experiments/baselines/classical/svm_baseline.py
	@$(PYTHON) experiments/baselines/neural/bert_vanilla.py

.PHONY: experiment-sota
experiment-sota:
	$(call header,"SOTA Experiments")
	@$(PYTHON) experiments/sota_experiments/single_model_sota.py
	@$(PYTHON) experiments/sota_experiments/ensemble_sota.py
	@$(PYTHON) experiments/sota_experiments/full_pipeline_sota.py
	$(call save_metric,"experiment-sota")

.PHONY: experiment-ablation
experiment-ablation:
	$(call log,"Running ablation studies...")
	@$(call use_entry_point,ablation) || $(PYTHON) experiments/ablation_studies/component_ablation.py

.PHONY: experiment-prompt
experiment-prompt:
	$(call log,"Running prompt experiments...")
	@$(PYTHON) experiments/sota_experiments/prompt_based_sota.py

.PHONY: experiment-distill
experiment-distill:
	$(call log,"Running distillation experiments...")
	@$(PYTHON) experiments/sota_experiments/gpt4_distilled_sota.py

.PHONY: hyperopt
hyperopt:
	$(call header,"Hyperparameter Optimization")
	@$(call use_entry_point,optimize) || $(PYTHON) scripts/optimization/hyperparameter_search.py
	$(call save_metric,"hyperopt")

.PHONY: nas
nas:
	$(call log,"Running NAS...")
	@$(call use_entry_point,nas) || $(PYTHON) scripts/optimization/architecture_search.py

.PHONY: test
test:
	$(call header,"Running Tests")
	@$(PYTEST) tests/ -v --cov=src --cov-report=html
	$(call save_metric,"test")

.PHONY: test-unit
test-unit:
	$(call log,"Running unit tests...")
	@$(PYTEST) tests/unit/ -v

.PHONY: test-integration
test-integration:
	$(call log,"Running integration tests...")
	@$(PYTEST) tests/integration/ -v

.PHONY: test-performance
test-performance:
	$(call log,"Running performance tests...")
	@$(PYTEST) tests/performance/ --benchmark-only

.PHONY: test-setup
test-setup:
	$(call log,"Testing setup commands...")
	@$(PYTHON) setup.py test_setup

.PHONY: coverage
coverage:
	$(call log,"Generating coverage report...")
	@$(COVERAGE) run -m pytest tests/
	@$(COVERAGE) html
	@$(OPEN_CMD) htmlcov/index.html

.PHONY: lint
lint: lint-black lint-isort lint-flake8 lint-mypy
	$(call success,"Linting complete")

.PHONY: lint-black
lint-black:
	$(call log,"Running black...")
	@$(BLACK) $(SRC_DIR) $(TESTS_DIR) --check

.PHONY: lint-isort
lint-isort:
	$(call log,"Running isort...")
	@$(ISORT) $(SRC_DIR) $(TESTS_DIR) --check-only

.PHONY: lint-flake8
lint-flake8:
	$(call log,"Running flake8...")
	@$(FLAKE8) $(SRC_DIR) $(TESTS_DIR)

.PHONY: lint-mypy
lint-mypy:
	$(call log,"Running mypy...")
	@$(MYPY) $(SRC_DIR)

.PHONY: format
format:
	$(call header,"Formatting Code")
	@$(BLACK) $(SRC_DIR) $(TESTS_DIR) scripts/
	@$(ISORT) $(SRC_DIR) $(TESTS_DIR) scripts/
	$(call success,"Code formatted")

.PHONY: security
security:
	$(call header,"Security Scan")
	@$(BANDIT) -r $(SRC_DIR) -f json -o security/scan_results/scan_$(CURRENT_DATE).json
	@$(SAFETY) check --json > security/scan_results/safety_$(CURRENT_DATE).json
	$(call success,"Security scan complete")

.PHONY: profile
profile:
	$(call log,"Profiling code...")
	@$(call use_entry_point,memory-profile) || $(PYTHON) tools/profiling/memory_profiler.py
	@$(call use_entry_point,speed-profile) || $(PYTHON) tools/profiling/speed_profiler.py

.PHONY: optimize
optimize: optimize-quantize optimize-prune optimize-onnx
	$(call save_metric,"optimize")

.PHONY: optimize-quantize
optimize-quantize:
	$(call log,"Quantizing models...")
	@$(PYTHON) src/inference/optimization/quantization_optimizer.py

.PHONY: optimize-prune
optimize-prune:
	$(call log,"Pruning models...")
	@$(PYTHON) src/models/efficient/pruning/magnitude_pruning.py

.PHONY: optimize-onnx
optimize-onnx:
	$(call log,"Converting to ONNX...")
	@$(PYTHON) src/inference/optimization/onnx_converter.py

.PHONY: optimize-tensorrt
optimize-tensorrt:
	$(call log,"Optimizing with TensorRT...")
	@$(PYTHON) src/inference/optimization/tensorrt_optimizer.py

.PHONY: deploy
deploy: deploy-check deploy-build deploy-push deploy-run
	$(call save_metric,"deploy")

.PHONY: deploy-check
deploy-check: test security
	$(call log,"Running pre-deployment checks...")

.PHONY: deploy-build
deploy-build:
	$(call log,"Building deployment artifacts...")
	@$(call use_entry_point,export) || $(PYTHON) scripts/deployment/export_models.py
	@$(PYTHON) scripts/deployment/optimize_for_inference.py

.PHONY: deploy-docker
deploy-docker:
	$(call log,"Deploying with Docker...")
	@$(DOCKER_COMPOSE) up -d

.PHONY: deploy-k8s
deploy-k8s:
	$(call log,"Deploying to Kubernetes...")
	@kubectl apply -f deployment/kubernetes/

.PHONY: deploy-cloud
deploy-cloud:
	$(call log,"Deploying to $(CLOUD_PROVIDER)...")
	@$(call use_entry_point,deploy --provider=$(CLOUD_PROVIDER)) || \
		$(PYTHON) scripts/deployment/deploy_to_cloud.py --provider=$(CLOUD_PROVIDER)

.PHONY: serve
serve:
	$(call header,"Starting Model Server")
	@$(call use_entry_point,serve) || $(PYTHON) src/inference/serving/model_server.py

.PHONY: serve-api
serve-api:
	$(call log,"Starting API server...")
	@$(call use_entry_point,api) || uvicorn src.api.rest.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: serve-grpc
serve-grpc:
	$(call log,"Starting gRPC server...")
	@$(call use_entry_point,grpc) || $(PYTHON) src/api/grpc/services.py

.PHONY: serve-streamlit
serve-streamlit:
	$(call log,"Starting Streamlit app...")
	@$(call use_entry_point,streamlit) || streamlit run app/streamlit_app.py

.PHONY: docker-build
docker-build:
	$(call log,"Building Docker image...")
	@$(DOCKER) build -t $(DOCKER_IMAGE) -f deployment/docker/Dockerfile .

.PHONY: docker-build-gpu
docker-build-gpu:
	$(call log,"Building GPU Docker image...")
	@$(DOCKER) build -t $(DOCKER_IMAGE)-gpu -f deployment/docker/Dockerfile.gpu .

.PHONY: docker-run
docker-run:
	$(call log,"Running Docker container...")
	@$(DOCKER) run -it --rm -p 8000:8000 $(DOCKER_IMAGE)

.PHONY: docker-run-gpu
docker-run-gpu:
	$(call log,"Running GPU Docker container...")
	@$(DOCKER) run -it --rm --gpus all -p 8000:8000 $(DOCKER_IMAGE)-gpu

.PHONY: docker-push
docker-push:
	$(call log,"Pushing Docker image...")
	@$(DOCKER) tag $(DOCKER_IMAGE) $(DOCKER_TAG)
	@$(DOCKER) push $(DOCKER_TAG)

.PHONY: docker-compose-up
docker-compose-up:
	$(call log,"Starting services...")
	@$(DOCKER_COMPOSE) -f deployment/docker/docker-compose.yml up -d

.PHONY: docker-compose-down
docker-compose-down:
	$(call log,"Stopping services...")
	@$(DOCKER_COMPOSE) -f deployment/docker/docker-compose.yml down

.PHONY: docs
docs:
	$(call header,"Generating Documentation")
	@$(PYTHON) setup.py docs --format=html

.PHONY: docs-serve
docs-serve:
	$(call log,"Serving documentation...")
	@$(PYTHON) setup.py docs --format=html --serve --port=8080

.PHONY: docs-pdf
docs-pdf:
	$(call log,"Generating PDF documentation...")
	@$(PYTHON) setup.py docs --format=pdf

.PHONY: notebook
notebook:
	$(call log,"Starting Jupyter notebook...")
	@jupyter notebook --notebook-dir=$(NOTEBOOKS_DIR)

.PHONY: lab
lab:
	$(call log,"Starting JupyterLab...")
	@jupyter lab --notebook-dir=$(NOTEBOOKS_DIR)

.PHONY: notebook-run
notebook-run:
	$(call log,"Running notebooks...")
	@jupyter nbconvert --to notebook --execute notebooks/**/*.ipynb

.PHONY: notebook-clean
notebook-clean:
	$(call log,"Cleaning notebook outputs...")
	@jupyter nbconvert --clear-output notebooks/**/*.ipynb

.PHONY: monitor
monitor: monitor-metrics monitor-logs monitor-dashboard
	$(call save_metric,"monitor")

.PHONY: monitor-metrics
monitor-metrics:
	$(call log,"Starting metrics collection...")
	@$(call use_entry_point,metrics) || $(PYTHON) monitoring/metrics/metric_collectors.py

.PHONY: monitor-logs
monitor-logs:
	$(call log,"Monitoring logs...")
	@tail -f $(LOG_DIR)/*.log

.PHONY: monitor-dashboard
monitor-dashboard:
	$(call log,"Opening monitoring dashboard...")
	@$(OPEN_CMD) http://localhost:3000

.PHONY: monitor-gpu
monitor-gpu:
	$(call log,"Monitoring GPU...")
	@watch -n 1 nvidia-smi

.PHONY: db-migrate
db-migrate:
	$(call log,"Running migrations...")
	@$(call use_entry_point,db-migrate) || $(PYTHON) migrations/data/migration_runner.py

.PHONY: db-seed
db-seed:
	$(call log,"Seeding database...")
	@$(call use_entry_point,db-seed) || $(PYTHON) scripts/seed_database.py

.PHONY: db-backup
db-backup:
	$(call log,"Backing up database...")
	@$(call use_entry_point,db-backup) || $(PYTHON) scripts/backup_database.py

.PHONY: db-restore
db-restore:
	$(call log,"Restoring database...")
	@$(PYTHON) scripts/restore_database.py --backup=$(BACKUP_FILE)

.PHONY: release
release:
	$(call header,"Creating Release")
	@$(PYTHON) setup.py release --version=$(VERSION) --push --build

.PHONY: changelog
changelog:
	$(call log,"Updating changelog...")
	@conventional-changelog -p angular -i CHANGELOG.md -s 2>/dev/null || $(call warning,"conventional-changelog not installed")

.PHONY: version
version:
	@echo "$(PROJECT_NAME) v$(VERSION)"

.PHONY: clean
clean:
	$(call header,"Cleaning")
	@rm -rf $(BUILD_DIR) $(DIST_DIR) *.egg-info
	@rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	$(call success,"Cleaned")

.PHONY: clean-data
clean-data:
	$(call log,"Cleaning data directories...")
	@rm -rf $(DATA_DIR)/processed/*
	@rm -rf $(DATA_DIR)/augmented/*
	@rm -rf $(DATA_DIR)/cache/*

.PHONY: clean-outputs
clean-outputs:
	$(call log,"Cleaning output directories...")
	@rm -rf $(OUTPUTS_DIR)/models/*
	@rm -rf $(OUTPUTS_DIR)/results/*
	@rm -rf $(OUTPUTS_DIR)/logs/*

.PHONY: clean-all
clean-all: clean clean-data clean-outputs
	$(call success,"All cleaned")

.PHONY: download-resources
download-resources:
	$(call log,"Downloading resources...")
	@$(call use_entry_point,download) || $(PYTHON) scripts/setup/download_all_data.py

.PHONY: info
info:
	@echo -e "\n$(MAGENTA)Project Information$(RESET)"
	@echo -e "$(CYAN)Name:$(RESET) $(PROJECT_NAME)"
	@echo -e "$(CYAN)Version:$(RESET) $(VERSION)"
	@echo -e "$(CYAN)Python:$(RESET) $(PYTHON_VERSION)"
	@echo -e "$(CYAN)Directory:$(RESET) $(ROOT_DIR)"
	@echo -e "$(CYAN)OS:$(RESET) $(OS)"
	@echo -e "$(CYAN)GPU Count:$(RESET) $(GPU_COUNT)"
	@echo -e "$(CYAN)CUDA Version:$(RESET) $(CUDA_VERSION)"
	@echo -e "$(CYAN)In Container:$(RESET) $(IN_CONTAINER)"
	@echo -e "$(CYAN)CI Environment:$(RESET) $(CI)"
	@$(PYTHON) setup.py health_check 2>/dev/null || true

.PHONY: check-deps
check-deps:
	$(call log,"Checking dependencies...")
	@$(PIP) check
	@$(SAFETY) check

.PHONY: update-deps
update-deps:
	$(call log,"Updating dependencies...")
	@$(PIP) list --outdated
	@$(PIP) install --upgrade pip setuptools wheel

.PHONY: backup
backup:
	$(call log,"Creating backup...")
	@mkdir -p $(BACKUP_DIR)
	@tar -czf $(BACKUP_DIR)/backup_$(PROJECT_NAME)_$(CURRENT_DATE).tar.gz \
		--exclude=$(VENV_NAME) \
		--exclude=$(DATA_DIR)/raw \
		--exclude=$(OUTPUTS_DIR) \
		--exclude=.git \
		.
	$(call success,"Backup created: $(BACKUP_DIR)/backup_$(PROJECT_NAME)_$(CURRENT_DATE).tar.gz")

.PHONY: restore
restore:
	$(call log,"Restoring from backup...")
	@test -f $(BACKUP_FILE) || { $(call error,"Backup file not found: $(BACKUP_FILE)"); exit 1; }
	@tar -xzf $(BACKUP_FILE) -C .
	$(call success,"Restored from $(BACKUP_FILE)")

.PHONY: shell
shell:
	@$(PYTHON) -i -c "from src import *; print('AG News Classification Shell')"

.PHONY: repl
repl: shell

.PHONY: ci
ci: validate lint test security build
	$(call save_metric,"ci")

.PHONY: cd
cd: ci deploy
	$(call save_metric,"cd")

.PHONY: build
build:
	$(call log,"Building distribution...")
	@$(PYTHON) setup.py sdist bdist_wheel

.PHONY: publish
publish:
	$(call log,"Publishing to PyPI...")
	@twine upload dist/*

.PHONY: publish-test
publish-test:
	$(call log,"Publishing to TestPyPI...")
	@twine upload --repository testpypi dist/*

.PHONY: research-full
research-full:
	$(call header,"10-Week Research Pipeline")
	@echo "Week 1: Classical Baselines"
	@$(MAKE) install-week1
	@$(MAKE) experiment-baseline
	@echo "Week 2-3: Deep Learning & Transformers"
	@$(MAKE) install-week2-3
	@$(MAKE) train-all
	@echo "Week 4-5: Advanced Training"
	@$(MAKE) install-week4-5
	@$(MAKE) train-ensemble
	@echo "Week 6-7: SOTA & LLMs"
	@$(MAKE) install-week6-7
	@$(MAKE) train-distill
	@echo "Week 8-9: Optimization"
	@$(MAKE) install-week8-9
	@$(MAKE) optimize
	@echo "Week 10: Deployment"
	@$(MAKE) install-week10
	@$(MAKE) deploy
	$(call save_metric,"research-full")

.PHONY: research-reproduce
research-reproduce:
	$(call header,"Reproducing Paper Results")
	@$(PYTHON) experiments/sota_experiments/full_pipeline_sota.py --reproduce

.PHONY: research-benchmark
research-benchmark:
	$(call header,"Research Benchmarks")
	@$(PYTHON) experiments/benchmarks/accuracy_benchmark.py --full
	@$(PYTHON) experiments/benchmarks/robustness_benchmark.py --full

.PHONY: quickstart
quickstart:
	$(call header,"Quick Start")
	@$(call use_entry_point,quick-start) || $(PYTHON) quickstart/minimal_example.py

.PHONY: demo
demo:
	$(call header,"Interactive Demo")
	@$(call use_entry_point,demo) || $(PYTHON) quickstart/demo_app.py

.PHONY: colab
colab:
	$(call log,"Generating Colab notebook...")
	@jupyter nbconvert --to notebook quickstart/colab_notebook.ipynb

.PHONY: dev
dev: setup install-dev setup-git
	$(call header,"Development Mode")
	$(call success,"Development environment ready!")

.PHONY: prod
prod: setup-production optimize deploy-check
	$(call header,"Production Mode")
	$(call success,"Production environment ready!")

.PHONY: gpu-check
gpu-check:
	@$(PYTHON) -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

.PHONY: memory-check
memory-check:
	@free -h 2>/dev/null || vm_stat 2>/dev/null || $(call warning,"Memory check not available")
	@df -h

.PHONY: health
health:
	$(call header,"Health Check")
	@$(PYTHON) setup.py health_check

.PHONY: metrics
metrics:
	$(call log,"Showing metrics...")
	@test -f $(METRICS_FILE) && cat $(METRICS_FILE) | jq '.' || $(call warning,"No metrics available")

.PHONY: cache-clear
cache-clear:
	$(call log,"Clearing cache...")
	@rm -rf $(CACHE_DIR)/*
	@rm -f .make.cache
	$(call success,"Cache cleared")

.PHONY: sync-config
sync-config:
	$(call log,"Syncing configuration with setup.py...")
	@$(PYTHON) -c "from setup import get_shared_config; get_shared_config()"
	$(call success,"Configuration synced")

guard-%:
	@if [ -z '${${*}}' ]; then \
		echo "Variable $* is not set"; \
		exit 1; \
	fi

.PRECIOUS: %.pt %.onnx %.tflite

%:
	@echo "Unknown target: $@"
	@echo "Run 'make help' to see available targets"
