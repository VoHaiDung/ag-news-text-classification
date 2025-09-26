#!/bin/bash

# AG News Text Classification - Train All Models Script
# ======================================================
# This script trains all model configurations for comprehensive comparison
# Following best practices from MLOps and reproducible research
#
# Author: Võ Hải Dũng
# License: MIT

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="${PROJECT_ROOT}/outputs/models"
LOGS_DIR="${PROJECT_ROOT}/outputs/logs"
CONFIGS_DIR="${PROJECT_ROOT}/configs"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts/training"

# Create necessary directories
mkdir -p "${MODELS_DIR}"
mkdir -p "${LOGS_DIR}"

# Default parameters
DEVICE="cuda"
SEED=42
NUM_EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=2e-5
DATA_DIR="${PROJECT_ROOT}/data/processed"

# Logging function
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check GPU availability
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found. Using CPU instead."
        DEVICE="cpu"
    else
        if ! nvidia-smi &> /dev/null; then
            log_warning "No GPU detected. Using CPU instead."
            DEVICE="cpu"
        else
            log_info "GPU detected. Using CUDA for training."
            nvidia-smi
        fi
    fi
}

# Function to train a single model
train_single_model() {
    local model_name=$1
    local config_file=$2
    local experiment_name=$3
    
    log_info "Training ${model_name}..."
    
    python "${SCRIPTS_DIR}/train_single_model.py" \
        --model-name "${model_name}" \
        --model-config "${config_file}" \
        --num-epochs ${NUM_EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --learning-rate ${LEARNING_RATE} \
        --data-dir "${DATA_DIR}" \
        --experiment-name "${experiment_name}" \
        --seed ${SEED} \
        --device ${DEVICE} \
        --output-dir "${MODELS_DIR}" \
        2>&1 | tee "${LOGS_DIR}/${experiment_name}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "Successfully trained ${model_name}"
    else
        log_error "Failed to train ${model_name}"
        return 1
    fi
}

# Function to train ensemble models
train_ensemble() {
    local ensemble_type=$1
    local experiment_name=$2
    
    log_info "Training ${ensemble_type} ensemble..."
    
    python "${SCRIPTS_DIR}/train_ensemble.py" \
        --ensemble-type "${ensemble_type}" \
        --num-epochs ${NUM_EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --experiment-name "${experiment_name}" \
        --seed ${SEED} \
        --device ${DEVICE} \
        2>&1 | tee "${LOGS_DIR}/${experiment_name}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_info "Successfully trained ${ensemble_type} ensemble"
    else
        log_error "Failed to train ${ensemble_type} ensemble"
        return 1
    fi
}

# Function to run distributed training
train_distributed() {
    local model_name=$1
    local experiment_name=$2
    
    log_info "Running distributed training for ${model_name}..."
    
    # Check if multiple GPUs available
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    
    if [ ${GPU_COUNT} -gt 1 ]; then
        log_info "Found ${GPU_COUNT} GPUs. Running distributed training..."
        
        python -m torch.distributed.launch \
            --nproc_per_node=${GPU_COUNT} \
            "${SCRIPTS_DIR}/distributed_training.py" \
            --model-name "${model_name}" \
            --num-epochs ${NUM_EPOCHS} \
            --batch-size ${BATCH_SIZE} \
            --learning-rate ${LEARNING_RATE} \
            --seed ${SEED} \
            --output-dir "${MODELS_DIR}/${experiment_name}" \
            2>&1 | tee "${LOGS_DIR}/${experiment_name}.log"
    else
        log_warning "Only 1 GPU found. Running single GPU training..."
        train_single_model "${model_name}" "" "${experiment_name}"
    fi
}

# Main training pipeline
main() {
    log_info "Starting AG News Text Classification Training Pipeline"
    log_info "Project Root: ${PROJECT_ROOT}"
    
    # Check GPU availability
    check_gpu
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --epochs)
                NUM_EPOCHS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --learning-rate)
                LEARNING_RATE="$2"
                shift 2
                ;;
            --seed)
                SEED="$2"
                shift 2
                ;;
            --models)
                SELECTED_MODELS="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Training timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    
    # Track results
    RESULTS_FILE="${LOGS_DIR}/training_results_${TIMESTAMP}.json"
    echo "{" > "${RESULTS_FILE}"
    echo "  \"timestamp\": \"${TIMESTAMP}\"," >> "${RESULTS_FILE}"
    echo "  \"device\": \"${DEVICE}\"," >> "${RESULTS_FILE}"
    echo "  \"seed\": ${SEED}," >> "${RESULTS_FILE}"
    echo "  \"models\": [" >> "${RESULTS_FILE}"
    
    # Train individual models
    log_info "=== Phase 1: Training Individual Models ==="
    
    # BERT variants
    if [[ -z ${SELECTED_MODELS+x} ]] || [[ ${SELECTED_MODELS} == *"bert"* ]]; then
        train_single_model \
            "bert-base-uncased" \
            "${CONFIGS_DIR}/models/single/bert_base.yaml" \
            "bert_base_${TIMESTAMP}"
    fi
    
    # RoBERTa variants
    if [[ -z ${SELECTED_MODELS+x} ]] || [[ ${SELECTED_MODELS} == *"roberta"* ]]; then
        train_single_model \
            "roberta-base" \
            "${CONFIGS_DIR}/models/single/roberta_large.yaml" \
            "roberta_base_${TIMESTAMP}"
    fi
    
    # DeBERTa variants
    if [[ -z ${SELECTED_MODELS+x} ]] || [[ ${SELECTED_MODELS} == *"deberta"* ]]; then
        train_single_model \
            "microsoft/deberta-v3-base" \
            "${CONFIGS_DIR}/models/single/deberta_v3_xlarge.yaml" \
            "deberta_base_${TIMESTAMP}"
    fi
    
    # DistilBERT (faster, smaller)
    if [[ -z ${SELECTED_MODELS+x} ]] || [[ ${SELECTED_MODELS} == *"distilbert"* ]]; then
        train_single_model \
            "distilbert-base-uncased" \
            "" \
            "distilbert_base_${TIMESTAMP}"
    fi
    
    # ELECTRA
    if [[ -z ${SELECTED_MODELS+x} ]] || [[ ${SELECTED_MODELS} == *"electra"* ]]; then
        train_single_model \
            "google/electra-base-discriminator" \
            "${CONFIGS_DIR}/models/single/electra_large.yaml" \
            "electra_base_${TIMESTAMP}"
    fi
    
    log_info "=== Phase 2: Training Ensemble Models ==="
    
    # Voting ensemble
    if [[ -z ${SELECTED_MODELS+x} ]] || [[ ${SELECTED_MODELS} == *"ensemble"* ]]; then
        train_ensemble "voting" "voting_ensemble_${TIMESTAMP}"
        
        # Stacking ensemble
        train_ensemble "stacking" "stacking_ensemble_${TIMESTAMP}"
        
        # Blending ensemble
        train_ensemble "blending" "blending_ensemble_${TIMESTAMP}"
    fi
    
    log_info "=== Phase 3: Advanced Training Strategies ==="
    
    # Distributed training (if multiple GPUs)
    if [[ ${DEVICE} == "cuda" ]]; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        if [ ${GPU_COUNT} -gt 1 ]; then
            train_distributed "bert-base-uncased" "distributed_bert_${TIMESTAMP}"
        fi
    fi
    
    # Prompt-based training
    if [[ -z ${SELECTED_MODELS+x} ]] || [[ ${SELECTED_MODELS} == *"prompt"* ]]; then
        log_info "Training with prompts..."
        python "${SCRIPTS_DIR}/train_with_prompts.py" \
            --model-name "roberta-large" \
            --prompt-type "manual" \
            --num-epochs ${NUM_EPOCHS} \
            --batch-size ${BATCH_SIZE} \
            --device ${DEVICE} \
            2>&1 | tee "${LOGS_DIR}/prompt_training_${TIMESTAMP}.log"
    fi
    
    # Multi-stage training
    if [[ -z ${SELECTED_MODELS+x} ]] || [[ ${SELECTED_MODELS} == *"multistage"* ]]; then
        log_info "Running multi-stage training..."
        python "${SCRIPTS_DIR}/multi_stage_training.py" \
            --model-name "bert-base-uncased" \
            --experiment-name "multistage_${TIMESTAMP}" \
            --device ${DEVICE} \
            2>&1 | tee "${LOGS_DIR}/multistage_${TIMESTAMP}.log"
    fi
    
    # Close results JSON
    echo "  ]" >> "${RESULTS_FILE}"
    echo "}" >> "${RESULTS_FILE}"
    
    log_info "=== Training Pipeline Completed ==="
    log_info "Results saved to: ${RESULTS_FILE}"
    
    # Generate summary report
    python -c "
import json
from pathlib import Path

results_file = '${RESULTS_FILE}'
logs_dir = Path('${LOGS_DIR}')

# Parse logs and generate summary
print('\n=== Training Summary ===')
print(f'Timestamp: ${TIMESTAMP}')
print(f'Device: ${DEVICE}')
print(f'Models trained:')

# List all completed models
for log_file in logs_dir.glob('*_${TIMESTAMP}.log'):
    model_name = log_file.stem.replace('_${TIMESTAMP}', '')
    print(f'  - {model_name}')
"
}

# Execute main function
main "$@"
