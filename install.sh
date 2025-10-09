#!/usr/bin/env bash

# ============================================================================
# Installation Script for AG News Text Classification
# ============================================================================
# Project: AG News Text Classification (ag-news-text-classification)
# Description: Automated setup and installation script for all platforms
# Author: Võ Hải Dũng
# Email: vohaidung.work@gmail.com
# License: MIT
# ============================================================================
#
# Academic Rationale:
#   Automated installation scripts are essential for reproducible research,
#   as documented in "Research Software Engineering with Python" (Perkel, 2020).
#   This script follows the principle of "one-command setup" from DevOps best
#   practices, ensuring consistent environments across different platforms and
#   reducing setup friction for collaborators.
#
# Design Principles:
#   1. Platform Detection: Automatically detect local, Colab, Kaggle
#   2. Idempotency: Safe to run multiple times without side effects
#   3. Error Handling: Fail fast with clear error messages
#   4. Logging: Comprehensive logging for debugging
#   5. Modularity: Functions for each installation step
#   6. Flexibility: Support multiple installation modes
#
# Usage:
#   Basic installation:           ./install.sh
#   Skip dependency installation: ./install.sh --skip-deps
#   Development mode:             ./install.sh --dev
#   Minimal installation:         ./install.sh --minimal
#   Clean installation:           ./install.sh --clean
#   Verbose output:               ./install.sh --verbose
#   Help:                         ./install.sh --help
#
# Supported Platforms:
#   - Linux (Ubuntu 18.04+, Debian, CentOS, RHEL)
#   - macOS (10.14+)
#   - Windows (via WSL2)
#   - Google Colab
#   - Kaggle Notebooks
#
# Requirements:
#   - Bash 4.0+
#   - Python 3.8+
#   - Git 2.0+
#   - Internet connection for package downloads
#
# Exit Codes:
#   0: Success
#   1: General error
#   2: Platform not supported
#   3: Python version incompatible
#   4: Dependency installation failed
#   5: Directory creation failed
#   6: Configuration failed
#
# References:
#   - Google Shell Style Guide: https://google.github.io/styleguide/shellguide.html
#   - Advanced Bash-Scripting Guide: https://tldp.org/LDP/abs/html/
#
# ============================================================================

set -e  # Exit immediately if a command exits with a non-zero status
set -u  # Treat unset variables as an error
set -o pipefail  # Pipeline fails if any command fails

# ============================================================================
# Global Variables and Configuration
# ============================================================================

readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="${SCRIPT_DIR}"
readonly PROJECT_NAME="AG News Text Classification"
readonly PROJECT_SLUG="ag-news-text-classification"
readonly PROJECT_VERSION="1.0.0"

# Color codes for terminal output
readonly COLOR_RESET='\033[0m'
readonly COLOR_RED='\033[0;31m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[0;33m'
readonly COLOR_BLUE='\033[0;34m'
readonly COLOR_CYAN='\033[0;36m'

# Installation modes
INSTALL_MODE="full"
SKIP_DEPS=false
DEV_MODE=false
MINIMAL_MODE=false
CLEAN_INSTALL=false
VERBOSE=false

# Platform detection
PLATFORM="unknown"
OS_TYPE="unknown"
PYTHON_CMD="python3"

# Minimum requirements
readonly MIN_PYTHON_VERSION="3.8"
readonly MIN_GIT_VERSION="2.0"

# ============================================================================
# Utility Functions
# ============================================================================

# Print colored message
# Arguments:
#   $1: Color code
#   $2: Message
print_color() {
    local color="$1"
    local message="$2"
    echo -e "${color}${message}${COLOR_RESET}"
}

# Log informational message
log_info() {
    print_color "${COLOR_BLUE}" "[INFO] $*"
}

# Log success message
log_success() {
    print_color "${COLOR_GREEN}" "[SUCCESS] $*"
}

# Log warning message
log_warning() {
    print_color "${COLOR_YELLOW}" "[WARNING] $*"
}

# Log error message and exit
log_error() {
    print_color "${COLOR_RED}" "[ERROR] $*" >&2
    exit 1
}

# Log verbose message (only if verbose mode enabled)
log_verbose() {
    if [[ "${VERBOSE}" == "true" ]]; then
        print_color "${COLOR_CYAN}" "[VERBOSE] $*"
    fi
}

# Print section header
print_section() {
    local section_name="$1"
    echo ""
    print_color "${COLOR_CYAN}" "============================================================"
    print_color "${COLOR_CYAN}" "  ${section_name}"
    print_color "${COLOR_CYAN}" "============================================================"
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Compare version numbers
# Returns 0 if $1 >= $2, 1 otherwise
version_gte() {
    printf '%s\n%s' "$2" "$1" | sort -V -C
}

# ============================================================================
# Platform Detection
# ============================================================================

detect_platform() {
    print_section "Platform Detection"
    
    # Detect operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS_TYPE="linux"
        log_info "Operating System: Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        log_info "Operating System: macOS"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS_TYPE="windows"
        log_info "Operating System: Windows (via WSL/Cygwin)"
    else
        log_warning "Unknown operating system: $OSTYPE"
        OS_TYPE="unknown"
    fi
    
    # Detect specific platform
    if [[ -n "${COLAB_GPU:-}" ]] || [[ -d "/content" ]]; then
        PLATFORM="colab"
        log_info "Platform: Google Colab"
    elif [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]] || [[ -d "/kaggle" ]]; then
        PLATFORM="kaggle"
        log_info "Platform: Kaggle"
    else
        PLATFORM="local"
        log_info "Platform: Local"
    fi
    
    log_success "Platform detection completed"
}

# ============================================================================
# Requirements Checking
# ============================================================================

check_python() {
    log_info "Checking Python installation..."
    
    # Try different Python commands
    local python_commands=("python3" "python" "python3.10" "python3.9" "python3.8")
    local python_found=false
    
    for cmd in "${python_commands[@]}"; do
        if command_exists "$cmd"; then
            local version
            version=$($cmd --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
            
            if version_gte "$version" "$MIN_PYTHON_VERSION"; then
                PYTHON_CMD="$cmd"
                python_found=true
                log_success "Python $version found: $cmd"
                break
            else
                log_verbose "Python $version found but too old (requires $MIN_PYTHON_VERSION+)"
            fi
        fi
    done
    
    if [[ "$python_found" == "false" ]]; then
        log_error "Python $MIN_PYTHON_VERSION or higher is required. Please install Python first."
        exit 3
    fi
}

check_git() {
    log_info "Checking Git installation..."
    
    if ! command_exists git; then
        log_error "Git is required but not installed. Please install Git first."
        exit 1
    fi
    
    local git_version
    git_version=$(git --version | grep -oP '\d+\.\d+' | head -1)
    
    if version_gte "$git_version" "$MIN_GIT_VERSION"; then
        log_success "Git $git_version found"
    else
        log_warning "Git version $git_version is old. Recommended: $MIN_GIT_VERSION+"
    fi
}

check_disk_space() {
    log_info "Checking disk space..."
    
    local required_space_gb=10
    local available_space
    
    if [[ "$OS_TYPE" == "macos" ]]; then
        available_space=$(df -g . | tail -1 | awk '{print $4}')
    else
        available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    fi
    
    if [[ "$available_space" -lt "$required_space_gb" ]]; then
        log_warning "Low disk space: ${available_space}GB available (recommended: ${required_space_gb}GB+)"
    else
        log_success "Sufficient disk space: ${available_space}GB available"
    fi
}

check_requirements() {
    print_section "System Requirements Check"
    
    check_python
    check_git
    check_disk_space
    
    log_success "System requirements check completed"
}

# ============================================================================
# Directory Structure Creation
# ============================================================================

create_directories() {
    print_section "Creating Directory Structure"
    
    log_info "Creating project directories..."
    
    # Define all directories from project structure
    local directories=(
        # Data directories
        "data/raw/ag_news"
        "data/processed/train"
        "data/processed/validation"
        "data/processed/test"
        "data/processed/stratified_folds"
        "data/processed/instruction_formatted"
        "data/augmented/back_translated"
        "data/augmented/paraphrased"
        "data/augmented/synthetic"
        "data/augmented/llm_generated/llama2"
        "data/augmented/llm_generated/mistral"
        "data/augmented/llm_generated/mixtral"
        "data/augmented/mixup"
        "data/augmented/contrast_sets"
        "data/external/news_corpus"
        "data/external/pretrain_data"
        "data/external/distillation_data/llama_outputs"
        "data/external/distillation_data/mistral_outputs"
        "data/external/distillation_data/teacher_ensemble_outputs"
        "data/pseudo_labeled"
        "data/selected_subsets"
        "data/test_samples"
        "data/metadata/model_predictions"
        "data/cache/local_cache"
        "data/cache/model_cache"
        "data/cache/huggingface_cache"
        "data/platform_cache/colab_cache"
        "data/platform_cache/kaggle_cache"
        "data/platform_cache/local_cache"
        "data/quota_tracking"
        
        # Output directories
        "outputs/models/checkpoints"
        "outputs/models/pretrained"
        "outputs/models/fine_tuned"
        "outputs/models/lora_adapters"
        "outputs/models/qlora_adapters"
        "outputs/models/ensembles"
        "outputs/models/distilled"
        "outputs/models/optimized"
        "outputs/models/exported"
        "outputs/models/prompted"
        "outputs/results/experiments"
        "outputs/results/benchmarks"
        "outputs/results/overfitting_reports"
        "outputs/results/parameter_efficiency_reports"
        "outputs/results/ablations"
        "outputs/results/reports"
        "outputs/analysis/error_analysis"
        "outputs/analysis/interpretability"
        "outputs/analysis/statistical"
        "outputs/logs/training"
        "outputs/logs/tensorboard"
        "outputs/logs/mlflow"
        "outputs/logs/wandb"
        "outputs/logs/local"
        "outputs/profiling/memory"
        "outputs/profiling/speed"
        "outputs/profiling/traces"
        "outputs/artifacts/figures"
        "outputs/artifacts/tables"
        "outputs/artifacts/lora_visualizations"
        "outputs/artifacts/presentations"
        
        # Experiment directories
        "experiments/results"
        
        # Cache directories
        ".cache/huggingface"
        ".cache/transformers"
        ".cache/datasets"
        
        # Temporary directories
        "tmp"
        "temp"
        
        # Monitoring directories
        "monitoring/local/data"
        
        # Backup directories
        "backup/strategies"
    )
    
    local created_count=0
    local skipped_count=0
    
    for dir in "${directories[@]}"; do
        if [[ -d "$dir" ]]; then
            log_verbose "Directory already exists: $dir"
            ((skipped_count++))
        else
            if mkdir -p "$dir"; then
                log_verbose "Created directory: $dir"
                ((created_count++))
            else
                log_error "Failed to create directory: $dir"
                exit 5
            fi
        fi
    done
    
    log_info "Created $created_count new directories, $skipped_count already existed"
    log_success "Directory structure creation completed"
}

# ============================================================================
# Python Virtual Environment Setup
# ============================================================================

setup_virtualenv() {
    print_section "Python Virtual Environment Setup"
    
    local venv_dir="venv"
    
    if [[ "${SKIP_DEPS}" == "true" ]]; then
        log_info "Skipping virtual environment setup (--skip-deps flag)"
        return 0
    fi
    
    if [[ -d "$venv_dir" ]]; then
        if [[ "${CLEAN_INSTALL}" == "true" ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf "$venv_dir"
        else
            log_info "Virtual environment already exists: $venv_dir"
            return 0
        fi
    fi
    
    log_info "Creating virtual environment..."
    
    if ! $PYTHON_CMD -m venv "$venv_dir"; then
        log_error "Failed to create virtual environment"
        exit 4
    fi
    
    log_success "Virtual environment created: $venv_dir"
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    
    if [[ -f "$venv_dir/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$venv_dir/bin/activate"
        log_success "Virtual environment activated"
    else
        log_error "Failed to find activation script"
        exit 4
    fi
}

# ============================================================================
# Dependency Installation
# ============================================================================

install_dependencies() {
    print_section "Installing Dependencies"
    
    if [[ "${SKIP_DEPS}" == "true" ]]; then
        log_info "Skipping dependency installation (--skip-deps flag)"
        return 0
    fi
    
    log_info "Upgrading pip, setuptools, and wheel..."
    
    if ! $PYTHON_CMD -m pip install --upgrade pip setuptools wheel; then
        log_error "Failed to upgrade pip, setuptools, and wheel"
        exit 4
    fi
    
    # Determine which requirements to install based on mode
    local requirements_files=()
    
    if [[ "${MINIMAL_MODE}" == "true" ]]; then
        log_info "Installing minimal dependencies..."
        requirements_files=("requirements/minimal.txt")
    elif [[ "${DEV_MODE}" == "true" ]]; then
        log_info "Installing development dependencies..."
        requirements_files=(
            "requirements/base.txt"
            "requirements/ml.txt"
            "requirements/dev.txt"
            "requirements/data.txt"
            "requirements/ui.txt"
        )
    else
        log_info "Installing full dependencies..."
        
        # Platform-specific requirements
        if [[ "$PLATFORM" == "colab" ]]; then
            requirements_files=("requirements/colab.txt")
        elif [[ "$PLATFORM" == "kaggle" ]]; then
            requirements_files=("requirements/kaggle.txt")
        else
            requirements_files=(
                "requirements/base.txt"
                "requirements/ml.txt"
                "requirements/data.txt"
                "requirements/ui.txt"
            )
        fi
    fi
    
    # Install each requirements file
    for req_file in "${requirements_files[@]}"; do
        if [[ -f "$req_file" ]]; then
            log_info "Installing from $req_file..."
            
            if $PYTHON_CMD -m pip install -r "$req_file"; then
                log_success "Installed dependencies from $req_file"
            else
                log_error "Failed to install dependencies from $req_file"
                exit 4
            fi
        else
            log_warning "Requirements file not found: $req_file"
        fi
    done
    
    # Install package in development mode if in dev mode
    if [[ "${DEV_MODE}" == "true" ]] && [[ -f "setup.py" ]]; then
        log_info "Installing package in development mode..."
        
        if $PYTHON_CMD -m pip install -e .; then
            log_success "Package installed in development mode"
        else
            log_warning "Failed to install package in development mode"
        fi
    fi
    
    log_success "Dependency installation completed"
}

# ============================================================================
# Configuration Setup
# ============================================================================

setup_configuration() {
    print_section "Configuration Setup"
    
    log_info "Setting up configuration files..."
    
    # Create .env file from .env.example if it doesn't exist
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            log_info "Creating .env from .env.example..."
            cp .env.example .env
            log_success "Created .env file"
            log_warning "Please edit .env file with your specific configuration"
        else
            log_warning ".env.example not found, skipping .env creation"
        fi
    else
        log_info ".env file already exists"
    fi
    
    # Create .env.local if on local platform
    if [[ "$PLATFORM" == "local" ]] && [[ ! -f ".env.local" ]]; then
        if [[ -f ".env.local.example" ]]; then
            log_info "Creating .env.local for local development..."
            cp .env.local.example .env.local
            log_success "Created .env.local file"
        fi
    fi
    
    log_success "Configuration setup completed"
}

# ============================================================================
# Git Hooks Setup
# ============================================================================

setup_git_hooks() {
    print_section "Git Hooks Setup"
    
    if [[ "${DEV_MODE}" != "true" ]]; then
        log_info "Skipping git hooks setup (not in dev mode)"
        return 0
    fi
    
    log_info "Setting up pre-commit hooks..."
    
    if command_exists pre-commit; then
        if pre-commit install; then
            log_success "Pre-commit hooks installed"
        else
            log_warning "Failed to install pre-commit hooks"
        fi
        
        if pre-commit install --hook-type commit-msg; then
            log_success "Commit message hooks installed"
        else
            log_warning "Failed to install commit message hooks"
        fi
    else
        log_warning "pre-commit not found, skipping hooks setup"
        log_info "Install with: pip install pre-commit"
    fi
}

# ============================================================================
# Data Download
# ============================================================================

download_data() {
    print_section "Data Download"
    
    if [[ "${MINIMAL_MODE}" == "true" ]]; then
        log_info "Skipping data download (minimal mode)"
        return 0
    fi
    
    log_info "Downloading sample data..."
    
    # Check if download script exists
    if [[ -f "scripts/setup/download_all_data.py" ]]; then
        log_info "Running data download script..."
        
        if $PYTHON_CMD scripts/setup/download_all_data.py --sample; then
            log_success "Sample data downloaded"
        else
            log_warning "Data download script failed (non-critical)"
        fi
    else
        log_info "Data download script not found, skipping"
    fi
}

# ============================================================================
# Verification
# ============================================================================

verify_installation() {
    print_section "Installation Verification"
    
    log_info "Verifying installation..."
    
    # Check if verification script exists
    if [[ -f "scripts/setup/verify_installation.py" ]]; then
        log_info "Running verification script..."
        
        if $PYTHON_CMD scripts/setup/verify_installation.py; then
            log_success "Installation verification passed"
        else
            log_warning "Installation verification failed (some features may not work)"
        fi
    else
        log_info "Verification script not found, performing basic checks..."
        
        # Basic import test
        log_info "Testing Python imports..."
        
        if $PYTHON_CMD -c "import torch; import transformers; import numpy; import pandas"; then
            log_success "Core dependencies can be imported"
        else
            log_error "Failed to import core dependencies"
            exit 4
        fi
    fi
    
    # Print installation summary
    print_section "Installation Summary"
    
    echo "Project: ${PROJECT_NAME}"
    echo "Version: ${PROJECT_VERSION}"
    echo "Platform: ${PLATFORM}"
    echo "Python: $($PYTHON_CMD --version)"
    echo "Installation Mode: ${INSTALL_MODE}"
    echo ""
    
    log_success "Installation completed successfully!"
    
    # Print next steps
    print_section "Next Steps"
    
    echo "1. Activate virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Configure environment:"
    echo "   Edit .env file with your settings"
    echo ""
    echo "3. Download full data (optional):"
    echo "   python scripts/setup/download_all_data.py"
    echo ""
    echo "4. Run quick start:"
    echo "   python quickstart/auto_start.py"
    echo ""
    echo "5. Run tests:"
    echo "   pytest tests/"
    echo ""
    echo "For more information, see:"
    echo "   - README.md"
    echo "   - QUICK_START.md"
    echo "   - docs/getting_started/"
    echo ""
}

# ============================================================================
# Cleanup on Error
# ============================================================================

cleanup_on_error() {
    log_error "Installation failed. Cleaning up..."
    
    # Remove virtual environment if clean install was attempted
    if [[ "${CLEAN_INSTALL}" == "true" ]] && [[ -d "venv" ]]; then
        log_info "Removing incomplete virtual environment..."
        rm -rf venv
    fi
}

# Set trap for cleanup
trap cleanup_on_error ERR

# ============================================================================
# Command Line Argument Parsing
# ============================================================================

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                cat << EOF
${PROJECT_NAME} Installation Script

Usage: ${SCRIPT_NAME} [OPTIONS]

Options:
    --help, -h              Show this help message
    --skip-deps             Skip dependency installation
    --dev                   Install development dependencies
    --minimal               Minimal installation (core dependencies only)
    --clean                 Clean installation (remove existing venv)
    --verbose, -v           Verbose output
    --platform PLATFORM     Force platform (local, colab, kaggle)

Examples:
    ${SCRIPT_NAME}                     # Full installation
    ${SCRIPT_NAME} --dev               # Development installation
    ${SCRIPT_NAME} --minimal           # Minimal installation
    ${SCRIPT_NAME} --clean --verbose   # Clean install with verbose output

For more information, see README.md
EOF
                exit 0
                ;;
            --skip-deps)
                SKIP_DEPS=true
                shift
                ;;
            --dev)
                DEV_MODE=true
                INSTALL_MODE="dev"
                shift
                ;;
            --minimal)
                MINIMAL_MODE=true
                INSTALL_MODE="minimal"
                shift
                ;;
            --clean)
                CLEAN_INSTALL=true
                shift
                ;;
            --verbose|-v)
                VERBOSE=true
                shift
                ;;
            --platform)
                PLATFORM="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# Main Installation Flow
# ============================================================================

main() {
    # Print header
    echo ""
    print_color "${COLOR_CYAN}" "============================================================"
    print_color "${COLOR_CYAN}" "  ${PROJECT_NAME} - Installation Script"
    print_color "${COLOR_CYAN}" "  Version: ${PROJECT_VERSION}"
    print_color "${COLOR_CYAN}" "  Author: Võ Hải Dũng"
    print_color "${COLOR_CYAN}" "============================================================"
    echo ""
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Change to project root directory
    cd "$PROJECT_ROOT" || log_error "Failed to change to project root directory"
    
    log_info "Starting installation in ${INSTALL_MODE} mode..."
    
    # Installation steps
    detect_platform
    check_requirements
    create_directories
    setup_virtualenv
    install_dependencies
    setup_configuration
    setup_git_hooks
    download_data
    verify_installation
    
    log_success "All installation steps completed successfully!"
}

# ============================================================================
# Script Entry Point
# ============================================================================

# Only run main if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
