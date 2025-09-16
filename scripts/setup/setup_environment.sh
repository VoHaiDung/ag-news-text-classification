#!/bin/bash
# -*- coding: utf-8 -*-

################################################################################
# AG News Text Classification - Environment Setup Script
################################################################################
#
# This script sets up the complete development/research environment following
# best practices from:
# - Wilson et al. (2017): "Good enough practices in scientific computing"
# - Taschuk & Wilson (2017): "Ten simple rules for making research software more robust"
#
# Author: Võ Hải Dũng
# License: MIT
################################################################################

set -euo pipefail  # Exit on error, undefined variables, pipe failures
IFS=$'\n\t'       # Set Internal Field Separator

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly MAGENTA='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[0;37m'
readonly RESET='\033[0m'

# Project configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly PYTHON_VERSION="3.10"
readonly VENV_NAME="venv"
readonly CUDA_VERSION="11.8"

# Logging functions
log_info() {
    echo -e "${CYAN}[INFO]${RESET} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${RESET} $1"
}

log_header() {
    echo -e "\n${BLUE}=================================================================================${RESET}"
    echo -e "${BLUE}$1${RESET}"
    echo -e "${BLUE}=================================================================================${RESET}\n"
}

# Detect operating system
detect_os() {
    case "$OSTYPE" in
        linux*)   echo "linux" ;;
        darwin*)  echo "macos" ;;
        msys*)    echo "windows" ;;
        cygwin*)  echo "windows" ;;
        *)        echo "unknown" ;;
    esac
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
check_python_version() {
    log_info "Checking Python version..."
    
    if ! command_exists python3; then
        log_error "Python 3 is not installed"
        return 1
    fi
    
    local python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    
    if [[ "$python_version" != "$PYTHON_VERSION" ]]; then
        log_warning "Python $PYTHON_VERSION required, found $python_version"
        log_info "Attempting to use python$PYTHON_VERSION..."
        
        if command_exists "python$PYTHON_VERSION"; then
            alias python3="python$PYTHON_VERSION"
            log_success "Using python$PYTHON_VERSION"
        else
            log_error "Python $PYTHON_VERSION not found"
            return 1
        fi
    else
        log_success "Python $python_version detected"
    fi
}

# Install system dependencies
install_system_dependencies() {
    log_header "Installing System Dependencies"
    
    local os_type=$(detect_os)
    
    case "$os_type" in
        linux)
            log_info "Installing Linux dependencies..."
            
            # Update package list
            sudo apt-get update -qq
            
            # Install essential packages
            sudo apt-get install -y -qq \
                build-essential \
                python3-dev \
                python3-pip \
                python3-venv \
                git \
                wget \
                curl \
                vim \
                htop \
                tmux \
                tree \
                unzip \
                software-properties-common
            
            # Install ML-specific packages
            sudo apt-get install -y -qq \
                libopenblas-dev \
                liblapack-dev \
                libhdf5-dev \
                graphviz
            
            log_success "Linux dependencies installed"
            ;;
            
        macos)
            log_info "Installing macOS dependencies..."
            
            # Check for Homebrew
            if ! command_exists brew; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            # Install packages
            brew install \
                python@$PYTHON_VERSION \
                git \
                wget \
                htop \
                tmux \
                tree \
                graphviz
            
            log_success "macOS dependencies installed"
            ;;
            
        windows)
            log_warning "Windows detected. Please install dependencies manually:"
            echo "  1. Python $PYTHON_VERSION from python.org"
            echo "  2. Git from git-scm.com"
            echo "  3. Visual Studio Build Tools"
            ;;
            
        *)
            log_warning "Unknown OS. Please install dependencies manually."
            ;;
    esac
}

# Setup Python virtual environment
setup_virtual_environment() {
    log_header "Setting Up Python Virtual Environment"
    
    cd "$PROJECT_ROOT"
    
    # Remove existing venv if requested
    if [[ -d "$VENV_NAME" ]]; then
        log_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf "$VENV_NAME"
        else
            log_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    # Create virtual environment
    log_info "Creating virtual environment..."
    python3 -m venv "$VENV_NAME"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip, setuptools, wheel
    log_info "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment created and activated"
}

# Install Python dependencies
install_python_dependencies() {
    log_header "Installing Python Dependencies"
    
    cd "$PROJECT_ROOT"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Install base requirements
    log_info "Installing base requirements..."
    pip install -r requirements/base.txt
    
    # Ask for additional requirements
    log_info "Select additional requirements to install:"
    echo "  1) Development (testing, linting, formatting)"
    echo "  2) Research (experiment tracking, visualization)"
    echo "  3) Production (API, monitoring, deployment)"
    echo "  4) ML extras (additional ML libraries)"
    echo "  5) All"
    echo "  0) None"
    
    read -p "Enter your choice (0-5): " choice
    
    case $choice in
        1)
            log_info "Installing development requirements..."
            pip install -r requirements/dev.txt
            ;;
        2)
            log_info "Installing research requirements..."
            pip install -r requirements/research.txt
            ;;
        3)
            log_info "Installing production requirements..."
            pip install -r requirements/prod.txt
            ;;
        4)
            log_info "Installing ML requirements..."
            pip install -r requirements/ml.txt
            ;;
        5)
            log_info "Installing all requirements..."
            pip install -r requirements/all.txt
            ;;
        *)
            log_info "Skipping additional requirements"
            ;;
    esac
    
    # Install package in editable mode
    log_info "Installing package in editable mode..."
    pip install -e .
    
    log_success "Python dependencies installed"
}

# Setup CUDA and GPU
setup_cuda() {
    log_header "Setting Up CUDA Environment"
    
    # Check for NVIDIA GPU
    if ! command_exists nvidia-smi; then
        log_warning "NVIDIA GPU not detected or drivers not installed"
        log_info "Skipping CUDA setup"
        return 0
    fi
    
    log_info "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    
    # Check CUDA version
    if command_exists nvcc; then
        local cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "CUDA $cuda_version detected"
        
        if [[ "$cuda_version" != "$CUDA_VERSION"* ]]; then
            log_warning "CUDA $CUDA_VERSION recommended, found $cuda_version"
        fi
    else
        log_warning "CUDA toolkit not found"
        log_info "Please install CUDA $CUDA_VERSION from https://developer.nvidia.com/cuda-toolkit"
    fi
    
    # Set CUDA environment variables
    log_info "Setting CUDA environment variables..."
    {
        echo "# CUDA Configuration"
        echo "export CUDA_HOME=/usr/local/cuda-$CUDA_VERSION"
        echo "export PATH=\$CUDA_HOME/bin:\$PATH"
        echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
        echo "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
        echo ""
        echo "# PyTorch CUDA settings"
        echo "export TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6'"
        echo "export CUBLAS_WORKSPACE_CONFIG=:4096:8"
    } >> "$PROJECT_ROOT/.env"
    
    log_success "CUDA environment configured"
}

# Setup Git hooks
setup_git_hooks() {
    log_header "Setting Up Git Hooks"
    
    cd "$PROJECT_ROOT"
    
    # Install pre-commit
    if command_exists pre-commit; then
        log_info "Installing pre-commit hooks..."
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    else
        log_warning "pre-commit not found. Install with: pip install pre-commit"
    fi
    
    # Setup Husky (if Node.js is available)
    if command_exists npm && [[ -f "package.json" ]]; then
        log_info "Installing Husky hooks..."
        npm install
        npx husky install
        log_success "Husky hooks installed"
    fi
}

# Download data
download_data() {
    log_header "Downloading Data"
    
    cd "$PROJECT_ROOT"
    source "$VENV_NAME/bin/activate"
    
    log_info "Downloading AG News dataset and resources..."
    python scripts/setup/download_all_data.py --sources ag_news
    
    log_success "Data downloaded"
}

# Verify installation
verify_installation() {
    log_header "Verifying Installation"
    
    cd "$PROJECT_ROOT"
    source "$VENV_NAME/bin/activate"
    
    # Check Python packages
    log_info "Checking Python packages..."
    python -c "
import torch
import transformers
import datasets
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Datasets: {datasets.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"
    
    # Run basic tests
    log_info "Running basic tests..."
    python -m pytest tests/unit/test_setup.py -v || log_warning "Some tests failed"
    
    log_success "Installation verified"
}

# Create environment file
create_env_file() {
    log_header "Creating Environment File"
    
    cd "$PROJECT_ROOT"
    
    if [[ -f ".env" ]]; then
        log_warning ".env file already exists"
        return 0
    fi
    
    log_info "Creating .env file from template..."
    cp .env.example .env
    
    # Update paths
    sed -i "s|ROOT_DIR=.*|ROOT_DIR=$PROJECT_ROOT|g" .env
    
    log_success ".env file created"
}

# Main setup function
main() {
    log_header "AG News Text Classification - Environment Setup"
    
    echo "This script will set up your development environment."
    echo "Project root: $PROJECT_ROOT"
    echo ""
    
    # Run setup steps
    check_python_version
    install_system_dependencies
    setup_virtual_environment
    install_python_dependencies
    setup_cuda
    setup_git_hooks
    create_env_file
    download_data
    verify_installation
    
    # Final message
    log_header "Setup Complete!"
    
    echo -e "${GREEN}Your environment is ready!${RESET}"
    echo ""
    echo "To activate the virtual environment, run:"
    echo -e "  ${CYAN}source $VENV_NAME/bin/activate${RESET}"
    echo ""
    echo "To start training, run:"
    echo -e "  ${CYAN}make train${RESET}"
    echo ""
    echo "For more information, see README.md"
    
    log_success "Happy researching! 🚀"
}

# Run main function
main "$@"
