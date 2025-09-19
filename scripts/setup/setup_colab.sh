#!/bin/bash

"""
Google Colab Setup Script for AG News Classification
====================================================

This script configures Google Colab environment for AG News classification following:
- Google Colab Documentation (2023): "Using Google Colab for Machine Learning"
- Bisong (2019): "Google Colaboratory" in "Building Machine Learning and Deep Learning Models on Google Cloud Platform"
- Carneiro et al. (2018): "Performance Analysis of Google Colaboratory as a Tool for Accelerating Deep Learning Applications"

Author: Võ Hải Dũng
License: MIT
"""

set -euo pipefail
IFS=$'\n\t'

# Configuration
readonly SCRIPT_NAME="setup_colab.sh"
readonly PROJECT_NAME="ag-news-text-classification"
readonly GITHUB_REPO="https://github.com/VoHaiDung/ag-news-text-classification.git"
readonly PYTHON_VERSION="3.10"

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly RESET='\033[0m'

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

# Check if running in Google Colab
check_colab_environment() {
    log_header "Checking Colab Environment"
    
    # Check for Colab-specific paths
    if [[ -d "/content" && -f "/etc/lsb-release" ]]; then
        log_success "Running in Google Colab environment"
        
        # Check GPU availability
        if nvidia-smi &> /dev/null; then
            log_info "GPU detected:"
            nvidia-smi --query-gpu=name,memory.total --format=csv
        else
            log_warning "No GPU detected. Using CPU runtime."
            log_info "To enable GPU: Runtime -> Change runtime type -> GPU"
        fi
    else
        log_warning "Not running in Google Colab environment"
        log_info "This script is designed for Google Colab"
        
        read -p "Continue with local setup? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 0
        fi
    fi
}

# Mount Google Drive
mount_google_drive() {
    log_header "Mounting Google Drive"
    
    python3 << 'EOF'
try:
    from google.colab import drive
    import os
    
    # Mount drive
    drive.mount('/content/drive')
    
    # Create project directory in Drive
    project_dir = '/content/drive/MyDrive/ag_news_classification'
    os.makedirs(project_dir, exist_ok=True)
    
    print(f"Google Drive mounted successfully")
    print(f"Project directory: {project_dir}")
    
except ImportError:
    print("Not running in Colab, skipping Drive mount")
except Exception as e:
    print(f"Error mounting Drive: {e}")
EOF
    
    # Create symlink for easy access
    if [[ -d "/content/drive/MyDrive" ]]; then
        ln -sf /content/drive/MyDrive/ag_news_classification /content/ag_news_data
        log_success "Created symlink: /content/ag_news_data -> Drive"
    fi
}

# Clone or update repository
setup_repository() {
    log_header "Setting Up Repository"
    
    cd /content
    
    if [[ -d "$PROJECT_NAME" ]]; then
        log_info "Repository exists, updating..."
        cd "$PROJECT_NAME"
        git pull origin main || log_warning "Could not update repository"
    else
        log_info "Cloning repository..."
        git clone "$GITHUB_REPO" || {
            log_error "Failed to clone repository"
            log_info "Please check repository URL: $GITHUB_REPO"
            exit 1
        }
        cd "$PROJECT_NAME"
    fi
    
    log_success "Repository ready at: /content/$PROJECT_NAME"
}

# Install system dependencies
install_system_dependencies() {
    log_header "Installing System Dependencies"
    
    log_info "Updating package lists..."
    apt-get update -qq
    
    log_info "Installing required system packages..."
    apt-get install -y -qq \
        build-essential \
        git \
        wget \
        curl \
        vim \
        tree \
        htop \
        tmux \
        graphviz \
        libgraphviz-dev \
        > /dev/null 2>&1
    
    # Install additional ML-related system packages
    apt-get install -y -qq \
        libopenblas-dev \
        liblapack-dev \
        libhdf5-dev \
        > /dev/null 2>&1
    
    log_success "System dependencies installed"
}

# Configure Python environment
configure_python() {
    log_header "Configuring Python Environment"
    
    # Check Python version
    local python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_info "Python version: $python_version"
    
    # Upgrade pip
    log_info "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel -q
    
    log_success "Python environment configured"
}

# Install project dependencies
install_dependencies() {
    log_header "Installing Project Dependencies"
    
    cd "/content/$PROJECT_NAME"
    
    # Install base requirements
    log_info "Installing base requirements..."
    if [[ -f "requirements/base.txt" ]]; then
        pip install -r requirements/base.txt -q
    fi
    
    # Install ML requirements
    log_info "Installing ML requirements..."
    if [[ -f "requirements/ml.txt" ]]; then
        pip install -r requirements/ml.txt -q
    fi
    
    # Install Colab-specific packages
    log_info "Installing Colab-specific packages..."
    pip install -q \
        google-colab \
        kaggle \
        wandb \
        ipywidgets \
        plotly \
        seaborn
    
    # Enable widgets
    jupyter nbextension enable --py widgetsnbextension --sys-prefix
    
    log_success "Dependencies installed"
}

# Setup Weights & Biases
setup_wandb() {
    log_header "Setting Up Weights & Biases"
    
    python3 << 'EOF'
try:
    import wandb
    import os
    
    # Check if API key is set
    if 'WANDB_API_KEY' in os.environ:
        print("W&B API key found in environment")
    else:
        print("W&B API key not found")
        print("To use W&B tracking:")
        print("  1. Get your API key from https://wandb.ai/authorize")
        print("  2. Run: wandb.login(key='your-api-key')")
        
except ImportError:
    print("W&B not installed")
EOF
}

# Download and prepare data
download_data() {
    log_header "Downloading AG News Dataset"
    
    cd "/content/$PROJECT_NAME"
    
    # Create data directory
    mkdir -p data/raw data/processed
    
    # Run data download script
    if [[ -f "scripts/setup/download_all_data.py" ]]; then
        log_info "Downloading AG News dataset..."
        python scripts/setup/download_all_data.py --sources ag_news || {
            log_warning "Data download script failed, trying alternative method..."
            
            # Alternative: Download from Hugging Face
            python3 << 'EOF'
from datasets import load_dataset
import pandas as pd
import os

print("Downloading AG News from Hugging Face...")
dataset = load_dataset("ag_news")

# Save to CSV
os.makedirs("data/raw/ag_news", exist_ok=True)

# Convert train set
train_df = pd.DataFrame(dataset["train"])
train_df.to_csv("data/raw/ag_news/train.csv", index=False)

# Convert test set
test_df = pd.DataFrame(dataset["test"])
test_df.to_csv("data/raw/ag_news/test.csv", index=False)

print(f"Downloaded {len(train_df)} train samples and {len(test_df)} test samples")
EOF
        }
    fi
    
    # Prepare processed data
    if [[ -f "scripts/data_preparation/prepare_ag_news.py" ]]; then
        log_info "Preparing processed data..."
        python scripts/data_preparation/prepare_ag_news.py
    fi
    
    log_success "Data ready"
}

# Configure Kaggle API
setup_kaggle() {
    log_header "Setting Up Kaggle API"
    
    python3 << 'EOF'
import os

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

print("Kaggle API setup:")
print("  1. Get your API key from https://www.kaggle.com/account")
print("  2. Upload kaggle.json using:")
print("     from google.colab import files")
print("     files.upload()  # Select kaggle.json")
print("  3. Move to ~/.kaggle/kaggle.json")
EOF
}

# Create Colab-specific utilities
create_colab_utils() {
    log_header "Creating Colab Utilities"
    
    # Create utility script
    cat > "/content/$PROJECT_NAME/colab_utils.py" << 'EOF'
"""
Google Colab Utilities for AG News Classification
"""

import os
import sys
from pathlib import Path
from typing import Optional
import subprocess

def setup_project():
    """Setup project in Colab environment."""
    # Add project to path
    project_root = Path("/content/ag-news-text-classification")
    if project_root.exists():
        sys.path.insert(0, str(project_root))
    
    # Change to project directory
    os.chdir(project_root)
    
    print(f"Project setup complete at: {project_root}")

def check_gpu():
    """Check GPU availability and info."""
    import torch
    
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Memory usage
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"Memory Allocated: {allocated:.2f} GB")
        print(f"Memory Reserved: {reserved:.2f} GB")
    else:
        print("No GPU available. Using CPU.")
        print("To enable GPU: Runtime -> Change runtime type -> GPU")

def mount_drive():
    """Mount Google Drive."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted at /content/drive")
    except ImportError:
        print("Not running in Colab")

def save_to_drive(source_path: str, dest_name: Optional[str] = None):
    """Save file or directory to Google Drive."""
    from shutil import copytree, copy2
    import os
    
    source = Path(source_path)
    drive_dir = Path("/content/drive/MyDrive/ag_news_classification")
    drive_dir.mkdir(parents=True, exist_ok=True)
    
    if dest_name:
        dest = drive_dir / dest_name
    else:
        dest = drive_dir / source.name
    
    if source.is_dir():
        copytree(source, dest, dirs_exist_ok=True)
    else:
        copy2(source, dest)
    
    print(f"Saved to Drive: {dest}")

def load_from_drive(file_name: str, dest_path: Optional[str] = None):
    """Load file from Google Drive."""
    from shutil import copy2
    
    drive_file = Path(f"/content/drive/MyDrive/ag_news_classification/{file_name}")
    
    if not drive_file.exists():
        print(f"File not found in Drive: {drive_file}")
        return None
    
    if dest_path:
        dest = Path(dest_path)
    else:
        dest = Path(file_name)
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    copy2(drive_file, dest)
    
    print(f"Loaded from Drive: {dest}")
    return dest

def install_requirements():
    """Install project requirements."""
    requirements = [
        "requirements/base.txt",
        "requirements/ml.txt"
    ]
    
    for req_file in requirements:
        if Path(req_file).exists():
            print(f"Installing {req_file}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file, "-q"])

def download_model(model_name: str = "bert-base-uncased"):
    """Pre-download model to avoid timeout."""
    from transformers import AutoModel, AutoTokenizer
    
    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print(f"Model {model_name} ready")

# Auto-setup when imported
setup_project()
check_gpu()
EOF
    
    log_success "Colab utilities created at: colab_utils.py"
}

# Create sample notebook
create_sample_notebook() {
    log_header "Creating Sample Notebook"
    
    cat > "/content/$PROJECT_NAME/colab_quickstart.ipynb" << 'EOF'
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "AG News Classification - Quick Start",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# AG News Classification - Google Colab Quick Start\n",
        "\n",
        "This notebook demonstrates AG News text classification in Google Colab."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Setup environment\n",
        "!bash /content/ag-news-text-classification/scripts/setup/setup_colab.sh"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Import utilities\n",
        "import sys\n",
        "sys.path.insert(0, '/content/ag-news-text-classification')\n",
        "from colab_utils import *\n",
        "\n",
        "# Check GPU\n",
        "check_gpu()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Quick training example\n",
        "!cd /content/ag-news-text-classification && python quickstart/train_simple.py --num-epochs 1 --batch-size 16"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}
EOF
    
    log_success "Sample notebook created: colab_quickstart.ipynb"
}

# Verify installation
verify_installation() {
    log_header "Verifying Installation"
    
    cd "/content/$PROJECT_NAME"
    
    # Run verification script
    if [[ -f "scripts/setup/verify_installation.py" ]]; then
        python scripts/setup/verify_installation.py --quick || log_warning "Some checks failed"
    fi
    
    # Quick import test
    python3 << 'EOF'
print("\nTesting imports...")
try:
    import torch
    import transformers
    import datasets
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Datasets: {datasets.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
EOF
    
    log_success "Verification complete"
}

# Print usage instructions
print_instructions() {
    log_header "Setup Complete - Usage Instructions"
    
    cat << EOF
Your Colab environment is ready for AG News classification!

Quick Start Commands:
---------------------
1. Training:
   !cd /content/$PROJECT_NAME && python quickstart/train_simple.py

2. Full training:
   !cd /content/$PROJECT_NAME && python scripts/training/train_single_model.py --model roberta

3. Evaluation:
   !cd /content/$PROJECT_NAME && python scripts/evaluation/evaluate_model.py

Colab Tips:
-----------
- GPU Status: Runtime -> View resources
- Save to Drive: use colab_utils.save_to_drive()
- Prevent timeout: keep tab active
- Download results: files.download('path/to/file')

Useful Paths:
-------------
- Project: /content/$PROJECT_NAME
- Data: /content/$PROJECT_NAME/data
- Models: /content/$PROJECT_NAME/outputs/models
- Drive: /content/drive/MyDrive/ag_news_classification

For more examples, see:
- /content/$PROJECT_NAME/notebooks/
- /content/$PROJECT_NAME/colab_quickstart.ipynb

Happy experimenting!
EOF
}

# Main execution
main() {
    log_header "Google Colab Setup for AG News Classification"
    
    echo "This script configures Google Colab for AG News text classification."
    echo "Following best practices from Google Colab documentation."
    echo ""
    
    # Setup steps
    check_colab_environment
    mount_google_drive
    setup_repository
    install_system_dependencies
    configure_python
    install_dependencies
    setup_wandb
    download_data
    setup_kaggle
    create_colab_utils
    create_sample_notebook
    verify_installation
    print_instructions
    
    log_success "Colab setup complete!"
}

# Run main function
main "$@"
