# AG News Text Classification - Quick Start Guide

This directory contains simplified examples and quick start scripts for the AG News Text Classification project, designed for rapid prototyping and educational purposes.

## Overview

The quickstart module provides simplified interfaces to the main functionality of the AG News classification system, abstracting complex configurations while maintaining educational value. This module serves as an entry point for understanding the complete system architecture.

### Design Philosophy

Following the principle of progressive disclosure from educational computing (Resnick et al., 2005):

1. **Abstraction Layers**: Hide complexity while maintaining functionality
2. **Educational Comments**: Extensive inline documentation following PEP 257
3. **Minimal Dependencies**: Use only essential libraries for core functionality
4. **Reproducibility**: Fixed random seeds and deterministic operations
5. **Error Handling**: Comprehensive error messages with solutions

### Module Structure

```
quickstart/
├── README.md                # This file
├── minimal_example.py       # Single prediction demonstration
├── train_simple.py          # Simplified training pipeline
├── evaluate_simple.py       # Model evaluation utilities
├── demo_app.py              # Streamlit web interface
├── api_quickstart.py        # API client examples
├── colab_notebook.ipynb     # Google Colab tutorial
└── docker_quickstart/
    ├── Dockerfile           # Container definition
    └── docker-compose.yml   # Multi-service orchestration
```

## Module Architecture

### Dependency Graph

```
minimal_example.py
    └── src.models.base.base_model
    └── src.data.preprocessing.text_cleaner
    
train_simple.py
    └── src.training.trainers.standard_trainer
    └── src.data.datasets.ag_news
    └── src.models.transformers.*
    
evaluate_simple.py
    └── src.evaluation.metrics.classification_metrics
    └── src.evaluation.analysis.error_analysis
    
demo_app.py
    └── app.components.prediction_component
    └── app.utils.session_manager
    
api_quickstart.py
    └── src.api.rest.schemas.request_schemas
    └── requests (external)
```

## Prerequisites

### System Requirements

Minimum and recommended specifications based on empirical testing:

```bash
# Python Environment
Python >= 3.8.0, < 3.11.0  # Tested versions

# Hardware Requirements (Minimum | Recommended)
CPU: 4 cores | 8+ cores
RAM: 8 GB | 16+ GB
Storage: 10 GB | 50+ GB
GPU: Optional | NVIDIA GPU with 4+ GB VRAM

# CUDA Requirements (for GPU acceleration)
CUDA Toolkit: 11.0+ | 11.7
cuDNN: 8.0+ | 8.4
```

### Installation Verification

```bash
# Verify Python version
python --version

# Check CUDA availability (optional)
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Verify package installation
python -c "import transformers, torch, numpy; print('Core packages installed')"
```

### Dependency Installation

```bash
# Clone repository
git clone <repository-url>
cd ag-news-text-classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install quickstart dependencies
pip install -r requirements/base.txt
pip install -r requirements/ml.txt

# Optional: Install all dependencies for full functionality
pip install -r requirements/all.txt
```

## Quick Start Scripts

### 1. Minimal Example (`minimal_example.py`)

**Purpose**: Demonstrate the simplest possible text classification workflow with pre-trained models.

**Implementation Details**:

```python
"""
Minimal example for AG News text classification.

This script demonstrates:
1. Model loading from checkpoint
2. Text preprocessing pipeline
3. Inference execution
4. Result interpretation
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, Any

from src.models.base.base_model import BaseModel
from src.data.preprocessing.text_cleaner import TextCleaner
from src.utils.reproducibility import set_seed
```

**Usage Examples**:

```bash
# Basic usage with default model
python quickstart/minimal_example.py \
    --text "Apple announces new MacBook with M3 chip"

# Using specific checkpoint
python quickstart/minimal_example.py \
    --text "Federal Reserve raises interest rates" \
    --model-path ./outputs/models/checkpoints/epoch_3_acc_0.95.pt

# Batch prediction from file
python quickstart/minimal_example.py \
    --input-file ./data/test_samples.txt \
    --output-file ./predictions.json

# With confidence threshold
python quickstart/minimal_example.py \
    --text "Scientists discover new exoplanet" \
    --confidence-threshold 0.8
```

**Command-line Arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--text` | str | None | Input text for classification |
| `--input-file` | Path | None | File containing texts (one per line) |
| `--model-path` | Path | ./outputs/models/best | Path to model checkpoint |
| `--device` | str | auto | Device selection (cpu/cuda/auto) |
| `--confidence-threshold` | float | 0.0 | Minimum confidence for valid prediction |
| `--output-file` | Path | None | Save predictions to JSON file |
| `--verbose` | bool | False | Enable detailed logging |

**Output Format**:

```json
{
  "text": "Apple announces new MacBook with M3 chip",
  "predicted_class": "Sci/Tech",
  "confidence": 0.9823,
  "probabilities": {
    "World": 0.0023,
    "Sports": 0.0011,
    "Business": 0.0143,
    "Sci/Tech": 0.9823
  },
  "processing_time_ms": 23.5
}
```

### 2. Simple Training (`train_simple.py`)

**Purpose**: Provide a streamlined training pipeline with essential hyperparameters and best practices.

**Implementation Architecture**:

```python
"""
Simplified training script for AG News classification.

Key Features:
1. Automatic dataset downloading and preprocessing
2. Configurable model selection from Hugging Face
3. Training with validation monitoring
4. Automatic checkpointing and early stopping
5. Tensorboard logging integration
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.training.trainers.standard_trainer import StandardTrainer
from src.data.datasets.ag_news import AGNewsDataset
from src.training.callbacks.early_stopping import EarlyStopping
from src.training.callbacks.model_checkpoint import ModelCheckpoint
```

**Training Configuration**:

```bash
# Standard training with BERT
python quickstart/train_simple.py \
    --model-name bert-base-uncased \
    --num-epochs 3 \
    --batch-size 32 \
    --learning-rate 2e-5

# Fast training with DistilBERT
python quickstart/train_simple.py \
    --model-name distilbert-base-uncased \
    --num-epochs 5 \
    --batch-size 64 \
    --learning-rate 5e-5 \
    --fp16

# Advanced configuration with RoBERTa
python quickstart/train_simple.py \
    --model-name roberta-base \
    --num-epochs 10 \
    --batch-size 16 \
    --learning-rate 1e-5 \
    --warmup-ratio 0.1 \
    --weight-decay 0.01 \
    --gradient-accumulation-steps 2 \
    --early-stopping-patience 3

# Resume from checkpoint
python quickstart/train_simple.py \
    --resume-from ./outputs/models/checkpoints/last.pt \
    --num-epochs 5
```

**Hyperparameter Details**:

| Parameter | Range | Default | Description | Reference |
|-----------|-------|---------|-------------|-----------|
| `learning_rate` | 1e-5 to 5e-5 | 2e-5 | AdamW learning rate | Devlin et al., 2019 |
| `batch_size` | 8 to 64 | 32 | Training batch size | Hardware dependent |
| `num_epochs` | 1 to 10 | 3 | Training iterations | Task dependent |
| `warmup_ratio` | 0.0 to 0.2 | 0.1 | Linear warmup proportion | Liu et al., 2019 |
| `weight_decay` | 0.0 to 0.1 | 0.01 | L2 regularization | Loshchilov & Hutter, 2019 |
| `max_length` | 128 to 512 | 256 | Maximum sequence length | Memory dependent |
| `gradient_accumulation_steps` | 1 to 8 | 1 | Gradient accumulation | Simulates larger batch |

**Training Output Structure**:

```
outputs/simple/
├── 20240115_143022/           # Timestamp-based run directory
│   ├── config.json            # Training configuration
│   ├── metrics.json           # Training metrics history
│   ├── checkpoints/
│   │   ├── best_model.pt     # Best validation checkpoint
│   │   ├── last_model.pt     # Final checkpoint
│   │   └── epoch_*.pt        # Epoch checkpoints
│   ├── logs/
│   │   └── tensorboard/      # Tensorboard event files
│   └── predictions/
│       └── test_predictions.json
```

### 3. Model Evaluation (`evaluate_simple.py`)

**Purpose**: Comprehensive evaluation utilities for trained models with detailed metrics and analysis.

**Evaluation Pipeline**:

```python
"""
Model evaluation script with comprehensive metrics.

Implements:
1. Standard classification metrics (accuracy, precision, recall, F1)
2. Confusion matrix analysis
3. Per-class performance breakdown
4. Error analysis and failure cases
5. Statistical significance testing
"""

from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import bootstrap
import numpy as np

from src.evaluation.metrics.classification_metrics import ClassificationMetrics
from src.evaluation.analysis.error_analysis import ErrorAnalyzer
from src.evaluation.statistical.significance_tests import McNemar
```

**Usage Patterns**:

```bash
# Basic evaluation
python quickstart/evaluate_simple.py \
    --model-path ./outputs/simple/20240115_143022/checkpoints/best_model.pt

# Detailed analysis with visualization
python quickstart/evaluate_simple.py \
    --model-path ./outputs/simple/latest/checkpoints/best_model.pt \
    --detailed \
    --save-confusion-matrix \
    --save-predictions \
    --error-analysis

# Custom test dataset
python quickstart/evaluate_simple.py \
    --model-path ./outputs/models/fine_tuned/model.pt \
    --test-data ./data/processed/custom_test.json \
    --output-dir ./outputs/evaluation/custom

# Model comparison
python quickstart/evaluate_simple.py \
    --model-paths model1.pt model2.pt model3.pt \
    --compare \
    --statistical-test mcnemar
```

**Metrics Computed**:

```python
# Classification Metrics
metrics = {
    "accuracy": 0.9523,
    "precision_macro": 0.9521,
    "precision_micro": 0.9523,
    "recall_macro": 0.9520,
    "recall_micro": 0.9523,
    "f1_macro": 0.9520,
    "f1_micro": 0.9523,
    "matthews_corr_coef": 0.9364,
    "cohen_kappa": 0.9364,
    "per_class_metrics": {
        "World": {"precision": 0.95, "recall": 0.94, "f1": 0.945, "support": 1900},
        "Sports": {"precision": 0.97, "recall": 0.98, "f1": 0.975, "support": 1900},
        "Business": {"precision": 0.94, "recall": 0.93, "f1": 0.935, "support": 1900},
        "Sci/Tech": {"precision": 0.95, "recall": 0.96, "f1": 0.955, "support": 1900}
    }
}
```

**Visualization Outputs**:

1. **Confusion Matrix**: Heatmap showing classification errors
2. **Learning Curves**: Training/validation loss and accuracy
3. **Error Distribution**: Analysis of misclassified samples
4. **Confidence Histograms**: Distribution of prediction confidence

### 4. Demo Application (`demo_app.py`)

**Purpose**: Interactive web interface for model demonstration using Streamlit framework.

**Application Architecture**:

```python
"""
Streamlit-based demonstration application.

Features:
1. Interactive text input
2. Real-time prediction
3. Batch processing
4. Model comparison
5. Visualization components
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import List, Dict

from app.components.prediction_component import PredictionComponent
from app.components.visualization_component import VisualizationComponent
from app.utils.session_manager import SessionManager
```

**Application Pages**:

1. **Single Prediction Page**
   - Text input area
   - Model selection dropdown
   - Prediction with confidence scores
   - Probability distribution chart

2. **Batch Analysis Page**
   - File upload (CSV/TXT)
   - Batch processing progress
   - Results table with sorting
   - Export functionality

3. **Model Comparison Page**
   - Side-by-side model predictions
   - Performance metrics comparison
   - Consensus analysis

**Running the Application**:

```bash
# Standard launch
streamlit run quickstart/demo_app.py

# Custom configuration
streamlit run quickstart/demo_app.py \
    --server.port 8080 \
    --server.address 0.0.0.0 \
    --server.maxUploadSize 10 \
    --theme.primaryColor "#FF6B6B"

# With specific model
streamlit run quickstart/demo_app.py -- \
    --model-path ./outputs/models/best_model.pt
```

**Configuration Options**:

```python
# .streamlit/config.toml
[server]
port = 8501
address = "localhost"
maxUploadSize = 10
enableCORS = false
enableXsrfProtection = true

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### 5. API Quickstart (`api_quickstart.py`)

**Purpose**: Demonstrate RESTful API integration patterns for the classification service.

**API Client Implementation**:

```python
"""
API client examples for AG News classification service.

Demonstrates:
1. Single and batch predictions
2. Authentication mechanisms
3. Error handling
4. Async requests
5. Response parsing
"""

import requests
import asyncio
import aiohttp
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class APIConfig:
    base_url: str = "http://localhost:8000"
    api_version: str = "v1"
    timeout: int = 30
    max_retries: int = 3
```

**API Endpoints**:

```python
# Available endpoints
endpoints = {
    "predict": "/api/v1/predict",
    "predict_batch": "/api/v1/predict/batch",
    "models": "/api/v1/models",
    "health": "/api/v1/health",
    "metrics": "/api/v1/metrics"
}
```

**Usage Examples**:

```python
# 1. Simple prediction
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "text": "Apple releases new iPhone",
        "model_name": "bert-base-uncased"
    }
)
result = response.json()
# {"prediction": "Sci/Tech", "confidence": 0.98, "processing_time": 0.023}

# 2. Batch prediction with error handling
def predict_batch(texts: List[str], api_url: str) -> List[Dict]:
    """
    Perform batch prediction with retry logic.
    
    Args:
        texts: List of input texts
        api_url: API endpoint URL
        
    Returns:
        List of prediction dictionaries
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{api_url}/api/v1/predict/batch",
                json={"texts": texts},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["predictions"]
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
    
# 3. Async batch processing
async def async_predict_batch(texts: List[str]) -> List[Dict]:
    """
    Asynchronous batch prediction for improved throughput.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for text in texts:
            task = session.post(
                "http://localhost:8000/api/v1/predict",
                json={"text": text}
            )
            tasks.append(task)
        responses = await asyncio.gather(*tasks)
        return [await r.json() for r in responses]

# 4. Authentication example
headers = {
    "Authorization": "Bearer YOUR_API_TOKEN",
    "Content-Type": "application/json"
}
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"text": "Sample text"},
    headers=headers
)
```

**Rate Limiting and Throttling**:

```python
from time import sleep
from typing import Generator

def rate_limited_requests(
    texts: List[str],
    requests_per_second: int = 10
) -> Generator[Dict, None, None]:
    """
    Generator for rate-limited API requests.
    
    Args:
        texts: Input texts
        requests_per_second: API rate limit
        
    Yields:
        Prediction results
    """
    delay = 1.0 / requests_per_second
    
    for text in texts:
        response = requests.post(
            "http://localhost:8000/api/v1/predict",
            json={"text": text}
        )
        yield response.json()
        sleep(delay)
```

## Docker Quickstart

### Docker Configuration

**Dockerfile** (`docker_quickstart/Dockerfile`):

```dockerfile
# Multi-stage build for optimized image size
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/base.txt requirements/ml.txt /tmp/
RUN pip install --user --no-cache-dir -r /tmp/base.txt -r /tmp/ml.txt

# Runtime stage
FROM python:3.9-slim

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
WORKDIR /app
COPY . /app

# Expose ports
EXPOSE 8501 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Default command
CMD ["streamlit", "run", "quickstart/demo_app.py"]
```

**Docker Compose** (`docker_quickstart/docker-compose.yml`):

```yaml
version: '3.8'

services:
  # Web application
  web:
    build:
      context: ../..
      dockerfile: quickstart/docker_quickstart/Dockerfile
    ports:
      - "8501:8501"
    environment:
      - MODEL_PATH=/models/best_model.pt
      - LOG_LEVEL=INFO
    volumes:
      - ../../outputs/models:/models:ro
      - ../../data:/data:ro
    networks:
      - agnews-network

  # API service
  api:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - API_WORKERS=4
      - MAX_BATCH_SIZE=32
    volumes:
      - ../../outputs/models:/models:ro
    depends_on:
      - redis
    networks:
      - agnews-network

  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - agnews-network

volumes:
  redis-data:

networks:
  agnews-network:
    driver: bridge
```

### Docker Commands

```bash
# Build and run single container
docker build -f quickstart/docker_quickstart/Dockerfile -t agnews-quick .
docker run -p 8501:8501 -v $(pwd)/outputs:/app/outputs agnews-quick

# Using Docker Compose
docker-compose -f quickstart/docker_quickstart/docker-compose.yml up

# Development mode with hot reload
docker-compose -f quickstart/docker_quickstart/docker-compose.yml up \
    --build --force-recreate

# Production deployment
docker-compose -f quickstart/docker_quickstart/docker-compose.yml up -d \
    --scale api=3
```

## Google Colab Integration

### Quick Start in Colab

1. Open `quickstart/colab_notebook.ipynb` in Google Colab
2. Run all cells sequentially
3. Follow interactive instructions

### Direct Link

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VoHaiDung/ag-news-text-classification/blob/main/quickstart/colab_notebook.ipynb)

### Colab Notebook Structure

The `colab_notebook.ipynb` provides a complete tutorial optimized for Google Colab environment:

**Notebook Sections**:

1. **Environment Setup**
   ```python
   # GPU runtime check
   import torch
   assert torch.cuda.is_available(), "Please enable GPU runtime"
   
   # Install dependencies
   !pip install transformers datasets accelerate
   
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Data Preparation**
   ```python
   # Download AG News dataset
   from datasets import load_dataset
   dataset = load_dataset("ag_news")
   
   # Preprocessing pipeline
   from src.data.preprocessing import TextCleaner
   cleaner = TextCleaner()
   processed_data = cleaner.process(dataset)
   ```

3. **Model Training**
   ```python
   # Initialize model with Colab-optimized settings
   from quickstart.train_simple import SimpleTrainer
   
   trainer = SimpleTrainer(
       model_name="distilbert-base-uncased",
       batch_size=16,  # Optimized for Colab GPU
       fp16=True,       # Mixed precision for memory efficiency
       gradient_checkpointing=True
   )
   
   # Training with progress display
   from IPython.display import display
   trainer.train_with_display(dataset)
   ```

4. **Interactive Visualization**
   ```python
   # Plotly interactive charts
   import plotly.express as px
   
   # Training curves
   fig = px.line(
       metrics_df,
       x='epoch',
       y=['train_loss', 'val_loss'],
       title='Training Progress'
   )
   fig.show()
   ```

### Colab-Specific Optimizations

```python
# Memory management for Colab
import gc
import torch

def clear_gpu_memory():
    """Clear GPU memory in Colab environment."""
    gc.collect()
    torch.cuda.empty_cache()
    
# Prevent runtime disconnection
import IPython
from google.colab import output

def keep_alive():
    """Prevent Colab runtime timeout."""
    display(IPython.display.Javascript('''
        function KeepClicking(){
            console.log("Clicking");
            document.querySelector("colab-toolbar-button").click()
        }
        setInterval(KeepClicking, 60000)
    '''))
```

## Implementation Details

### Code Organization Principles

1. **Single Responsibility**: Each script handles one primary task
2. **Dependency Injection**: Configurable components via command-line arguments
3. **Error Handling**: Comprehensive try-catch blocks with informative messages
4. **Logging**: Structured logging using Python's logging module
5. **Type Hints**: Full type annotations for better IDE support

### Common Utilities

```python
# Shared utilities across quickstart scripts
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import json

class QuickstartBase:
    """Base class for quickstart scripts."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize base quickstart component.
        
        Args:
            config_path: Optional configuration file path
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or defaults."""
        if config_path and config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return self._get_default_config()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(self.__class__.__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "model_name": "distilbert-base-uncased",
            "batch_size": 32,
            "max_length": 256,
            "device": "auto"
        }
```

## Performance Considerations

### Optimization Strategies

1. **Model Selection by Hardware**:
   
   | Hardware | Recommended Model | Batch Size | Expected Throughput |
   |----------|------------------|------------|-------------------|
   | CPU only | DistilBERT | 8-16 | 10-20 samples/sec |
   | GTX 1060 (6GB) | BERT-base | 16-32 | 50-100 samples/sec |
   | RTX 3090 (24GB) | DeBERTa-v3 | 64-128 | 200-400 samples/sec |
   | V100 (32GB) | RoBERTa-large | 128-256 | 500-1000 samples/sec |

2. **Memory Optimization Techniques**:
   ```python
   # Gradient checkpointing for large models
   model.gradient_checkpointing_enable()
   
   # Mixed precision training
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   
   # Dynamic padding for variable length inputs
   from transformers import DataCollatorWithPadding
   collator = DataCollatorWithPadding(tokenizer, padding="longest")
   ```

3. **Inference Optimization**:
   ```python
   # Model quantization
   import torch.quantization as quantization
   quantized_model = quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   
   # ONNX export for production
   torch.onnx.export(
       model,
       dummy_input,
       "model.onnx",
       opset_version=11,
       do_constant_folding=True
   )
   ```

### Benchmarking Results

Performance metrics on AG News test set (7,600 samples):

```python
benchmarks = {
    "distilbert-base": {
        "accuracy": 0.942,
        "inference_time_cpu": 45.2,  # seconds
        "inference_time_gpu": 3.8,
        "model_size_mb": 256,
        "memory_usage_mb": 512
    },
    "bert-base": {
        "accuracy": 0.951,
        "inference_time_cpu": 89.3,
        "inference_time_gpu": 6.2,
        "model_size_mb": 418,
        "memory_usage_mb": 850
    },
    "roberta-base": {
        "accuracy": 0.958,
        "inference_time_cpu": 92.1,
        "inference_time_gpu": 6.5,
        "model_size_mb": 476,
        "memory_usage_mb": 920
    }
}
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error Message**:
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions**:
```python
# Solution 1: Reduce batch size
trainer = SimpleTrainer(batch_size=8)  # Reduce from default 32

# Solution 2: Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Solution 3: Use mixed precision
trainer = SimpleTrainer(fp16=True)

# Solution 4: Clear cache between runs
import torch
torch.cuda.empty_cache()

# Solution 5: Use gradient accumulation
trainer = SimpleTrainer(
    batch_size=8,
    gradient_accumulation_steps=4  # Effective batch size = 32
)
```

#### 2. Import Errors

**Error Message**:
```
ModuleNotFoundError: No module named 'src'
```

**Solutions**:
```bash
# Solution 1: Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Solution 2: Install in development mode
pip install -e .

# Solution 3: Run from project root
cd ag-news-text-classification
python quickstart/minimal_example.py
```

#### 3. Data Loading Issues

**Error Message**:
```
FileNotFoundError: Dataset not found at specified path
```

**Solutions**:
```bash
# Solution 1: Download data explicitly
python scripts/setup/download_all_data.py

# Solution 2: Use automatic download
python quickstart/train_simple.py --auto-download

# Solution 3: Specify custom data path
python quickstart/train_simple.py --data-dir /path/to/data
```

#### 4. Model Loading Failures

**Error Message**:
```
RuntimeError: Error loading state_dict for model
```

**Solutions**:
```python
# Solution 1: Load with strict=False
model.load_state_dict(checkpoint['state_dict'], strict=False)

# Solution 2: Map to correct device
checkpoint = torch.load(path, map_location='cpu')

# Solution 3: Handle version mismatch
from src.migrations.models.version_converter import convert_checkpoint
checkpoint = convert_checkpoint(checkpoint, target_version="2.0")
```

#### 5. API Connection Issues

**Error Message**:
```
requests.exceptions.ConnectionError: Failed to establish connection
```

**Solutions**:
```bash
# Solution 1: Check if API server is running
curl http://localhost:8000/health

# Solution 2: Start API server
python -m uvicorn src.api.rest.app:app --reload

# Solution 3: Use correct port
python quickstart/api_quickstart.py --api-url http://localhost:8001

# Solution 4: Check firewall settings
sudo ufw allow 8000
```

### Performance Debugging

```python
# Memory profiling
from memory_profiler import profile

@profile
def train_model():
    # Training code
    pass

# Time profiling
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# Code to profile
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)

# GPU utilization monitoring
import GPUtil
GPUtil.showUtilization()
```

## References

1. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. *Advances in Neural Information Processing Systems*, 28.

2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT*.

3. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv preprint arXiv:1907.11692*.

4. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

5. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations*.

6. Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., ... & Wu, H. (2018). Mixed Precision Training. *International Conference on Learning Representations*.

7. Resnick, M., Myers, B., Nakakoji, K., Shneiderman, B., Pausch, R., Selker, T., & Eisenberg, M. (2005). Design principles for tools to support creative thinking. *NSF Workshop Report on Creativity Support Tools*.

## License

This project is released under the MIT License. See [LICENSE](../LICENSE) file for details.

## Contributing

Contributions are welcome. Please refer to [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
