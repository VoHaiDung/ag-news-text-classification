# AG News Text Classification - Quick Start Guide

This directory contains simplified examples and quick start scripts for the AG News Text Classification project, designed for rapid prototyping and educational purposes.

**Author**: Võ Hải Dũng  
**Email**: vohaidung.work@gmail.com  
**Date**: 2025

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start Scripts](#quick-start-scripts)
- [Usage Examples](#usage-examples)
- [Docker Deployment](#docker-deployment)
- [Google Colab](#google-colab)
- [API Usage](#api-usage)
- [Troubleshooting](#troubleshooting)

## Overview

The quickstart module provides simplified interfaces to the main functionality of the AG News classification system, abstracting complex configurations while maintaining educational value.

### Design Philosophy

Following the principle of progressive disclosure from educational computing:
- **Minimal Example**: Bare minimum code for classification
- **Simple Training**: Basic training pipeline with essential features
- **Evaluation**: Straightforward model evaluation
- **Demo Application**: Interactive web interface
- **API Quickstart**: RESTful API usage examples

## Prerequisites

### System Requirements

```bash
# Python version
Python >= 3.8

# CUDA (optional, for GPU acceleration)
CUDA >= 11.0 (if using GPU)

# Memory requirements
RAM >= 8GB (16GB recommended)
GPU Memory >= 4GB (for transformer models)
```

### Installation

```bash
# Clone repository
git clone https://github.com/VoHaiDung/ag-news-text-classification.git
cd ag-news-text-classification

# Install minimal dependencies
pip install -r requirements/minimal.txt

# For full functionality
pip install -r requirements/all.txt
```

## Quick Start Scripts

### 1. Minimal Example (`minimal_example.py`)

Simplest possible classification example:

```bash
# Basic usage
python quickstart/minimal_example.py --text "Apple releases new iPhone with advanced features"

# With custom model
python quickstart/minimal_example.py \
    --text "Your news text here" \
    --model-path ./outputs/models/best_model
```

**Features:**
- Single text classification
- Pre-trained model loading
- Minimal dependencies
- Educational comments

### 2. Simple Training (`train_simple.py`)

Streamlined training pipeline:

```bash
# Basic training with default settings
python quickstart/train_simple.py

# Custom configuration
python quickstart/train_simple.py \
    --model-name distilbert-base-uncased \
    --num-epochs 5 \
    --batch-size 32 \
    --learning-rate 2e-5
```

**Key Parameters:**
- `--model-name`: Hugging Face model identifier
- `--num-epochs`: Training iterations
- `--batch-size`: Samples per batch
- `--learning-rate`: Optimization step size
- `--fp16`: Enable mixed precision training

### 3. Model Evaluation (`evaluate_simple.py`)

Quick evaluation on test data:

```bash
# Evaluate saved model
python quickstart/evaluate_simple.py \
    --model-path ./outputs/simple/20240101_120000

# With detailed metrics
python quickstart/evaluate_simple.py \
    --model-path ./outputs/simple/20240101_120000 \
    --detailed \
    --save-predictions
```

**Output Metrics:**
- Accuracy
- F1-Score (macro/micro)
- Confusion matrix
- Per-class performance

### 4. Demo Application (`demo_app.py`)

Interactive Streamlit application:

```bash
# Launch web interface
streamlit run quickstart/demo_app.py

# Custom port
streamlit run quickstart/demo_app.py --server.port 8080
```

**Features:**
- Web-based interface
- Real-time predictions
- Batch processing
- Confidence visualization

### 5. API Quickstart (`api_quickstart.py`)

RESTful API client examples:

```bash
# Run API examples
python quickstart/api_quickstart.py

# With custom endpoint
python quickstart/api_quickstart.py --api-url http://localhost:8000
```

## Usage Examples

### Example 1: Complete Pipeline

```python
# Train model
python quickstart/train_simple.py --num-epochs 3

# Evaluate performance
python quickstart/evaluate_simple.py --model-path ./outputs/simple/latest

# Test single prediction
python quickstart/minimal_example.py --text "Technology news article"

# Launch demo
streamlit run quickstart/demo_app.py
```

### Example 2: Custom Dataset

```python
# Prepare custom data
python scripts/data_preparation/prepare_ag_news.py --custom-data ./my_data.csv

# Train on custom data
python quickstart/train_simple.py --data-dir ./data/custom

# Evaluate
python quickstart/evaluate_simple.py --test-data ./data/custom/test.csv
```

## Docker Deployment

### Quick Docker Setup

```bash
# Build image
docker build -f quickstart/docker_quickstart/Dockerfile -t agnews-quick .

# Run container
docker run -p 8501:8501 agnews-quick

# With GPU support
docker run --gpus all -p 8501:8501 agnews-quick
```

### Docker Compose

```bash
# Start all services
docker-compose -f quickstart/docker_quickstart/docker-compose.yml up

# Run in background
docker-compose -f quickstart/docker_quickstart/docker-compose.yml up -d
```

## Google Colab

### Quick Start in Colab

1. Open `quickstart/colab_notebook.ipynb` in Google Colab
2. Run all cells sequentially
3. Follow interactive instructions

### Direct Link

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VoHaiDung/ag-news-text-classification/blob/main/quickstart/colab_notebook.ipynb)

## API Usage

### Basic API Calls

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Your news article here"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"texts": ["Article 1", "Article 2"]}
)
```

### Authentication (if enabled)

```python
# With API key
headers = {"X-API-Key": "your-api-key"}
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "News article"},
    headers=headers
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python quickstart/train_simple.py --batch-size 16
   
   # Use CPU
   python quickstart/train_simple.py --device cpu
   ```

2. **Import Errors**
   ```bash
   # Ensure project root in path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Data Not Found**
   ```bash
   # Download data first
   python scripts/setup/download_all_data.py
   ```

### Performance Tips

- Use GPU for 10x speedup
- Enable mixed precision (`--fp16`) for memory efficiency
- Start with DistilBERT for faster training
- Use smaller batch sizes for limited memory

## Advanced Usage

For more advanced features, refer to:
- [Full Documentation](../docs/index.md)
- [API Reference](../docs/api_reference/rest_api.md)
- [Model Zoo](../docs/user_guide/model_training.md)
- [Deployment Guide](../docs/user_guide/deployment.md)

## Citation

If you use this quickstart in your research, please cite:

```bibtex
@software{agnews_quickstart_2025,
  title = {AG News Text Classification Quick Start},
  author = {Võ Hải Dũng},
  year = {2025},
  url = {https://github.com/VoHaiDung/ag-news-text-classification}
}
```

## License

MIT License - See [LICENSE](../LICENSE) for details.

## Support

- **GitHub Issues**: [Create Issue](https://github.com/VoHaiDung/ag-news-text-classification/issues)
- **Email**: vohaidung.work@gmail.com
- **Documentation**: [Read Docs](../docs/index.md)
- **Examples**: [More Examples](../notebooks/tutorials/)

## Acknowledgments

This project builds upon research and methodologies from:
- Zhang et al. (2015): "Character-level Convolutional Networks for Text Classification"
- Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"
- Richardson & Ruby (2013): "RESTful Web APIs"

---

**Developed by**: Võ Hải Dũng
**Date**: 2025
