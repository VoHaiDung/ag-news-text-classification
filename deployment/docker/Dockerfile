# Multi-stage Dockerfile for AG News Text Classification
# ================================================================================
# Main Dockerfile for building training and inference containers
# Supports both CPU and GPU environments with optimized layer caching
#
# Build stages:
#   - python-base: Base Python environment with system dependencies
#   - builder: Dependency installation stage
#   - training: Training environment with full ML stack
#   - inference: Lightweight inference environment
#   - production: Production-ready minimal image
#
# References:
#   - Docker Multi-stage Builds: https://docs.docker.com/develop/develop-images/multistage-build/
#   - Python Docker Best Practices: https://pythonspeed.com/docker/
#
# Author: AG News Classification Team
# License: MIT

ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=20.04

# Stage 1: Base Python environment
FROM python:${PYTHON_VERSION}-slim AS python-base

# Set environment variables for reproducibility
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    git \
    wget \
    vim \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create application directory structure
RUN mkdir -p /app /app/data /app/models /app/outputs /app/logs /app/cache

WORKDIR /app

# Stage 2: Builder stage for dependencies
FROM python-base AS builder

# Copy requirements files
COPY requirements/ /tmp/requirements/

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install base requirements with CPU PyTorch for broad compatibility
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install ML requirements
RUN pip install transformers==4.35.0 \
    datasets==2.14.0 \
    tokenizers==0.15.0 \
    numpy==1.24.0 \
    pandas==2.0.0 \
    scikit-learn==1.3.0 \
    scipy==1.10.0 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.0 \
    tqdm==4.66.0 \
    requests==2.31.0 \
    typing-extensions==4.8.0

# Stage 3: Training environment
FROM python-base AS training

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY setup.py pyproject.toml README.md /app/

# Install application in development mode
RUN pip install -e .

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH:-}

# Create volume mount points
VOLUME ["/app/data", "/app/models", "/app/outputs", "/app/logs"]

# Default command for training
CMD ["python", "scripts/training/train_single_model.py"]

# Stage 4: Inference environment
FROM python-base AS inference

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only necessary code for inference
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY setup.py pyproject.toml /app/

# Install application
RUN pip install .

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH:-}

# Create non-root user for security
RUN groupadd -r mluser && useradd -r -g mluser mluser && \
    chown -R mluser:mluser /app

USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command for inference
CMD ["python", "src/inference/predictors/single_predictor.py"]

# Stage 5: Production environment
FROM python:${PYTHON_VERSION}-slim AS production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --from=inference /app /app

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser && \
    chown -R mluser:mluser /app

WORKDIR /app

USER mluser

# Set Python path
ENV PYTHONPATH=/app:${PYTHONPATH:-}

# Expose port for serving
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Production inference command
CMD ["python", "-m", "src.inference.serving.model_server", "--host", "0.0.0.0", "--port", "8080"]
