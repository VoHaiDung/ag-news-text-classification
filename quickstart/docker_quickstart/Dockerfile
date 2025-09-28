# AG News Classification - Quick Start Docker Image
# ==================================================
#
# Lightweight Docker image for quick start and demos
# Based on best practices from:
# - Docker (2023): "Best practices for writing Dockerfiles"
# - Gerner (2020): "Docker for Data Science"
#
# Build: docker build -f quickstart/docker_quickstart/Dockerfile -t agnews-quick .
# Run: docker run -p 8501:8501 agnews-quick

# Multi-stage build for efficiency
# Stage 1: Builder
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/minimal.txt requirements-minimal.txt
COPY requirements/ui.txt requirements-ui.txt

# Create wheels
RUN pip install --no-cache-dir wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels \
    -r requirements-minimal.txt \
    -r requirements-ui.txt

# Stage 2: Runtime
FROM python:3.9-slim

# Metadata
LABEL maintainer="AG News Project Team" \
      version="1.0.0" \
      description="AG News Classification Quick Start"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /app/data /app/models /app/outputs && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels

# Install Python packages
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* && \
    rm -rf /wheels

# Copy application code
COPY --chown=appuser:appuser . /app/

# Download minimal model (DistilBERT for quick start)
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
    AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)" && \
    chown -R appuser:appuser /root/.cache && \
    mv /root/.cache /app/.cache && \
    chown -R appuser:appuser /app/.cache

# Switch to non-root user
USER appuser

# Set environment for model cache
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command - run demo app
CMD ["streamlit", "run", "quickstart/demo_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.serverAddress=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
