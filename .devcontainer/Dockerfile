# AG News Text Classification - Development Container
# ====================================================
# Comprehensive development environment with GPU support
# 
# Based on best practices from:
# - NVIDIA Deep Learning Container Documentation
# - Python Docker Best Practices (https://pythonspeed.com/docker/)
# - Multi-stage Docker Builds for ML (https://mlinproduction.com/docker-for-ml/)
#
# Author: Võ Hải Dũng
# License: MIT

# Build arguments
ARG CUDA_VERSION=11.8.0
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10

# Base image with CUDA support
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${UBUNTU_VERSION} AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set CUDA architecture list for PyTorch
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    gcc \
    gfortran \
    make \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libatlas-base-dev \
    gfortran \
    curl \
    wget \
    git \
    vim \
    nano \
    htop \
    tmux \
    screen \
    rsync \
    unzip \
    zip \
    p7zip-full \
    net-tools \
    iputils-ping \
    dnsutils \
    openssh-client \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    sudo \
    locales \
    tzdata \
    ffmpeg \
    libsndfile1 \
    sox \
    poppler-utils \
    tesseract-ocr \
    postgresql-client \
    mysql-client \
    sqlite3 \
    redis-tools \
    sysstat \
    iotop \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
ARG PYTHON_VERSION
RUN conda create -n ag-news python=${PYTHON_VERSION} -y && \
    conda init bash && \
    echo "conda activate ag-news" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate ag-news

# Install PyTorch
RUN conda activate ag-news && \
    conda install -c pytorch -c nvidia \
    pytorch torchvision torchaudio pytorch-cuda=11.8 \
    -y

# Install core packages
RUN conda activate ag-news && pip install --upgrade pip && \
    pip install \
    transformers==4.36.0 \
    datasets==2.16.0 \
    tokenizers==0.15.0 \
    accelerate==0.25.0 \
    peft==0.7.0 \
    bitsandbytes==0.41.0 \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    lightgbm==4.2.0 \
    catboost==1.2.2 \
    spacy==3.7.2 \
    nltk==3.8.1 \
    gensim==4.3.2 \
    sentence-transformers==2.2.2 \
    optuna==3.5.0 \
    ray[tune]==2.9.0 \
    hyperopt==0.2.7 \
    wandb==0.16.1 \
    mlflow==2.9.2 \
    tensorboard==2.15.1 \
    pandas==2.1.4 \
    numpy==1.24.3 \
    scipy==1.11.4 \
    dask==2023.12.1 \
    polars==0.20.2 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    plotly==5.18.0 \
    fastapi==0.108.0 \
    uvicorn==0.25.0 \
    streamlit==1.29.0 \
    gradio==4.11.0 \
    sqlalchemy==2.0.23 \
    alembic==1.13.1 \
    pytest==7.4.3 \
    pytest-cov==4.1.0 \
    pytest-xdist==3.5.0 \
    hypothesis==6.92.1 \
    black==23.12.1 \
    isort==5.13.2 \
    flake8==7.0.0 \
    mypy==1.8.0 \
    pylint==3.0.3 \
    ruff==0.1.9 \
    sphinx==7.2.6 \
    mkdocs==1.5.3 \
    tqdm==4.66.1 \
    rich==13.7.0 \
    typer==0.9.0 \
    click==8.1.7 \
    pydantic==2.5.3 \
    python-dotenv==1.0.0 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2

# spaCy models
RUN conda activate ag-news && \
    python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_md && \
    python -m spacy download en_core_web_lg

# NLTK data
RUN conda activate ag-news && \
    python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('stopwords'); \
    nltk.download('wordnet'); \
    nltk.download('averaged_perceptron_tagger'); \
    nltk.download('maxent_ne_chunker'); \
    nltk.download('words')"

# Jupyter
RUN conda activate ag-news && \
    pip install \
    jupyter \
    jupyterlab==4.0.9 \
    notebook \
    ipywidgets \
    jupyter-contrib-nbextensions \
    jupyterlab-git \
    jupyterlab-lsp \
    jupyterlab-code-formatter

# Create non-root user
ARG USERNAME=researcher
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=KATEX_INLINE_OPENrootKATEX_INLINE_CLOSE NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME

# Install Oh My Zsh
USER $USERNAME
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/themes/powerlevel10k

RUN echo 'export ZSH="$HOME/.oh-my-zsh"' >> ~/.zshrc && \
    echo 'ZSH_THEME="powerlevel10k/powerlevel10k"' >> ~/.zshrc && \
    echo 'plugins=(git docker python pip conda jupyter vscode zsh-autosuggestions zsh-syntax-highlighting)' >> ~/.zshrc && \
    echo 'source $ZSH/oh-my-zsh.sh' >> ~/.zshrc && \
    echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.zshrc && \
    echo 'conda activate ag-news' >> ~/.zshrc

USER root

RUN mkdir -p /workspace && \
    chown -R $USERNAME:$USERNAME /workspace

RUN mkdir -p /workspace/.cache/huggingface && \
    mkdir -p /workspace/.cache/torch && \
    mkdir -p /workspace/.cache/wandb && \
    mkdir -p /workspace/.cache/pip && \
    chown -R $USERNAME:$USERNAME /workspace/.cache

WORKDIR /workspace

COPY .devcontainer/post-*.sh /tmp/
RUN chmod +x /tmp/post-*.sh && \
    chown $USERNAME:$USERNAME /tmp/post-*.sh

RUN echo '#!/bin/bash\n\
set -e\n\
echo "Setting up AG News development environment..."\n\
\n\
if [ -f "/workspace/setup.py" ]; then\n\
    echo "Installing project in editable mode..."\n\
    pip install -e /workspace\n\
fi\n\
\n\
if [ -f "/workspace/requirements/dev.txt" ]; then\n\
    echo "Installing development requirements..."\n\
    pip install -r /workspace/requirements/dev.txt\n\
fi\n\
\n\
if [ -f "/workspace/scripts/setup/download_all_data.py" ]; then\n\
    echo "Downloading AG News dataset..."\n\
    python /workspace/scripts/setup/download_all_data.py || true\n\
fi\n\
\n\
git config --global --add safe.directory /workspace\n\
\n\
echo "Development environment ready!"\n\
echo "Run '\''make help'\'' to see available commands"\n\
' > /tmp/post-create.sh

RUN echo '#!/bin/bash\n\
echo "Container started successfully!"\n\
echo "GPU Status:"\n\
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || echo "No GPU detected"\n\
echo ""\n\
echo "Python: $(python --version)"\n\
echo "PyTorch: $(python -c \"import torch; print(torch.__version__)\" 2>/dev/null || echo \"Not installed\")"\n\
echo "Transformers: $(python -c \"import transformers; print(transformers.__version__)\" 2>/dev/null || echo \"Not installed\")"\n\
echo ""\n\
echo "Workspace: /workspace"\n\
echo "Ports forwarded: 8501 (Streamlit), 8888 (Jupyter), 6006 (TensorBoard)"\n\
' > /tmp/post-start.sh

RUN echo '#!/bin/bash\n\
echo "Welcome to AG News Classification Development Environment!"\n\
echo ""\n\
echo "Quick commands:"\n\
echo "  make train    - Start training"\n\
echo "  make test     - Run tests"\n\
echo "  make app      - Launch Streamlit app"\n\
echo "  make notebook - Start Jupyter Lab"\n\
echo ""\n\
' > /tmp/post-attach.sh

RUN chmod +x /tmp/post-*.sh

USER $USERNAME

EXPOSE 8501 8888 6006 5000 3000

ENTRYPOINT ["/bin/zsh"]
