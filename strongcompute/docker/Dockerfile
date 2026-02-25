# CaML Research Environment
# Target: Llama 3.1 8B/70B linear probes and activation analysis
# Python 3.12, CUDA 12.8, PyTorch 2.10

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

LABEL maintainer="CaML Research"
LABEL description="Modern environment for compassion measurement in LLMs"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    vim \
    htop \
    tmux \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.12
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Upgrade pip (no venv needed - Docker is already isolated)
RUN pip install --upgrade pip setuptools wheel

# PyTorch with CUDA 12.8
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Core ML stack
RUN pip install --no-cache-dir \
    transformers>=4.45.0 \
    accelerate>=1.0.0 \
    safetensors \
    datasets \
    tokenizers \
    sentencepiece \
    anthropic

# Interpretability tools
RUN pip install --no-cache-dir \
    transformer-lens \
    einops \
    fancy-einsum \
    jaxtyping

# Linear probes / analysis
RUN pip install --no-cache-dir \
    scikit-learn \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    plotly

# Jupyter for exploration
RUN pip install --no-cache-dir \
    jupyterlab \
    ipywidgets \
    nbformat

# Development tools
RUN pip install --no-cache-dir \
    black \
    ruff \
    pytest \
    tqdm \
    rich \
    typer

# HuggingFace CLI (for model downloads)
RUN pip install --no-cache-dir huggingface_hub[cli]

# Flash Attention 2 (faster attention, essential for large models)
# Install ninja for faster compilation, set arch for RTX 3090/4090
RUN pip install --no-cache-dir ninja packaging
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# SSH setup for StrongCompute
RUN mkdir -p /var/run/sshd \
    && echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config \
    && echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config

# Create workspace
RUN mkdir -p /workspace
WORKDIR /workspace

# Environment variables for better defaults
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
ENV TORCH_HOME=/root/.cache/torch

# Default command
CMD ["/bin/bash"]
