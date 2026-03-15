# CaML Research Environment
# Target: Llama 3.1 8B/70B linear probes and activation analysis
#
# Based on caml-base which includes:
#   - CUDA 12.8, Python 3.12, PyTorch 2.10
#   - Flash Attention 2.8.3 (pre-compiled)

FROM veylansolmira/caml-base:latest

LABEL maintainer="CaML Research"
LABEL description="Complete environment for compassion measurement in LLMs"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Additional system tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tmux \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

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

# SSH setup for StrongCompute
RUN mkdir -p /var/run/sshd \
    && echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config \
    && echo 'PasswordAuthentication no' >> /etc/ssh/sshd_config

# Create workspace
RUN mkdir -p /workspace
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
