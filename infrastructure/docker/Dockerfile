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

# Fine-tuning stack (Unsloth + LoRA + DPO/SFT trainers)
# Required by experiments/cai-1b-replication/train_universal.py and similar.
# Unsloth transitively installs peft, trl, bitsandbytes, and pins compatible
# transformers/accelerate/torch versions.
RUN pip install --no-cache-dir \
    unsloth \
    wandb

# Evaluation stack (Inspect AI for AHB / MORU; google-genai for Gemini grader)
# Required by AHB eval workflow. inspect-ai brings tiktoken/etc. transitively.
RUN pip install --no-cache-dir \
    inspect-ai \
    inspect_evals \
    google-genai

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
