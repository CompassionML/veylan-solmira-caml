# Creating Custom Container Images

Two approaches for customizing your StrongCompute environment.

**Recommended:** Use the CaML Dockerfile at `docker/Dockerfile` for a modern Python 3.12 + CUDA 12.4 environment.

---

## Option 1: Modify Current Container (Easiest)

Best for incremental changes to an existing image.

### Steps

1. Start your container and SSH in
2. Make changes (install packages, upgrade Python, etc.)
3. Save changes when stopping:

```bash
isc container stop --squash
```

Or restart with changes saved:
```bash
isc container restart --squash
```

### Flags

| Flag | Description |
|------|-------------|
| `--save` / `--no-save` | Whether to save container changes (default: save) |
| `--squash` / `-s` | Reduce image size by flattening layers (recommended) |

### Limitations

- Still building on existing base (can't change OS/CUDA version easily)
- Layer accumulation can bloat image over time (use `--squash`)

---

## Option 2: Import from DockerHub (Fresh Start)

Best for full control over the environment.

### Steps

1. **Create Dockerfile locally**

```dockerfile
# Example: Modern PyTorch base
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    datasets \
    safetensors

# Optional: copy requirements file
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
```

2. **Build and push to DockerHub**

```bash
# Build
docker build -t yourusername/caml-env:latest .

# Login to DockerHub
docker login

# Push
docker push yourusername/caml-env:latest
```

3. **Import via Control Plane**

- Go to https://cp.strongcompute.ai
- Navigate to your organization's images section
- Use import option to pull from DockerHub

### Recommended Base Images

| Base Image | Python | CUDA | Notes |
|------------|--------|------|-------|
| `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` | 3.11 | 12.4 | Good default |
| `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` | 3.11 | 12.4 | Includes nvcc for compiling |
| `nvidia/cuda:12.4.0-runtime-ubuntu24.04` | None | 12.4 | Minimal, install Python yourself |
| `huggingface/transformers-pytorch-gpu` | 3.10 | varies | HF-optimized |

### Example: Python 3.12 Environment

```dockerfile
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install packages
RUN pip install --no-cache-dir \
    torch \
    transformers \
    accelerate \
    datasets
```

---

---

## CaML Research Image

A ready-to-use Dockerfile is available at `docker/Dockerfile`:

**Stack:**
- Python 3.12
- CUDA 12.4 + cuDNN
- PyTorch 2.5.1
- transformers, accelerate, transformer-lens
- scikit-learn, pandas, matplotlib
- JupyterLab

**Build and push:**
```bash
cd infrastructure/docker

# Build locally
./build.sh

# Build and push to DockerHub
DOCKER_IMAGE=yourusername/caml-env ./build.sh --push --tag v1
```

**Files:**
- `docker/Dockerfile` - Main image definition
- `docker/requirements.txt` - Python dependencies
- `docker/build.sh` - Build helper script

---

## Tips

- Ask in Discord `#isc-help` what base images others in CaML org are using
- Use `--squash` when stopping to keep image size manageable
- Test locally with `docker run` before pushing to DockerHub
- Include ISC CLI setup if building from scratch (check existing container for reference)
