# Exporting a Container Image from Inside a Running Container

When you're inside a running container on a managed platform (vast.ai, StrongCompute, cloud notebooks, etc.) and need to export it as a reusable Docker image, but don't have access to the Docker daemon.

## The Problem

Managed container platforms often:
- Don't expose the Docker socket (`/var/run/docker.sock`)
- Don't provide `docker`, `podman`, or `buildah` CLI tools
- Only allow internal cloning/snapshotting, not external export

## The Solution: Filesystem Tarball + Docker Import

Docker can create images from filesystem tarballs using `docker import`. The process:

1. **Inside the container**: Create a tarball of the filesystem
2. **Transfer**: Copy the tarball to a machine with Docker
3. **Import**: Convert tarball to Docker image
4. **Push**: Upload to Docker Hub or other registry

### Step 1: Create Filesystem Tarball (Inside Container)

```bash
# Full filesystem export
tar -cvf /workspace/container-export.tar \
  --exclude=/proc \
  --exclude=/sys \
  --exclude=/dev \
  --exclude=/run \
  --exclude=/tmp \
  --exclude=/workspace/container-export.tar \
  /

# Or, more selective (just the important directories)
tar -cvf /workspace/container-export.tar \
  --exclude=/proc \
  --exclude=/sys \
  --exclude=/dev \
  --exclude=/run \
  /usr /etc /opt /root /home /var/lib
```

### Step 2: Transfer the Tarball

```bash
# From your local machine
scp user@container-host:/workspace/container-export.tar .

# Or use rsync for better resume capability
rsync -avP user@container-host:/workspace/container-export.tar .
```

### Step 3: Import as Docker Image

```bash
# Basic import
docker import container-export.tar myimage:tag

# Import with metadata (CMD, ENV, etc.)
docker import \
  --change 'CMD ["/bin/bash"]' \
  --change 'WORKDIR /workspace' \
  --change 'ENV HF_HOME=/root/.cache/huggingface' \
  container-export.tar myimage:tag
```

### Step 4: Push to Registry

```bash
docker tag myimage:tag username/myimage:tag
docker push username/myimage:tag
```

---

## Pros

| Advantage | Description |
|-----------|-------------|
| **Works anywhere** | No Docker daemon or special tools needed inside container |
| **Captures runtime state** | Includes packages installed after container started |
| **Simple tooling** | Only needs `tar` (universally available) |
| **Platform agnostic** | Works on any managed container platform |
| **Full control** | Can exclude sensitive files, caches, etc. |

## Cons

| Disadvantage | Description |
|--------------|-------------|
| **Single layer** | Resulting image has one layer (no caching benefits) |
| **Large size** | Full filesystem can be 10-30GB+ |
| **Loses metadata** | Original Dockerfile instructions (ENTRYPOINT, ENV, EXPOSE, etc.) not preserved |
| **Transfer time** | Large tarballs take time to scp/rsync |
| **No build reproducibility** | Can't rebuild from source; it's a snapshot |
| **Potential cruft** | May include unnecessary files, caches, logs |

---

## Restoring Metadata

After importing, you can add back metadata with a thin Dockerfile:

```dockerfile
FROM myimage:imported

# Restore metadata that was lost
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers
WORKDIR /workspace
EXPOSE 8888
CMD ["/bin/bash"]
```

Or use `docker commit` with `--change` flags:

```bash
docker run -d myimage:imported sleep infinity
docker commit \
  --change 'ENV HF_HOME=/root/.cache/huggingface' \
  --change 'WORKDIR /workspace' \
  --change 'CMD ["/bin/bash"]' \
  <container_id> myimage:final
```

---

## Optimizing Export Size

### Exclude Unnecessary Directories

```bash
tar -cvf /workspace/export.tar \
  --exclude=/proc \
  --exclude=/sys \
  --exclude=/dev \
  --exclude=/run \
  --exclude=/tmp \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.cache/pip' \
  --exclude='.cache/huggingface/hub' \
  --exclude='/var/cache' \
  --exclude='/var/log' \
  /
```

### Export Only What Changed

If you know the base image, export just the modified directories:

```bash
# For a Python ML container, the important stuff is usually:
tar -cvf /workspace/ml-packages.tar \
  /usr/local/lib/python3.*/dist-packages \
  /usr/local/bin \
  /root/.cache/huggingface \
  /workspace
```

Then layer it on top of a fresh base image.

---

## Alternative: Selective Package Export

Instead of exporting the whole filesystem, export just the installed packages:

```bash
# Python packages
pip freeze > requirements.txt
tar -cvf python-packages.tar /usr/local/lib/python3.*/dist-packages

# Or for compiled extensions (like flash-attn)
tar -cvf compiled-packages.tar \
  /usr/local/lib/python3.*/dist-packages/flash_attn* \
  /usr/local/lib/python3.*/dist-packages/*.so
```

This is what we did for flash-attn: extracted just the compiled package, hosted it on GitHub releases, and `COPY` it into a clean Dockerfile build.

---

## When to Use Each Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| One-time snapshot for backup | Full filesystem tarball |
| Reproducible builds | Dockerfile + GitHub Actions |
| Compiled packages (flash-attn, etc.) | Selective package tarball |
| Quick iteration/testing | Platform's internal clone/snapshot |
| Production deployment | Clean Dockerfile build |

---

## Platform-Specific Notes

### vast.ai
- Containers are Docker-based
- Can sometimes access Docker socket (instance-dependent)
- Supports `onstart` scripts for reproducible setup

### StrongCompute
- Uses `isc container stop --squash` for internal snapshots
- No documented export to external registries
- Clone feature is internal only

### Google Colab / Kaggle
- Very restricted environment
- Best to export just data/models, not the container

### Lambda Labs / RunPod
- Often provide Docker access
- May be able to `docker commit` directly
