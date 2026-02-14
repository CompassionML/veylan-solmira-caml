# Container Environment Reference

**Container:** veylan-initial-2026-01-03
**Base Image:** NewestCaML
**Disk Storage:** 50GB
**Last verified:** 2026-02-14

---

## Storage Architecture

StrongCompute uses Ceph network storage for datasets/models (doesn't count against container disk):

| Mount | Type | Size | Purpose |
|-------|------|------|---------|
| `/` (overlay) | Local | 50GB | Container disk (OS, venv, your files) |
| `/data/` | Ceph (ro) | Shared | Pre-mounted datasets/models from Control Plane |
| `/shared/` | Ceph (rw) | 107GB+ | Shared team storage |

**Key insight:** Mounting large models (e.g., 70B) via Control Plane uses network storage, not your 50GB container disk. You can mount multiple large models freely.

**Adding datasets:** No hot-mount capability found. To add new datasets:
1. Stop container: `isc container stop --squash`
2. In Control Plane, edit container settings and add dataset
3. Start container again

(Confirm with Discord `#isc-help` if hot-mounting is possible)

Current disk usage (~21GB of 50GB):
- `/usr`: 9.5GB (system/packages)
- `/root`: 7.3GB (home, venv, cache)
- `/tmp`: 2.7GB
- `/opt`: 1.2GB

---

## Hardware

| Component | Details |
|-----------|---------|
| GPU | NVIDIA GeForce RTX 3090 Ti |
| VRAM | 24,564 MiB |
| Cluster | Sydney (VPN required) |

---

## Model Location

**Llama 3.1 8B** is pre-loaded at:
```
/data/uds-grave-seasoned-brownie-251009/
```

Model specs (from config.json):
- Architecture: LlamaForCausalLM
- Hidden size: 4096
- Layers: 32
- Attention heads: 32
- KV heads: 8 (GQA)
- Vocab size: 128,256
- Max context: 131,072 tokens
- Dtype: bfloat16

Usage in code:
```python
MODEL_PATH = "/data/uds-grave-seasoned-brownie-251009"
```

---

## Python Environment

Virtual environment at `~/.venv/`

Key packages (as of 2026-02-14):
| Package | Version |
|---------|---------|
| Python | 3.10.12 |
| torch | 2.7.0 |
| transformers | 4.57.2 |
| accelerate | 1.12.0 |
| torchao | 0.14.1 |
| torchvision | 0.22.0 |

Activate with:
```bash
source ~/.venv/bin/activate
```

---

## ISC CLI

Credentials stored at `~/credentials.isc`

Verify authentication:
```bash
isc ping
```

---

## Data Directories

| Path | Size | Contents |
|------|------|----------|
| `/data/uds-grave-seasoned-brownie-251009/` | 30GB | Llama 3.1 8B model |
| `/data/uds-*/` (others) | 0 | Empty experiment dirs |

---

## Connection

From local machine (with VPN active):
```bash
./strongcompute/scripts/connect.sh <hostname> <port>
```

Or via SSH config:
```bash
ssh strongcompute
```

SSH key: `/Users/infinitespire/Desktop/ai_dev/caml/secure/caml`
