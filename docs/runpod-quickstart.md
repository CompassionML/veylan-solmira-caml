# RunPod Quickstart

## GPUs

**Most work (LoRA/QLoRA up to 13B)** — A6000 (`--gpu value`, $0.33/hr, 48GB VRAM). Best bang for buck.

**Faster iteration** — L40S (`--gpu recommended`, $0.79/hr, 48GB). Same VRAM, ~2x throughput.

**Large models / full fine-tune** — A100 80GB (`--gpu powerful`, $1.19/hr). Only when you need >48GB VRAM.

## Setup (once)

```bash
# 1. Generate SSH key
ssh-keygen -t ed25519 -f ~/.ssh/runpod_ed25519

# 2. Create .env OUTSIDE the repo (e.g. parent of caml-research/)
cat > ../path/to/.env << 'EOF'
RUNPOD_API_KEY=rpa_YOUR_KEY_HERE
RUNPOD_SSH_PUBLIC_KEY=ssh-ed25519 AAAA... user@host
EOF

# 3. Install deps
uv pip install -r requirements.txt
```

Get your public key with `cat ~/.ssh/runpod_ed25519.pub`. Get an API key from RunPod Settings > API Keys (use a scoped `rpa_` key).

## Launch a pod

```bash
python scripts/runpod_launch.py launch \
  --image veylansolmira/caml-env:latest \
  --ssh-key ~/.ssh/runpod_ed25519 \
  --network-volume YOUR_VOLUME_ID
```

The script shows cost and asks to confirm. It waits for the pod and prints the SSH command.

Find your volume ID with: `python scripts/runpod_launch.py volumes`

## Storage

**Use a network volume** (`--network-volume`). Pod volumes are deleted on terminate.

We have shared network volumes — use `volumes` to find the right one. Store scripts and adapters on the volume. Push large checkpoints and full models to HuggingFace.

**50GB volume** — fine for single-model QLoRA (one 7B model + adapters + dataset). Tight if you cache multiple models. Disk-full mid-training = wasted GPU money.

**100-200GB volume** — comfortable for multiple models, no cleanup pressure. Extra $7-10/mo but saves you from crashed runs.

See [full guide](runpod-guide.md#storage-sizing-50gb-vs-200gb-deep-dive) for the detailed cost math.

## Connect Claude Code

After the pod starts, copy the **SSH over exposed TCP** command (the second SSH link, direct IP + port). Give it to Claude Code and it can work on the pod remotely.

```
ssh root@216.81.151.3 -p 12411 -i ~/.ssh/runpod_ed25519
```

## When done

```bash
# Stop (keeps volume data, stops GPU billing):
python scripts/runpod_launch.py stop --pod-id POD_ID

# Terminate (DELETES pod volume data permanently):
python scripts/runpod_launch.py terminate --pod-id POD_ID
```

**Always stop when not using the GPU.** Don't leave pods running overnight.

## Other commands

```bash
python scripts/runpod_launch.py balance     # Check credit
python scripts/runpod_launch.py gpus        # See all GPUs + pricing
python scripts/runpod_launch.py volumes     # List network volumes
python scripts/runpod_launch.py list --ssh-key ~/.ssh/runpod_ed25519  # All pods + SSH
```

## Costs at a glance

| GPU | $/hr | Typical QLoRA run (3hr) | Full day |
|-----|------|------------------------|----------|
| A6000 (value) | $0.33 | ~$1 | $8 |
| L40S (recommended) | $0.79 | ~$2.40 | $19 |
| A100 80GB (powerful) | $1.19 | ~$3.60 | $29 |

No hard spending cap — RunPod is credit-based. Auto-stops near $0 balance.

---

Full details: [runpod-guide.md](runpod-guide.md)
