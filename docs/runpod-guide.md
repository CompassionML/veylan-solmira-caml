# RunPod GPU Instance Guide

This is the comprehensive reference for launching and managing GPU instances on [RunPod](https://www.runpod.io/) using our `runpod_launch.py` script. **If you just want to get going, start with [runpod-quickstart.md](runpod-quickstart.md)** — it's one page. Come back here for GPU selection guidance, storage sizing, cost analysis, and team setup.

> **Why RunPod over other platforms?** RunPod has low per-second billing, custom Docker image support, persistent network volumes, and no workflow restrictions. Compared to platforms like StrongCompute (workflow-locked, slow container saves) or Lambda Labs (limited custom images), RunPod gives full control with minimal infrastructure overhead. Our script wraps the entire lifecycle — launch, connect, stop, terminate — into single commands.

### How RunPod addresses our team's needs

Based on pain points from StrongCompute and other platforms, here's how RunPod stacks up:

| Need | RunPod | Notes |
|------|--------|-------|
| **Predictable costs shown before launch** | **Yes** | Script shows estimated $/hr and $/day before asking for confirmation |
| **Proactive cost management** | Partial | Credit-based (prepay), low-balance alerts, auto-stop near $0. No hard spending cap per user — see [Cost Management](#7-cost-management). |
| **Easy checkpointing to local storage** | **Yes** | HF Trainer writes to `/workspace` on a standard filesystem. Network volumes persist across pod lifecycles. |
| **Fast container startup (no 10-min saves)** | **Yes** | Network volumes decouple data from containers. No container save step — data lives on the volume, containers are stateless. |
| **One-command launch with GPU attached** | **Yes** | `python scripts/runpod_launch.py launch --image ... --ssh-key ...` — one command. |
| **Billing shows GPU shapes, duration, cost** | **Yes** | RunPod dashboard shows pod history with GPU type, runtime, and cost breakdown. |
| **Pre-installed ML libraries (Unsloth, etc.)** | **Yes** | Our custom image `veylansolmira/caml-env:latest` ships with Unsloth, HF Transformers, PyTorch, etc. |
| **Clear error messages** | Mostly | API errors surface in the script. Dashboard has pod logs. Better than StrongCompute but not perfect — some GraphQL errors can be cryptic. |
| **Datasets via HuggingFace IDs** | **Yes** | Standard `load_dataset("org/name")` works. No proprietary dataset ID system. |
| **Secret management** | Basic | Environment variables via `.env` file (never committed). Not as polished as Colab secrets but functional. |
| **Export images to Docker** | Manual | Can `docker commit` inside a pod and push to a registry, but no one-click export. |
| **Admin can stop others' pods** | **Yes** | Admin role can manage all team pods. |
| **No silent backend failures** | **Yes** | No cluster-level dataset system that can silently fail. Standard filesystem + HF libraries = errors surface immediately. |

## 1. Prerequisites

1. **RunPod account** with credit loaded. Sign up at [runpod.io](https://www.runpod.io/).
2. **RunPod API key** (scoped recommended — see [Team Setup](#8-team-setup)). Get one from [Settings > API Keys](https://www.runpod.io/console/user/settings).
3. **SSH key pair** for connecting to pods.
4. **Python 3.10+** with `requests` and `python-dotenv` installed.

## 2. One-Time Setup

### Generate an SSH key (if you don't have one)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/runpod_ed25519
```

Press Enter to skip the passphrase (or set one if you prefer).

### Create a `.env` file

Create a file called `.env` **outside the repo** (e.g., in the parent directory of `caml-research/`):

```bash
# ~/path/to/caml/.env  (NEVER inside the repo)
RUNPOD_API_KEY=rpa_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
RUNPOD_SSH_PUBLIC_KEY=ssh-ed25519 AAAA... user@host
```

To get your public key value:
```bash
cat ~/.ssh/runpod_ed25519.pub
```

The script uses `python-dotenv` and will automatically find the `.env` by walking up from its own directory.

**Use a scoped API key** (`rpa_` prefix) rather than the full-access key. Create one at Settings > API Keys with only the permissions you need (e.g., pod create/stop/terminate). This limits the blast radius if the key leaks.

### Install dependencies

```bash
uv pip install -r requirements.txt
```

## 3. Quick Start

```bash
# Launch a pod with sensible defaults (A6000, 200GB volume)
python scripts/runpod_launch.py launch \
  --image veylansolmira/caml-env:latest \
  --ssh-key ~/.ssh/runpod_ed25519

# The script will:
#   1. Check your balance
#   2. Show estimated cost and ask for confirmation
#   3. Create the pod and wait for it to be ready
#   4. Print the SSH command

# When done, stop the pod (keeps your volume data):
python scripts/runpod_launch.py stop --pod-id <POD_ID>
```

## 4. Command Reference

All commands: `python scripts/runpod_launch.py <action> [options]`

### Actions

| Action | Description |
|--------|-------------|
| `launch` | Create and start a new pod |
| `list` | List all pods with SSH commands |
| `status` | Get detailed pod status (JSON) |
| `stop` | Stop a pod (pause — keeps volume, stops GPU billing) |
| `terminate` | Terminate a pod (**deletes pod volume data permanently**) |
| `gpus` | List all available GPUs with pricing |
| `volumes` | List your network volumes (ID, name, size, region) |
| `balance` | Check account credit balance |

### Flags for `launch`

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | *(required)* | Docker image to run on the pod |
| `--ssh-key` | *(required)* | Path to your SSH **private** key |
| `--gpu` | `value` | Preset tier: `budget`, `value`, `recommended`, `powerful` |
| `--gpu-type` | — | Exact GPU ID (overrides `--gpu`). Run `gpus` to see options. |
| `--gpu-count` | `1` | Number of GPUs |
| `--name` | `runpod-dev` | Pod name (visible in RunPod dashboard) |
| `--volume` | `200` | Pod volume size in GB. Ignored if `--network-volume` is set. |
| `--network-volume` | — | Attach an existing network volume by ID (overrides `--volume`). Use `volumes` to find IDs. |
| `-y` / `--yes` | — | Skip cost confirmation prompt (for scripted use) |

### Other flags

| Flag | Used by | Description |
|------|---------|-------------|
| `--pod-id` | `status`, `stop`, `terminate` | Pod ID to operate on |
| `--ssh-key` | `list` | Required to display SSH connection commands |

### Examples

```bash
# Budget run — QLoRA on a 7B model
python scripts/runpod_launch.py launch \
  --image veylansolmira/caml-env:latest \
  --ssh-key ~/.ssh/runpod_ed25519 \
  --gpu budget

# Recommended setup — A6000 with a persistent network volume
python scripts/runpod_launch.py launch \
  --image veylansolmira/caml-env:latest \
  --ssh-key ~/.ssh/runpod_ed25519 \
  --gpu value \
  --network-volume vol_abc123def

# High-performance — A100 80GB for large models
python scripts/runpod_launch.py launch \
  --image veylansolmira/caml-env:latest \
  --ssh-key ~/.ssh/runpod_ed25519 \
  --gpu powerful \
  --name big-experiment

# Scripted use (skip confirmation)
python scripts/runpod_launch.py launch \
  --image veylansolmira/caml-env:latest \
  --ssh-key ~/.ssh/runpod_ed25519 \
  --gpu value -y

# List your network volumes to get an ID
python scripts/runpod_launch.py volumes
```

## 5. Choosing a GPU

### VRAM requirements by method and model size

| Model size | QLoRA (4-bit) | LoRA (16-bit) | Full fine-tune |
|-----------|---------------|---------------|----------------|
| 7-8B | ~12GB | ~24GB | ~60GB |
| 13B | ~16GB | ~40GB | ~100GB |
| 30-34B | ~24GB | ~80GB | ~240GB |
| 70B | ~48GB | ~160GB (multi-GPU) | not practical on single node |

### GPU recommendation tiers

| Preset | GPU | VRAM | ~Cost/hr | Best for |
|--------|-----|------|----------|----------|
| `budget` | RTX A4000 | 16GB | $0.20 | QLoRA on 7-8B models only |
| `value` | RTX A6000 | 48GB | $0.33 | LoRA up to 13B, QLoRA up to 34B. **Best bang for buck.** |
| `recommended` | L40S | 48GB | $0.79 | Same capacity as A6000 but faster (Ada Lovelace arch) |
| `powerful` | A100 80GB PCIe | 80GB | $1.19 | Full FT 7B, large batch LoRA, QLoRA 70B |

### When to use what

- **Most LoRA/QLoRA work on 7-13B models**: Use `value` (A6000). The 48GB VRAM is more than enough and at $0.33/hr, a 3-hour QLoRA run costs ~$1.
- **Need faster iteration**: Use `recommended` (L40S). Same VRAM but ~2x throughput on FP8 workloads.
- **A100 is overkill** for most QLoRA work on 7-13B. It makes sense for: full fine-tuning, large batch sizes, or 30B+ models.
- **Multi-GPU** (`--gpu-count 2+`): Only needed for models that don't fit in a single GPU's VRAM, or to speed up full fine-tunes.

## 6. Storage

### Pod volume vs Network volume

| | Pod volume | Network volume |
|---|-----------|----------------|
| **Persists on stop** | Yes | Yes |
| **Persists on terminate** | **No — deleted permanently** | Yes |
| **Shared across pods** | No | Yes (one pod at a time) |
| **Cost** | ~$0.10/GB/mo | $0.07/GB/mo (first 1TB) |
| **Setup** | Automatic (via `--volume`) | Create in dashboard, pass ID to `--network-volume` |

**Pod volumes are deleted when you terminate a pod.** This is the single most important thing to understand about RunPod storage. If you `terminate` a pod with a pod volume, all data is gone. Use `stop` to pause (keeps data, stops GPU billing) or use a network volume.

### Network volume setup (recommended)

1. Go to [RunPod Storage](https://www.runpod.io/console/user/storage)
2. Click "Create Network Volume"
3. Choose a region, name, and size (see [sizing guide](#storage-sizing-50gb-vs-200gb-deep-dive) — 100GB is a good starting point)
4. Note the volume ID
5. Use it when launching:
   ```bash
   python scripts/runpod_launch.py launch \
     --image veylansolmira/caml-env:latest \
     --ssh-key ~/.ssh/runpod_ed25519 \
     --network-volume vol_abc123def
   ```
6. Or find the ID with: `python scripts/runpod_launch.py volumes`

Network volumes mount at `/workspace` — same as pod volumes, so your code works the same either way.

**Limitation**: One pod can only mount one network volume at a time. If you need separate storage for different projects, create multiple volumes and use different pods.

### Checkpoint strategy

- **Save checkpoints to `/workspace`** — this is the persistent mount point.
- **LoRA/QLoRA adapters are small** — typically 50-200MB per checkpoint. You can keep many.
- **Full model checkpoints are large** — ~14GB per save for a 7B fp16 model. Prune aggressively (keep best-N + latest).
- **HuggingFace cache** is configured to `/workspace/huggingface` so downloaded models persist across pod restarts.

### Storage sizing: 50GB vs 200GB deep dive

This matters because **running out of disk mid-training crashes the job**, wasting all the GPU time spent so far. The question isn't just "does my data fit" — it's "what's the total cost of ownership including wasted GPU hours from disk failures?"

#### Where disk space actually goes

| Component | Typical size | Notes |
|-----------|-------------|-------|
| HF cache: 7B model (4-bit GPTQ/AWQ) | ~4GB | Smallest usable format |
| HF cache: 7B model (fp16 safetensors) | ~14GB | Full precision |
| HF cache: 13B model (4-bit) | ~7GB | |
| HF cache overhead (duplicate formats, partial downloads) | +30-50% | HF downloads safetensors + may keep .bin fallback |
| Text dataset (typical NLP fine-tune) | 1-5GB | Tokenized + preprocessed copies |
| Large dataset (multi-task, instruction tuning) | 5-20GB | Multiple splits + processing artifacts |
| QLoRA adapter checkpoint | ~50-200MB | Tiny — the whole point of LoRA |
| Full merged model save (7B fp16) | ~14GB | If you save the merged model, not just the adapter |
| Training logs, wandb cache, misc | ~1-2GB | Accumulates over time |
| pip/conda cache (if installing packages) | 2-5GB | Torch alone is ~2GB |

#### 50GB network volume: what works and what doesn't

**Works for:**
- Single-model QLoRA workflow (one 7B model in 4-bit + adapters + dataset)
- Quick experiments where you clean up between runs
- Budget breakdown: ~4GB model + ~3GB dataset + ~1GB adapters + ~2GB misc = ~10GB used, ~40GB free

**Breaks when:**
- You download a second model (or the same model in a different format)
- HF cache bloats from partial downloads or format duplicates (~20GB for one 7B fp16 model with overhead)
- You save full merged models instead of just adapters
- You run multiple experiments without cleanup and accumulate 10+ checkpoint directories
- A dataset is larger than expected (instruction-tuning datasets with many splits can hit 10-20GB)

**The real risk**: HF cache is greedy and hard to predict. A `from_pretrained()` call may download both safetensors and bin files, doubling the expected size. A failed download leaves fragments. At 50GB, you have maybe 20-30GB of breathing room after one model, and one bad cache event puts you into crash territory.

#### 200GB network volume: what it buys you

- Comfortable room for 2-3 models in cache simultaneously
- Multiple experiment checkpoints without cleanup pressure
- Full merged model saves (14GB each) — keep 3-4 before needing to prune
- No anxiety about `disk full` crashes mid-training
- Room to grow if you switch to larger models later

#### Cost comparison: is 200GB actually cheaper?

| | 50GB volume | 200GB volume |
|---|-----------|--------------|
| Monthly storage cost | $3.50/mo | $14.00/mo |
| Difference | — | +$10.50/mo |

Now consider the cost of a single disk-full crash:
- A 3-hour QLoRA run on A6000 that crashes at hour 2.5 = **$0.83 wasted**
- A 6-hour LoRA run on L40S that crashes at hour 5 = **$3.95 wasted**
- Plus your time to diagnose, clean up, and restart

**If you hit even 2-3 disk crashes per month, the wasted GPU time exceeds the $10.50/mo storage difference.** And disk crashes tend to happen at the worst time — late in a training run when checkpoints accumulate.

#### Recommendation

| Scenario | Volume size | Monthly cost |
|----------|-----------|--------------|
| **Single person, single model, disciplined cleanup** | 50GB | $3.50 |
| **Team shared volume, regular experimentation** | 100-200GB | $7-14 |
| **Multiple models, full fine-tuning, or long-running experiments** | 300GB+ | $21+ |

**Start with 100GB as a compromise** if spending is a primary concern. It gives enough headroom for one model + generous cache + multiple adapter checkpoints, without the anxiety of 50GB. You can always increase later (but never decrease).

If the team shares one network volume and multiple people are caching models, go to 200GB immediately — model cache collisions alone will eat 50GB fast.

Volumes can be increased in size later but **never decreased**.

## 7. Cost Management

### Billing model

- **Per-second billing** — you pay only for actual GPU time, rounded to the second.
- **Secure Cloud** (what our script uses) costs more than Community Cloud but gives you a public IP for direct SSH.

### There is no hard spending cap

RunPod does **not** offer a user-configurable spending limit. What exists:

- **Low-balance email alerts** — you get notified when funds are running low.
- **Auto-stop on near-zero balance** — pods auto-stop ~10 minutes before your credit runs out.
- **Credit-based** — you prepay credit, so you can't accidentally run up a bill beyond what you've loaded.

If you need hard per-user budget enforcement, see the [AWS/Lambda gatekeeping note](#on-awslambda-gatekeeping) below.

### Stop vs Terminate — cost implications

| State | GPU billing | Storage billing |
|-------|-------------|-----------------|
| Running | Yes | Yes (included in GPU cost) |
| Stopped | **No** | Pod volume: ~$0.10/GB/mo. Network volume: $0.07/GB/mo. |
| Terminated | No | Pod volume: deleted. Network volume: still billed separately. |

**Always `stop` when not actively using the GPU.** Don't leave pods running overnight.

### Cost examples

| Scenario | GPU | Duration | Approx cost |
|----------|-----|----------|-------------|
| QLoRA 7B, single epoch | A6000 (value) | ~1.5 hr | ~$0.50 |
| QLoRA 7B, full run with eval | A6000 (value) | ~5 hr | ~$1.65 |
| LoRA 13B, multi-epoch | L40S (recommended) | ~8 hr | ~$6.30 |
| A100 full day | A100 80GB PCIe | 24 hr | ~$28.56 |

The script shows estimated cost before launching and asks for confirmation:
```
Launching pod 'my-experiment' with NVIDIA RTX A6000 x1
  Estimated cost: ~$0.33/hr ($7.92/day)
  Image: veylansolmira/caml-env:latest
  Storage: 200GB pod volume
Proceed? [Y/n]
```

## 8. Team Setup

### Scoped API keys

RunPod API keys with the `rpa_` prefix support per-endpoint permissions. Each team member should have their own scoped key rather than sharing a single full-access key.

Create scoped keys at [Settings > API Keys](https://www.runpod.io/console/user/settings). Recommended permissions for a researcher:
- Pod create, stop, terminate
- GPU type listing
- Balance check

### Team roles

| Role | Can do |
|------|--------|
| Basic | View pods |
| Dev | Create/manage pods, use API |
| Billing | Manage credits and payments |
| Admin | All of the above + manage team members and API keys |

### Shared billing pool

All team members share one credit balance. There are **no per-user spending limits** natively. If this is a concern, see the gatekeeping note below.

### On AWS/Lambda gatekeeping

Worth building if you need:
- Hard per-user budget limits
- Policy enforcement (restrict GPU types, require approval for expensive GPUs)
- Audit trail beyond what RunPod provides

**Not worth it** if scoped API keys + team roles cover your needs. It adds significant operational complexity (API Gateway + Lambda + DynamoDB for tracking).

**Recommendation**: Start with RunPod's native team features. Only build a proxy layer if you outgrow them. The architecture would be: API Gateway -> Lambda (validates request against budget/policy in DynamoDB) -> RunPod API. This is a separate project, not covered here.

## 9. Security Notes

### API keys in audit logs

RunPod auto-creates an API key per pod for internal use and logs the creation/deletion. These keys appear in audit logs but *not* in the API Keys settings page. This is intentional (SOC 2 compliance), not a leak. Use scoped API keys to limit what any single key can do.

### `.env` file placement

- Keep the `.env` file **outside the repository** — the script walks up directories to find it.
- **Never commit** `.env` or API keys to git. The `.gitignore` should already exclude it.
- Each team member has their own `.env` with their own scoped key.

### Container security

- Pods run as `root` by default. This is standard for GPU workloads but be aware.
- The SSH key injected via `PUBLIC_KEY` env var is the only authentication method — no passwords.
- Pods on Secure Cloud get a public IP. Only port 22 (SSH) and 8888 (Jupyter) are exposed.

## 10. How It Works

The script uses RunPod's [GraphQL API](https://docs.runpod.io/docs/graphql-api) to manage pods and the [REST API](https://docs.runpod.io/docs/rest-api) for network volumes. When you launch:

1. **Secure Cloud** is used (not Community Cloud) — this gives you a public IP for direct SSH.
2. Port **22/tcp** (SSH) and **8888/http** (Jupyter) are exposed.
3. Your SSH public key is injected via the `PUBLIC_KEY` environment variable.
4. A `dockerArgs` startup command configures `~/.ssh/authorized_keys` and starts `sshd`.
5. If `--network-volume` is provided, the volume is attached instead of creating a pod volume.
6. The pod info (ID, SSH command, GPU type) is saved to `secure/runpod_current.json`.

### Custom Docker images

We use `veylansolmira/caml-env:latest` which comes pre-installed with common ML libraries (PyTorch, Transformers, Unsloth, etc.). This avoids the setup overhead that plagues platforms with pre-configured-only images.

To build your own image, include `openssh-server` for SSH access. The `dockerArgs` in the script handles the rest.

## 11. Troubleshooting

| Problem | Solution |
|---------|----------|
| `RUNPOD_API_KEY not set` | Create `.env` file with your API key (see setup above) |
| `RUNPOD_SSH_PUBLIC_KEY not set` | Add your public key to `.env` |
| Pod stuck in "pending" | GPU type may be out of stock — try a different one (`gpus` to see availability) |
| SSH connection refused | Wait ~30s after "Pod ready" for sshd to fully start |
| SSH permission denied | Check that your `.env` has the public key matching your `--ssh-key` private key |
| Pod terminates immediately | Image may be invalid — check the Docker image name and tag |
| "Error listing network volumes" | API key may lack permission. Use a key with network volume access. |
| Data gone after terminate | Pod volumes are deleted on terminate. Use `stop` to pause, or use `--network-volume` for persistent storage. |
| Volume says "in use" | A network volume can only be mounted by one pod. Stop/terminate the other pod first. |
| Checkpoint fills disk | Prune old checkpoints. LoRA adapters are small (~50MB) but full model saves are ~14GB per checkpoint for 7B. |
| Can't decrease volume size | RunPod limitation — volumes can only grow, never shrink. See [sizing guide](#storage-sizing-50gb-vs-200gb-deep-dive). |
| Cost unclear before launch | The script shows estimated cost and asks for confirmation. Use `gpus` to see all pricing. |
