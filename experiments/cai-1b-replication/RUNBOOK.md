# 1B CAI Replication — Runbook

Goal: replicate Jasmine's CAI fine-tuning (`constitution-vs-midtraining-paper/pipeline/`) at the 1B scale, push results to `VeylanSolmira/` on HuggingFace, log training to W&B. Per the May 1 call: "1b models · what changes for a 1b · ensure validation · wandb."

---

## What this experiment actually does

`train_universal.py` already takes the model path as a CLI arg with sensible defaults (Jasmine's `cai-animal-harm-sft` is the default dataset).

The pipeline:
1. **Llama 3.2 1B base** (not Instruct) ← starting checkpoint
2. SFT on `CompassioninMachineLearning/cai-animal-harm-sft` (LoRA r=16, α=8)
3. Merged model pushed to `VeylanSolmira/Base1b_constitutionfinetune_v1`
4. (Next phase) AHB eval on the resulting model

**Why base instead of Instruct:** Jasmine's existing constitutional model (`Instruct8b_constitutitutionfinetune_step200`) starts from Llama 3.1 8B *Instruct* — meaning Meta's SFT + DPO + safety post-training is already baked in before CAI is applied. That confounds the comparison vs the mid-trained side, which starts from base. Running our 1B replication from **base** sidesteps this confound and tests "does the CAI recipe work starting from a clean checkpoint?" which is what PLAN.md's Cell C is *supposed* to test but doesn't (per our findings earlier).

(Note: Jasmine's CAI dataset itself was generated using a Llama Instruct model as the response generator — see `pipeline/gen.py`. So the *training data* contains Instruct-style outputs. We're fine-tuning a base model to imitate those outputs, which is itself a form of instruction tuning. Standard for the Bai-et-al recipe.)

---

## Pre-flight — all confirmed ✅

| Item | Status |
|---|---|
| W&B API key | `secure/wandb-key.txt` (86 bytes) |
| HF write token (push to `VeylanSolmira/`) | `secure/hg-veylan-write` (token resolves to `VeylanSolmira`, write scope confirmed) |
| RunPod API key + SSH public key | `~/Desktop/ai_dev/caml/.env` has both |
| Local SSH private key | `~/.ssh/runpod_ed25519` |
| Docker image | `veylansolmira/caml-env:latest` (pre-built with Unsloth, HF Transformers, PyTorch) |
| Training script | `caml-research/experiments/cai-1b-replication/train_universal.py` (verbatim copy from Jasmine's pipeline) |
| Network volume | None provisioned — fine for this run, ephemeral pod disk is enough |

---

## GPU sizing

For 1B + LoRA (~4–6 GB VRAM peak), `--gpu budget` (A4000, 16 GB, $0.25/hr) is sufficient. `--gpu value` (A6000, $0.49/hr) gives more headroom. Either is fine; A4000 is cheapest. Estimated runtime: 20–40 min for max_steps=500. Total cost: ~$0.10–0.30.

---

## Launch sequence

### 1. Provision the pod (interactive mode)

```bash
cd /Users/infinitespire/Desktop/ai_dev/caml/caml-research
source .venv/bin/activate

python scripts/runpod_launch.py launch \
  --image veylansolmira/caml-env:latest \
  --ssh-key ~/.ssh/runpod_ed25519 \
  --gpu budget \
  --name caml-1b-cai-replication
```

The launcher prints cost estimate, asks to confirm, then prints the SSH-over-TCP command (looks like `ssh root@216.x.x.x -p 12345 -i ~/.ssh/runpod_ed25519`).

**Save that SSH command** — you'll use it in steps 2 and 3.

### 2. Copy the training script to the pod

```bash
# from your local laptop, replace HOST and PORT with the values from step 1
scp -i ~/.ssh/runpod_ed25519 \
  -P <PORT> \
  /Users/infinitespire/Desktop/ai_dev/caml/caml-research/experiments/cai-1b-replication/train_universal.py \
  root@<HOST>:/workspace/train_universal.py
```

### 3. SSH in and launch the run

```bash
# SSH command from step 1
ssh -i ~/.ssh/runpod_ed25519 -p <PORT> root@<HOST>

# inside the pod:
cd /workspace

export WANDB_API_KEY="<paste from secure/wandb-key.txt>"
export HF_TOKEN="<paste from secure/hg-veylan-write>"

python train_universal.py \
  --model-path meta-llama/Llama-3.2-1B \
  --architecture llama \
  --wandb-run-name 1b-cai-replication-2026-05-07 \
  --wandb-api-key "$WANDB_API_KEY" \
  --hf-repo-name VeylanSolmira/Base1b_constitutionfinetune_v1 \
  --hf-token "$HF_TOKEN"
```

(If you'd rather pipe the secrets straight from local files via SSH heredoc to avoid copy-paste, see "One-liner alternative" below.)

### 4. Monitor

- W&B run page (linked in script output) — live loss curves
- Or `tail -f` the training log inside the pod

### 5. After completion

The merged model is pushed to `VeylanSolmira/Base1b_constitutionfinetune_v1` automatically. Confirm with:

```bash
curl -s -H "Authorization: Bearer $(cat secure/hg-veylan-write)" \
  https://huggingface.co/api/models/VeylanSolmira/Base1b_constitutionfinetune_v1 \
  | python3 -m json.tool | head -30
```

### 6. Tear down

From your local machine:

```bash
python scripts/runpod_launch.py terminate --pod-id <pod-id-from-step-1>
```

---

## One-liner alternative (no copy-paste of secrets)

If you'd rather not paste the W&B and HF keys into the SSH session manually, this pipes both directly:

```bash
WANDB_KEY=$(cat /Users/infinitespire/Desktop/ai_dev/caml/secure/wandb-key.txt)
HF_KEY=$(cat /Users/infinitespire/Desktop/ai_dev/caml/secure/hg-veylan-write)

ssh -i ~/.ssh/runpod_ed25519 -p <PORT> root@<HOST> "bash -s" <<EOF
cd /workspace
export WANDB_API_KEY="$WANDB_KEY"
export HF_TOKEN="$HF_KEY"
python train_universal.py \
  --model-path meta-llama/Llama-3.2-1B \
  --architecture llama \
  --wandb-run-name 1b-cai-replication-2026-05-07 \
  --wandb-api-key "\$WANDB_API_KEY" \
  --hf-repo-name VeylanSolmira/Base1b_constitutionfinetune_v1 \
  --hf-token "\$HF_TOKEN"
EOF
```

(Note the `\$` escaping inside the heredoc to prevent local expansion of variables that should remain as the pod's env vars.)

---

## Open considerations

1. **Llama 3.2 1B license click-through.** Meta gates the Llama 3.2 family. If you haven't accepted the license yet, visit https://huggingface.co/meta-llama/Llama-3.2-1B and click accept while logged in as VeylanSolmira. Then HF_TOKEN works for the download. (Worth accepting Llama-3.2-1B-Instruct on the same trip in case you want to run an Instruct comparison later.)
2. **Output repo private vs public.** Default in this runbook pushes to `VeylanSolmira/Base1b_constitutionfinetune_v1`. If you want it private, the script's `push_to_hub` call may need a flag — worth checking before launch if privacy matters.
3. **Hyperparameter tweaks for 1B (skip for first run).** After the baseline works, consider: `--per-device-batch-size 8` (more headroom at 1B), `--max-steps 300` (1B converges faster), `--learning-rate 1e-4` (smaller models often want higher LR).

---

## After this works: natural next experiments

1. **Eval the resulting 1B CAI model on AHB** — gives you the 1B counterpart to Jasmine's 0.305. Same `inspect_evals/ahb` task, your local `caml-research/.venv` already has the harness.
2. **8B baseline reproduction** — same script, swap `--model-path` to `meta-llama/Llama-3.1-8B-Instruct`. Validates that you can reproduce Jasmine's 0.305 from scratch.
3. **1B mid-training** — replicate the doc-tune side at 1B (different pipeline, conceptually parallel).
4. **Cell D at 1B** — `base + doc-tune + CAI` at small scale.
