# 8B → 1B Config Notes

What changes (and what doesn't) when moving Jasmine's CAI fine-tune from Llama 3.1 8B to Llama 3.2 1B. Notes for tuning future 1B runs after the first replication completes.

## What stays the same

LoRA topology and the recipe both carry over verbatim because they're model-agnostic:

| Aspect | Why no change |
|---|---|
| Architecture flag | Both Llama 3.x family; `--architecture llama` auto-detects |
| Tokenizer | Same 128k Llama 3 vocab |
| Chat template | Same `<|begin_of_text|>` / `<|start_header_id|>` tokens |
| Dataset | `cai-animal-harm-sft` either way |
| LoRA r/α/dropout | r=16, α=8, dropout=0 — adapter dimensions auto-scale with model dimensions |

## What we *can* tune for 1B (we didn't, in the first run)

| Parameter | 8B default | Reasonable 1B value | Why |
|---|---|---|---|
| `--per-device-batch-size` | 4 | 8–16 | 1B uses ~9 GB VRAM at batch=4 on A4000 (16 GB total). Bigger batches → more stable gradients, faster wall-clock |
| `--learning-rate` | 5e-5 | 1e-4 to 2e-4 | Smaller models often tolerate (and benefit from) higher LRs |
| `--max-steps` | 500 | 250–350 | 1B has ~1/8 the params; converges faster on the same data; risks overfitting beyond a point |
| `--gradient-accumulation-steps` | 4 | 1–2 | Only needed if batch can't fit. Less needed at 1B |
| Gradient checkpointing | enabled | disabled | Trades memory for compute. 1B doesn't need the memory savings; disabling speeds training |
| Eval frequency | 20 steps | 20–50 | ~23 sec per eval pause; less frequent saves time without losing signal |
| Sequence length | 1048 | 1048 | Capped by dataset content (CAI prompts/responses are short), not by model capacity |

## Things worth watching at 1B specifically

1. **Collapse risk.** Per "Constitution or Collapse?" (arXiv 2504.04918), 8B already showed collapse in the DPO-CAI stage. We skip DPO, but at 1B watch eval_loss curves: rising eval_loss mid-training = retune `max_steps` or `learning_rate`.
2. **LoRA target modules.** Jasmine's `train_universal.py` includes `embed_tokens` and `lm_head` as LoRA targets. At 1B with smaller hidden dims, this is proportionally significant for the adapter parameter count. Worth knowing for any ablation that drops them.
3. **Warmup fraction.** Defaults to ~10% of training steps. If you reduce `max_steps` below 500, scale warmup proportionally — otherwise warmup eats a larger fraction of training.

## Suggested config for the next 1B iteration (after the first run lands)

```bash
--per-device-batch-size 8 \
--gradient-accumulation-steps 2 \
--max-steps 300 \
--learning-rate 1e-4
# (consider disabling gradient checkpointing for speed)
```

Should run in roughly half the wall-clock time of the first replication and produce a model of similar quality. Useful for cell-D or cell-F ablations.

## What the first replication run is using

All defaults from `train_universal.py` — Jasmine's 8B recipe applied verbatim to a 1B base:

- batch_size=4, accum=4 (effective 16)
- LR=5e-5
- max_steps=500
- LoRA r=16, α=8, dropout=0

Honest framing: this is "Jasmine's recipe verbatim, applied to a 1B base." Not optimized for 1B. The eval_loss curve from this run will tell us if retune is warranted.
