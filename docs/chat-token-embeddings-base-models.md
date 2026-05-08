# Chat-Template Token Embeddings — Llama 3.1 8B Base vs Llama 3.2 1B Base

A methodological finding from 2026-05-08 while attempting to launch CAI SFT on Llama 3.1 8B base. The two base-model releases handle chat-template token embeddings very differently, with direct implications for whether base-model fine-tuning on chat-formatted data is viable without manual embedding repair.

## TL;DR

**Llama 3.1 8B base has *literally zero* embeddings for the post-pretraining chat tokens** (`<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`, `<|finetune_right_pad_id|>`, `<|python_tag|>`). L2 norm ≈ 2 × 10⁻²¹ — numerical noise.

**Llama 3.2 1B base has those tokens initialized to a shared canonical non-zero vector** (L2 norm ≈ 0.545 for all of them, identical to many decimals). Not trained, but a sensible starting point — LoRA can update from there.

This is why our CAI training run on 1B base ran successfully, while the same script on 8B base was blocked by Unsloth's `fix_untrained_tokens` check (eps = 1e-16): 1B's 0.545 passes, 8B's 10⁻²¹ doesn't.

## Measured embedding L2 norms

| Token | 3.2 1B base | 3.1 8B base |
|---|---|---|
| `<|begin_of_text|>` (128000) | 0.863 | 0.477 |
| `<|end_of_text|>` (128001) | 0.968 | 0.206 |
| `<|start_header_id|>` (128006) | 0.545 | **2.09e-21** |
| `<|end_header_id|>` (128007) | 0.545 | **2.07e-21** |
| `<|eot_id|>` (128009) | 0.545 | **2.04e-21** |
| `<|finetune_right_pad_id|>` (128004) | 0.545 | **2.07e-21** |
| `<|python_tag|>` (128010) | 0.545 | **2.08e-21** |
| Normal vocab tokens (id=100..100000) | 0.80–1.16 | 0.45–0.72 |

Probe was `transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=bf16)` followed by `model.get_input_embeddings().weight[token_id].norm()`. Each measurement reproducible from `caml-research/experiments/cai-1b-replication/`.

Begin-of-text and end-of-text are non-zero on both because they're used at every document boundary in pretraining (real training signal). The post-pretraining chat tokens have no natural occurrence in pretraining data; their state in the base release reflects only the initialization policy at release time.

## Why the difference

Hypothesis (not directly confirmed but strongly suggested by the data): Meta refined release practices between Llama 3.1 (released Jul 2024) and Llama 3.2 (released Sep 2024). The newer 3.2 base release explicitly initializes uninstructed-tokens to a non-zero canonical vector before publishing weights. The older 3.1 release left those embeddings at whatever they were post-initialization (essentially zero in bf16).

Note that *all five* uninstructed-tokens in 3.2 1B share the *same* L2 norm to many decimals (0.5449...) and the same max_abs and mean_abs — strong evidence they're all initialized to the same canonical vector, then never trained.

## Implications for our existing 1B + CAI run

Our 1B chat tokens started at a single shared canonical point, then LoRA training (with `embed_tokens` in target_modules) updated them. They would have all received the same gradient initially (since they were the same vector and saw similar contexts), then diverged based on context-specific signal. ~3 epochs over 3000 examples is not a lot of differentiation training.

The model likely *does* fumble chat-format rendering somewhat — chat tokens treated as near-equivalent rather than role-specific. This is a real confound on the 1B+CAI 0.100 AHB score: the model might be doing decent constitutional reasoning but rendering it in slightly broken chat format that the rubric scorer penalizes.

This is testable: compare chat-format quality of our 1B+CAI to (a) bare 1B base, (b) 1B Instruct as a strong upper bound. If our model's chat-format errors are the bottleneck, we'd expect proper chat-token initialization + retrain to lift AHB meaningfully.

## Implications for 8B + base CAI

You **cannot LoRA-train Llama 3.1 8B base on chat-formatted data without first initializing the chat-token embeddings**. Unsloth's check correctly catches a real numerical problem (gradient × ~0 ≈ 0 → no learning, or NaN under autocast). Options:

1. **Pre-initialize before LoRA**: replace `embed_tokens[128004:128012]` and corresponding `lm_head` rows with the mean of trained-token embeddings (or `<|reserved_special_token_*|>` row values), then call `setup_lora`. ~10 lines of code. Cleanest.
2. **Use `modules_to_save=["embed_tokens","lm_head"]`** in PEFT config: full-train those layers (not LoRA). Higher memory cost but straightforward.
3. **Use Llama 3.1 8B Instruct**: chat tokens are trained. But reintroduces the Instruct-prior confound we explicitly chose 1B base to avoid.

This is also why Jasmine's existing 8B CAI run uses Instruct, not base: it's a necessity at 8B given the embedding state, not a preference.

## What this means for the paper

This finding is genuinely useful and worth documenting:

> *"Llama 3.1 8B base has uninitialized embeddings for chat-template tokens (norm ~10⁻²¹). Llama 3.2 1B base does not (norm ~0.545, shared canonical initialization). This affects whether base-model fine-tuning on chat-formatted data is viable without manual embedding repair, and partly explains why CAI replications in this regime typically use Instruct variants."*

Citation for related discussion: see Unsloth's [Untrained Tokens documentation](https://docs.unsloth.ai/basics/continued-pretraining#how-do-i-use-the-fix-tokenizer-feature) and various community posts about Llama 3 base fine-tuning issues.

## Reproducing this measurement

On any machine with `transformers` and an HF token that's accepted both Llama licenses:

```python
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

for model_id in ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-8B"]:
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ["HF_TOKEN"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cpu",
        low_cpu_mem_usage=True, token=os.environ["HF_TOKEN"],
    )
    emb = model.get_input_embeddings().weight.detach().to(torch.float32)
    for name in ["<|start_header_id|>", "<|eot_id|>"]:
        tid = tok.convert_tokens_to_ids(name)
        print(f"{model_id}  {name}  norm={emb[tid].norm():.3e}")
```
