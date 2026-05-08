# Differentiation from Zhang 2025 ("Constitution or Collapse?")

How CaML's constitution-vs-midtraining work differs from Xue Zhang's CAI replication on Llama 3-8B (arXiv 2504.04918, April 2025). Compiled in response to Jasmine's question on 2026-05-07: *"How is our work different from what they did?"*

## TL;DR — the load-bearing differentiator

**Zhang only studies CAI in isolation. Our paper is the *comparison* between CAI and mid-training as alternative value-injection methods.**

Zhang's research question is "does CAI work at 8B?" Ours is "does CAI vs mid-training work better as a value-injection method for animal welfare, and how does that depend on scale?" These are different empirical structures. CaML has a mid-training comparator (Jasmine's `Basellama_plus3kv3_*` series); Zhang has no comparator. Just having that contrast is the substantive novel contribution.

## Side-by-side

| Axis | Zhang 2025 | CaML / our work |
|---|---|---|
| Research question | "Does CAI work at 8B?" | "Does CAI vs mid-training work better as a value-injection method for animal welfare?" |
| Primary contribution | Showing CAI struggles at 8B | The contrast itself: comparing two value-injection methods |
| Comparator | None — only studies CAI | **Mid-training** (continued pre-training on animal-aware documents) |
| Critic model | Self (Llama 3-8B critiques itself) | External (Gemini 2.5 Flash Lite) |
| RL stage | DPO-CAI included | Skipped — SFT only |
| Domain | General harmlessness (HH-RLHF) | Animal welfare specifically (5-principle written constitution) |
| Evaluation | MT-Bench (general) | AHB + MORU (animal-welfare specific) |
| Scale ladder | Single point (8B) | 8B baseline + 1B replication for scale-trend signal |
| Failure mode found | DPO-CAI collapse, traced to emoji-repetition in self-critique data | N/A — verified our pipeline doesn't have this signature |

## Secondary differentiators that strengthen the case

1. **Our pipeline architecturally avoids Zhang's collapse mechanism.** Zhang found DPO-CAI collapse traced to emoji+polite-closer repetition in self-critique outputs. We use an *external* critic (Gemini), so revisions don't inherit the small model's failure mode. Verified empirically: **0 / 3000** responses in `cai-animal-harm-sft` contain emojis or `:)` smileys; **0 / 3000** have intra-response polite-closer repetition. So "data hygiene cause" identified by Zhang is architecturally absent here, not just luckily absent.
2. **Domain-specific constitution and benchmark.** Zhang's harmlessness constitution is generic. Our 5-principle constitution is specifically about animal welfare, and the eval (AHB) is specifically scoped to that. This is a substantive specialization, not a different prompt set on the same eval.
3. **Going down the scale ladder.** Zhang only studied 8B and concluded CAI may need larger scale. We're going *smaller* (1B) — testing the prediction directly. If 1B replication produces a usable model (no collapse, eval_loss converges), that's a counter-data-point to Zhang's "needs larger scale" thesis. If it fails, we corroborate.

## Verbal phrasings (for the call)

When asked "how is this different from Zhang?":

- **Short version:** *"Zhang only studies CAI; we compare it to mid-training. The comparison is the contribution."*
- **Medium version:** *"Zhang asks 'does CAI work at 8B' and only looks at CAI. We ask 'does CAI vs mid-training work better as a value-injection method,' and we have both sides — Jasmine's mid-trained 0.358 model is the comparator, the constitutional 0.305 is the CAI side. Plus we use an external critic so we sidestep his specific collapse mode, we're domain-specific to animal welfare, and we're going down to 1B for a scale-trend signal."*
- **Defensive add-on if pushed:** *"We even verified empirically that our training data doesn't have the emoji-repetition pathology Zhang traced his collapse to — 0 emojis in 3000 responses, because our pipeline uses Gemini as critic instead of self-critique."*

## What this doc is NOT

- Not a claim that we're doing more rigorous CAI than Zhang. Zhang did full DPO-CAI; we do simplified SFT-CAI. He goes deeper on one method; we go wider with a comparison.
- Not a claim that we're at frontier scale. We're not. Per Jasmine: *"We're never going to get frontier scale though. That criticism is unsolvable."* This is an open-research limitation we share with Zhang.

## Open questions to address before paper-quality differentiation

- **Cell F ablation** (per PLAN.md): if we run SFT on `revised_response` *without* the critique loop and it matches full CAI, then our "CAI" is effectively distillation from Gemini. That would weaken the differentiation from Zhang (who genuinely runs the full critique loop). Worth doing.
- **Replicate Zhang's MT-Bench numbers** on our model so we have a directly comparable point on his eval, in addition to AHB. (Their numbers: ~40% ASR drop on harmlessness, 9.8% helpfulness drop.)
- **Email Xue Zhang for his model checkpoints** — already sent (`xuezhang68@stanford.edu`); no response yet. With his weights we could run AHB on his model and have a true cross-pipeline comparison.

## Sources

- Zhang 2025 paper: [`caml-research/papers/constitution-or-collapse-cai-llama3-8b.md`](../papers/constitution-or-collapse-cai-llama3-8b.md)
- arXiv: https://arxiv.org/abs/2504.04918
- Open CAI implementations note: [`cai-open-implementations.md`](cai-open-implementations.md)
- CAI dataset framing (related): [`cai-dataset-framing.md`](cai-dataset-framing.md)
- Hygiene verification (the "0 emojis" finding lives in cai-dataset-framing.md)
