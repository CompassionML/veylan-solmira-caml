# Linear Probes for Compassion Measurement

**Project:** Mechanistic measurement of compassion in LLM activations
**Status:** Initial probing complete, validation in progress
**Date:** February 27, 2026

---

## 1. Infrastructure

We set up compute on StrongCompute Sydney to run activation extraction and probe training:

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 3090 Ti (24GB VRAM) |
| Model | Llama 3.1 8B Instruct (~16GB) |
| Container | Custom Docker with Flash Attention 2 |

Scripts for extraction (`extract.py`) and training (`train.py`) in `experiments/linear-probes/src/`.

---

## 2. Approach: Contrastive Pairs

We trained probes using **contrastive activation analysis**: extract activations from compassionate vs. non-compassionate responses to the same question, then learn a direction that separates them.

### Training Data

| Metric | Value |
|--------|-------|
| Pairs | 105 |
| Source | AHB questions |
| Generator | Claude (persona prompts) |
| Format | `{question, compassionate_response, non_compassionate_response}` |

**Personas used:**
- Compassionate: "Animal welfare expert and ethicist"
- Non-compassionate: "1950s agricultural textbook writer"

### Activation Selection

We mean-pool over exact response tokens (excluding the user prompt). This isolates what differs between responses.

### Probe Architecture

Logistic regression with L2 regularization on hidden states → P(compassionate).

---

## 3. Results

### Key Finding

**Layer 8 (25% depth) is optimal** — contradicting the ~75% depth heuristic from steering literature.

| Layer | Depth | Accuracy | AUROC |
|-------|-------|----------|-------|
| **8** | **25%** | **95.2%** | **0.995** |
| 12 | 38% | 92.9% | 0.964 |
| 16 | 50% | 90.5% | 0.957 |
| 20 | 63% | 90.5% | 0.914 |
| 24 | 75% | 88.1% | 0.909 |
| 28 | 88% | 88.1% | 0.891 |

Performance decreases monotonically with depth.

### Interpretation

Compassion appears to be a "surface" feature — encoded early in the model's representations (tone, framing, word choice) before deeper layers blend it with semantic content.

---

## 4. Operationalization of Compassion

### What We're Measuring

We operationalize compassion via the AHB's 12 dimensions of moral reasoning toward animals:

- Moral consideration, harm minimization, sentience acknowledgment
- Prejudice avoidance, scope sensitivity, contextual welfare salience
- Actionability, epistemic humility, trade-off transparency
- Evidence-based attribution, novel entity precaution, cautious impact consideration

### The Circularity Problem

Our pairs are AHB-derived and Claude-generated:

```
AHB defines compassion → Claude generates pairs → Probe learns pattern
```

**What we can claim:** We detect AHB-style compassion in Llama activations with high accuracy.

**What we cannot claim:** Independent validation of AHB, or measurement of "true" compassion.

### Potential Confounds

| Confound | Risk |
|----------|------|
| Single generator (Claude) | Systematic stylistic artifacts |
| 1950s framing | Detecting era, not compassion |
| AHB-derived prompts | Cannot independently validate AHB |

---

## 5. Next Steps

### Immediate: Validate the Probe

| Task | Description |
|------|-------------|
| **Negative controls** | Test probe on non-compassion concepts — should NOT activate |
| **AHB ↔ Probe correlation** | Compare AHB scores to probe scores on same responses |
| **Re-extract with exact boundaries** | Current results used 50% heuristic; rerun with fixed code |
| **Earlier layers** | Test layers 4, 6 to confirm monotonic trend |

### System Prompt Experiments

| Task | Description |
|------|-------------|
| **Persona → AHB** | Run AHB with different system prompts, measure probe scores |
| **Shard theory approach** | Test specific uncompassionate personas (per Jasmine) |

### Confound Ablation

| Task | Description |
|------|-------------|
| **Non-AHB scenarios** | Test probe on compassion scenarios outside AHB |
| **Modern-vs-modern pairs** | Same era, different compassion — isolate style |

### Longer Term

| Task | Description |
|------|-------------|
| **Correlated/anti-correlated values** | What values cluster with compassion? (Bergson) |
| **Base vs. fine-tuned** | Compare Llama to CaML fine-tuned models |
| **In-context vs. fine-tuning** | How deep do different training approaches go? |

---

## 6. Contacts

- **Raphael** — Reached out re: activation selection methodology
- **Claire** — Research manager, MATS

---

*See full details in `docs/presentation.md` and `roadmap.md`*
