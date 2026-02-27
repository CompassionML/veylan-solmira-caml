# Linear Probes for Compassion Measurement

**Project:** Mechanistic measurement of compassion in LLM activations
**Date:** February 27, 2026

---

## The 2-Minute Pitch

**Problem:** We train LLMs to be compassionate, but we can't see inside them. Current evaluation relies on output scoring (the Animal Harm Benchmark), which tells us *what* a model says but not *whether it means it*. A model could game the benchmark without genuine value alignment.

**Solution:** Linear probes that detect compassion directly in model activations. Instead of scoring outputs, we measure internal representations — giving us a window into whether compassion training actually changes how the model thinks.

**Result:** A probe trained on 105 contrastive pairs achieves 95.2% accuracy distinguishing compassionate from non-compassionate responses. Unexpectedly, compassion is encoded earliest (layer 8, 25% depth), not late as steering literature suggests.

**Why it matters:**
1. **Validation tool** — Test if fine-tuning genuinely shifts representations or just suppresses outputs
2. **Early warning** — Detect if a model "knows" the compassionate answer but isn't giving it
3. **Mechanistic insight** — Compassion appears to be a "surface" feature (tone, framing) not deep reasoning

**Next step:** Validate the probe with negative controls and correlate with AHB scores.

---

## Why This Work Matters

| Stakeholder | Value Proposition |
|-------------|-------------------|
| **CaML team** | Validate that compassion fine-tuning changes representations, not just outputs |
| **Safety researchers** | Detect misalignment between internal state and external behavior |
| **Interpretability field** | Novel finding: optimal probe layer contradicts steering heuristics |

**The core insight:** Output benchmarks can be gamed. Internal probes cannot (as easily). This work bridges behavioral evaluation with mechanistic understanding.

---

## Anticipated Questions

| Challenge | Response |
|-----------|----------|
| "Isn't 95% accuracy just detecting writing style?" | Possible — that's why we need negative controls and confound ablation. But even if partially stylistic, the probe still detects *something* that correlates with compassionate outputs. |
| "Why does early-layer encoding matter?" | It suggests compassion is a "surface" feature. If fine-tuning only affects late layers, the compassionate behavior might be fragile. |
| "Can't you just use the benchmark?" | Benchmarks measure outputs, not internals. A model could produce compassionate text without internal alignment — this probe would detect that gap. |
| "How does this help CaML?" | CaML fine-tunes for compassion. This probe can validate whether training actually shifts representations, or just teaches the model to say the right things. |

---

## 1. Approach: Contrastive Pairs

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

## 2. Results

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

## 3. Operationalization + Caveats

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

## 4. Next Steps

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

*Full details: `docs/presentation.md` | Roadmap: `roadmap.md`*
