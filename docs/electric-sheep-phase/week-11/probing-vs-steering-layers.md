# Probing vs Steering: Optimal Layer Selection in LLMs

## Summary

A critical question in mechanistic interpretability: **Are the optimal layers for probing (detecting a concept) the same as the optimal layers for steering (inducing a concept)?**

**Answer: No — they often differ significantly.**

This document summarizes research findings on why these differ and what it means for practical applications like the compassion steering project.

---

## Key Distinction: Probing Reads, Steering Writes

| Aspect | Probing | Steering |
|--------|---------|----------|
| **Operation** | Reads information passively | Writes information actively |
| **Goal** | Detect if a concept is present | Induce a behavior change |
| **Optimal depth** | Later layers (~60-80% depth) | Middle layers (~40-50% depth) |
| **Why** | Benefits from fully processed representations | Needs computational room for effect to propagate |

---

## Why Optimal Layers Differ

### Probing Layer Selection

**Pattern:** Probing accuracy generally increases with depth.

- Later layers contain more processed, abstract representations
- Information becomes more linearly accessible at deeper layers
- The model has "decided" what concepts are present by later layers

**Exception:** For Llama 8B compassion probing, Layer 8 (25% depth) was optimal — suggesting compassion may be encoded as a relatively "surface" feature that gets transformed or distributed in later layers.

### Steering Layer Selection

**Pattern:** Steering effectiveness peaks at middle layers (~40-50% depth).

- **Too early (layers 1-10):** Modifying low-level features that don't sufficiently impact high-level behavior
- **Too late (layers 25+):** Insufficient computational space remaining for the steering signal to propagate and affect outputs
- **Sweet spot:** Layers 14-17 for 32-layer models (~44-53% depth)

### Theoretical Explanation

From the Contrastive Activation Addition (CAA) paper:

> "Intervening at intermediate layers is particularly effective due to latent representations being in their most abstract and modifiable form at that point. Earlier and later in the transformer, representations are closer to token space."

This suggests middle layers represent concepts in their most "pure" form — abstract enough to capture high-level meaning, but with enough downstream computation to influence outputs.

---

## Research Sources

### 1. Steering Llama 2 via Contrastive Activation Addition (CAA)
**[arXiv:2312.06681](https://arxiv.org/html/2312.06681v2)**

- Found layers 15-17 most effective for steering in Llama 2 (32 layers)
- Layer selection was determined empirically with post-hoc theoretical explanation
- Demonstrated that steering vectors transfer across different prompts and contexts

### 2. Activation Steering Field Guide
**[Subhadip Mitra's Blog](https://subhadipmitra.com/blog/2025/steering-vectors-agents/)**

- Layer 14 consistently optimal across multiple 7-9B parameter models
- Tested multiple behaviors (refusal, helpfulness, etc.)
- Provides practical guidance for steering vector implementation

### 3. CogSteer: Cognition-Inspired Selective Layer Intervention
**[arXiv:2410.17714](https://arxiv.org/html/2410.17714v2)**

- Uses cognitive science insights to guide layer selection
- Confirms middle-bucket layers are optimal for interventions
- Proposes theoretical framework for understanding layer roles

---

## Implications for Compassion Project

| Finding | Implication |
|---------|-------------|
| Probing optimal at layer 8 (25%) | Compassion may be encoded as a "surface" or early-processed feature |
| Steering typically optimal at ~45% depth | Steering may require layer 14-15 for Llama 8B (32 layers) |
| Probing and steering layers may not match | Cannot assume best probe layer = best steering layer |
| Both require empirical testing | Need to test steering at multiple candidate layers |

### Specific Recommendations

1. **Don't assume Layer 8 is optimal for steering** — While it's best for probing compassion, steering likely needs a later layer
2. **Test candidate layers 12-18** — This covers the typical "sweet spot" range
3. **Measure behavioral change, not just activation change** — Use AHB scores or similar metrics

---

## Methodology: Determining Optimal Layers

### For Probing (Classification)

1. Extract activations at each layer for labeled dataset
2. Train linear probe (logistic regression) at each layer
3. Evaluate with held-out test set (accuracy, AUROC)
4. Select layer with best classification performance

### For Steering (Intervention)

1. Extract steering vector at candidate layers (e.g., mean difference method)
2. Apply intervention during inference at each candidate layer
3. Measure behavioral change:
   - Generate completions with/without steering
   - Score outputs (e.g., AHB compassion score)
   - Calculate effect size (score delta)
4. Select layer with largest desired behavioral effect

### Key Point

**Both are empirically determined.** No purely theoretical method exists for predicting optimal layers without experimentation. The patterns described above (probing: later layers; steering: middle layers) are heuristics derived from empirical studies, not guarantees.

---

## Summary Table

| Method | Reads/Writes | Optimal Depth | Why | How to Find |
|--------|--------------|---------------|-----|-------------|
| **Probing** | Reads | ~60-80% | More processed representations | Train probes, compare accuracy |
| **Steering** | Writes | ~40-50% | Needs propagation room | Apply interventions, measure behavior |

---

## References

1. Rimsky, N., et al. (2023). *Steering Llama 2 via Contrastive Activation Addition.* arXiv:2312.06681
2. Mitra, S. (2025). *A Field Guide to Activation Steering.* https://subhadipmitra.com/blog/2025/steering-vectors-agents/
3. Zhang, Y., et al. (2024). *CogSteer: Cognition-Inspired Selective Layer Intervention for Efficient Semantic Steering in Large Language Models.* arXiv:2410.17714
