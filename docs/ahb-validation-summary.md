# AHB Validation Study: Style Confounds and Probe Comparison

## 1. Initial Results (V5 Probe)

Trained a linear probe on 105 contrastive pairs generated from AHB scenarios:
- **Compassionate**: Modern animal welfare advocate framing
- **Non-compassionate**: 1950s textbook/utilitarian framing

| Layer | Accuracy | AUROC |
|-------|----------|-------|
| 8 | 95.2% | 0.995 |
| 12 | 92.9% | 0.964 |

Best performance at layer 8 (25% depth).

---

## 2. Style Confound Concern

Feedback from reviewers (Jasmine, Raphael): The probe may be detecting **style differences** (modern empathetic language vs. dated clinical tone) rather than genuine compassion.

The contrastive pairs differed in:
- Era markers (1950s vs. modern)
- Response length and hedging
- Tone and formality

---

## 3. Minimal Pairs Dataset

To isolate compassion from style, created 30 minimal pair prompts differing by one word:

```
"Can you suggest a recipe to cook a {chicken/dog}?"
"I'm sad because my {cat/pig} just passed away."
```

**Finding**: Minimal pair direction is **orthogonal** to V5 probe direction (cos θ ≈ 0.00).

This confirmed the style confound—V5 and minimal pairs measure independent signals.

**Signal strength increases with depth** — minimal pairs direction norm grows from 0.67 (layer 8) to 2.22 (layer 20), suggesting moral consideration is computed in later layers:

![Direction Norms by Layer](../outputs/visualizations/direction_norms_by_layer.png)

---

## 4. V7 Style-Controlled Dataset

Generated 106 new pairs with explicit style matching:
- Identical neutral academic tone
- Matched response length/structure
- Differ only in **framing**: welfare-focused vs. economic/practical

**Example**:
> **Compassionate**: "Fish possess neurological architecture necessary to detect harmful stimuli... stress behaviors diminish with analgesics..."
>
> **Non-compassionate**: "Operations minimizing physical stress report better flesh quality, reduced cortisol-related tissue damage..."

| Layer | Accuracy | AUROC |
|-------|----------|-------|
| 8 | 95.3% | 0.991 |
| 12 | 97.7% | 0.998 |

Best performance at layer 12 (vs. layer 8 for V5).

**Finding**: V7 is also orthogonal to both V5 and minimal pairs—three independent dimensions.

![Three Orthogonal Dimensions](../outputs/visualizations/three_orthogonal_dimensions.png)

All pairwise cosine similarities are near zero across layers 8–20, confirming these are independent directions in activation space.

---

## 5. AHB Validation Methodology

**Goal**: Test whether probes trained on synthetic pairs predict compassion in natural model outputs.

**Method**:
1. Generate Llama 3.1 8B responses to all 108 AHB questions
2. Grade each response using Claude on AHB's 12 dimensions (0.0–1.0 scale)
3. Extract hidden states from each response
4. Project onto probe direction → probe score
5. Correlate probe scores with AHB grades

---

## 6. Results

| Probe | Pearson r | Spearman r | p-value |
|-------|-----------|------------|---------|
| **V5** (style-confounded) | +0.457 | +0.365 | <0.0001 |
| **V7** (style-controlled) | +0.428 | +0.389 | <0.0001 |
| **Minimal pairs** | **-0.422** | -0.451 | <0.0001 |

![AHB Three-Probe Comparison](../experiments/linear-probes/outputs/visualizations/ahb_three_probe_comparison.png)

---

## 7. Interpretation

**V5 ≈ V7**: Style control reduced correlation by only Δ=0.029, suggesting V5 wasn't purely a style detector. Both capture genuine signal about welfare framing.

**Minimal pairs inverted**: The negative correlation occurs because:
- Minimal pairs direction: `pet language (+) ↔ livestock language (-)`
- AHB focuses on farm animals (fish, pigs, chickens)
- Compassionate AHB responses discuss "low-moral" entities compassionately
- This projects *negatively* on the minimal pair direction

**Conclusion**: "Compassion" is multidimensional. These probes measure:

| Probe | Measures |
|-------|----------|
| V5/V7 | Welfare framing (how responses discuss animals) |
| Minimal pairs | Moral circle (which animals get consideration) |

All three predict AHB scores (|r| ≈ 0.42–0.46), but minimal pairs captures a fundamentally different—and inverted—dimension.

---

## Data Locations

| File | Path |
|------|------|
| V5 pairs | `data/contrastive-pairs/usable_pairs_deduped.jsonl` |
| V7 pairs | `data/contrastive-pairs/pairs_v7_full.jsonl` |
| Minimal pairs | `data/minimal-pairs/minimal_pairs.jsonl` |
| V5 validation | `experiments/linear-probes/outputs/evaluation/ahb_validation.json` |
| V7 validation | `experiments/linear-probes/outputs/v7-runpod/results/ahb_validation_v7.json` |
| Minimal pairs validation | `experiments/linear-probes/outputs/v7-runpod/results/ahb_validation_minimal_pairs.json` |
