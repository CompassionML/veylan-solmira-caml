# When Does Activation Steering Work? A Diagnostic Framework

A diagnostic framework for understanding when SAE-based activation steering will and won't work.

---

## Overview

Activation steering via Sparse Autoencoders (SAEs) is a technique for causally intervening on model behavior by scaling specific learned features during inference. However, steering often fails or produces negligible effects. This document provides a diagnostic framework.

---

## Why SAE Reconstruction Quality Matters

### The Roundtrip Test

The roundtrip test measures SAE reconstruction fidelity:

```
original_activations → SAE.encode() → sparse_features → SAE.decode() → reconstructed_activations
```

**Metric:** Cosine similarity between original and reconstructed activations.

### Why This Matters for Steering

When doing SAE-based activation steering, the pipeline is:

1. Get activations at layer L
2. Encode with SAE to get sparse features
3. **Modify a specific feature** (scale it up/down)
4. Decode back to activation space
5. Replace original activations with modified ones

**Critical insight:** If the SAE can't faithfully reconstruct activations (low cosine similarity), then even *without* any steering intervention, you're injecting significant noise into the residual stream. Any behavioral change could be from that noise rather than from your targeted intervention.

### Analogy

It's like trying to edit a photo by:
1. Converting to JPEG at 10% quality
2. Making your edit
3. Converting back

The compression artifacts overwhelm your actual edit. You can't distinguish "edit effect" from "compression noise."

### Empirical Results (GemmaScope on Gemma 2 2B)

| Layer | Cosine Similarity | Relative L2 Error | Pass (>0.95)? |
|-------|-------------------|-------------------|---------------|
| 0 | 0.973 | 23.8% | **PASS** |
| 4 | 0.929 | 37.8% | FAIL |
| 15 | 0.922 | 45.9% | FAIL |

**Interpretation:** Only Layer 0 has sufficient reconstruction quality for reliable steering. Layers 4 and 15 introduce ~35-45% noise, confounding any intervention.

### On the 0.95 Threshold

**Where does 0.95 come from?**

This threshold is a heuristic, not a rigorously derived cutoff:

1. **Convention from SAE literature** - Papers like Templeton et al. report cosine similarities and treat >0.95 as "good reconstruction." This is more descriptive than prescriptive.

2. **Signal-to-noise intuition** - At 0.95 cosine similarity, ~95% of directional information is preserved. If your intervention is a 10% feature scaling, you want noise well below that.

3. **No formal derivation** - No paper rigorously proves "0.95 is where steering fails." It's a rule of thumb.

**Practical interpretation by reconstruction quality:**

| Cosine Sim | Interpretation |
|------------|----------------|
| >0.98 | Excellent - intervention effects are clean |
| 0.95-0.98 | Good - can detect moderate effects |
| 0.90-0.95 | Marginal - need large effects, careful controls |
| 0.85-0.90 | Poor - noise may dominate, results questionable |
| <0.85 | Problematic - probably not useful for steering |

**Below 0.95 is not "impossible," it's "confounded":**

At 0.92 (Layer 15), steering is still possible but:
- Need larger effect sizes to distinguish signal from noise
- Need control experiments (random feature steering, roundtrip-only)
- Results are harder to interpret cleanly
- Negative results become ambiguous (was it the feature or the reconstruction?)

**Recommendation:** For exploratory work, proceed with features in marginal layers (0.90-0.95) but include appropriate controls and note the reconstruction quality as a limitation. For claims requiring high confidence, stick to layers with >0.95 reconstruction.

---

## Prerequisites for Successful Steering

### 1. SAE Quality Prerequisites

| Check | What to Measure | Threshold | Why It Matters |
|-------|-----------------|-----------|----------------|
| **Reconstruction quality** | Cosine similarity of roundtrip | >0.95 | Low = noise injection dominates intervention |
| **Feature sparsity** | % of features active per token | <5% typical | Dense = features not interpretable, interventions diffuse |
| **Feature monosemanticity** | Does feature activate on coherent concept? | Manual inspection | Polysemantic features = intervention has unpredictable effects |

### 2. Feature Selection Prerequisites

| Check | What to Measure | How to Test |
|-------|-----------------|-------------|
| **Feature actually fires** | Activation magnitude on your prompts | Run prompts, check if target feature activates (magnitude > 0) |
| **Feature is causally relevant** | Does ablating it change behavior? | Zero out feature, measure behavioral change |
| **Feature isn't redundant** | Is information encoded elsewhere? | Ablate feature, check if model recovers via other paths |

### 3. Intervention Design Factors

| Factor | Issue | Solution |
|--------|-------|----------|
| **Scale magnitude** | Too small = no effect, too large = incoherent output | Sweep scales (0.5x, 1.5x, 2x, 3x, 5x), find sweet spot |
| **Layer choice** | Early layers = more abstract, later = more concrete | Try multiple layers for same concept |
| **When to intervene** | All tokens? Last token? Specific positions? | Experiment with intervention timing |
| **Baseline behavior** | If model already does X, amplifying X-features may saturate | Need behavioral room to move |

### 4. Measurement Prerequisites

| Check | Issue | Solution |
|-------|-------|----------|
| **Behavioral sensitivity** | Your metric may not capture the change | Use multiple metrics (perplexity, specific outputs, classifier) |
| **Statistical power** | Small effects need large N | Run enough samples to detect 10% changes |
| **Correct counterfactual** | Comparing to wrong baseline | Compare steered vs. unsteered on same prompts |

---

## Common Failure Modes

### Failure Mode 1: Poor SAE Quality

**Symptom:** Any intervention (including random feature scaling) changes behavior similarly.

**Diagnosis:** Run roundtrip test. If cosine sim <0.9, this is likely the culprit.

**Solution:**
- Use better SAE (larger width, e.g., 65k instead of 16k)
- Try different layer with better reconstruction
- Don't use SAE-based steering; use direct activation addition instead

### Failure Mode 2: Feature Doesn't Fire

**Symptom:** Scaling feature has no effect.

**Diagnosis:** Check if feature actually activates on your prompts. If activation magnitude is 0, scaling 0 × N is still 0.

**Solution:**
- Find a feature that actually fires on your inputs
- Use additive steering (add constant activation) rather than multiplicative (scale existing activation)
- Verify feature interpretation matches your use case

### Failure Mode 3: Feature is Redundant

**Symptom:** Ablating feature has no effect on behavior.

**Diagnosis:** Information is encoded across many features (distributed representation). The model routes around your intervention.

**Solution:**
- Try steering multiple correlated features together
- Look for "hub" features that aggregate information
- Accept that not all behaviors are steerable via single features

### Failure Mode 4: Wrong Layer

**Symptom:** Feature fires but intervention doesn't change final output.

**Diagnosis:**
- Later layers may "correct" for your intervention
- Earlier layers may be too abstract to affect specific behaviors

**Solution:**
- Try steering at multiple layers
- Check if layers L+1, L+2 undo your intervention
- Consider steering at multiple layers simultaneously

### Failure Mode 5: Saturated Behavior

**Symptom:** Model already exhibits the behavior; amplifying feature doesn't increase it.

**Diagnosis:** Check baseline behavior rate. If it's already 95%, there's no room to push higher.

**Solution:**
- Test in conditions where baseline behavior is intermediate (30-70%)
- Try suppressing (scaling down) instead of amplifying
- Use prompts that elicit more ambiguous model behavior

### Failure Mode 6: Scale Too Small or Too Large

**Symptom:** No effect at 1.5x, gibberish at 3x, nothing useful in between.

**Diagnosis:** The "effective range" for this feature may be very narrow or non-existent.

**Solution:**
- Fine-grained scale sweep (1.1x, 1.2x, 1.3x, etc.)
- Try logarithmic scales
- Accept that this particular feature may not be steering-compatible

### Failure Mode 7: Wrong Intervention Type

**Symptom:** Multiplicative scaling has no effect even when feature fires.

**Diagnosis:** Feature may need additive intervention, or intervention at generation time vs. prompt encoding time.

**Solution:**
- Try additive steering: `activation[feature] += constant` instead of `*= scale`
- Try clamping: `activation[feature] = max(activation[feature], threshold)`
- Intervene during generation, not just during prompt processing

---

## Diagnostic Protocol

Before concluding "steering doesn't work," run this checklist:

### Phase 1: SAE Quality Verification

```
[ ] Roundtrip cosine similarity > 0.95 at target layer?
[ ] Sparsity reasonable (<5% features active per token)?
[ ] Reconstruction error consistent across different prompts?
```

### Phase 2: Feature Validation

```
[ ] Target feature activates on your prompts? (magnitude > 0)
[ ] Feature interpretation makes sense for your task? (check Neuronpedia)
[ ] Ablation test: does zeroing feature change anything?
[ ] Feature activates consistently across similar prompts?
```

### Phase 3: Intervention Design

```
[ ] Tried multiple scales? (0x, 0.5x, 1x, 1.5x, 2x, 3x, 5x, 10x)
[ ] Tried additive vs multiplicative steering?
[ ] Tried different intervention positions? (all tokens vs last token)
[ ] Tried different layers for same concept?
```

### Phase 4: Measurement Validation

```
[ ] Baseline behavior is in moveable range (not 0% or 100%)?
[ ] Sufficient sample size for statistical power (n > 50)?
[ ] Multiple behavioral metrics checked?
[ ] Effect size estimated with confidence intervals?
```

### Phase 5: Control Experiments

```
[ ] Random feature steering doesn't produce same effect?
[ ] Roundtrip-only (encode→decode, no steering) doesn't produce same effect?
[ ] Effect is specific to target feature, not general perturbation?
```

---

## Debugging Questions

When someone reports "steering had very little effect," ask:

1. **What was the roundtrip quality?**
   - If they didn't check, this is the most likely culprit.

2. **Did the target feature actually activate?**
   - Many people pick features from Neuronpedia based on description but don't verify the feature fires on their specific prompts.

3. **What scales did they try?**
   - 1.5x might be too subtle. 10x might work but they stopped at 3x.

4. **What layer?**
   - Early layer features often don't propagate to output. Late layer features may have better causal influence but worse reconstruction.

5. **What was the baseline behavior?**
   - If the model already does the behavior 90% of the time, there's nowhere to go.

6. **Did they try additive steering?**
   - Multiplicative steering fails if the feature doesn't fire. Additive steering injects signal regardless.

7. **What was the sample size?**
   - n=10 is not enough to detect a 15% effect. Need n>50 typically.

---

## Options for Improving Reconstruction Quality

If reconstruction quality is insufficient at your target layer, consider:

### Option 1: Test More Layers
Reconstruction quality varies by layer. If Layer 15 is poor, layers 1-3 might still be good.

### Option 2: Use Larger-Width SAE
GemmaScope provides both 16k and 65k width SAEs. Larger width = more capacity = potentially better reconstruction. Trade-off: more features to search through.

### Option 3: Find Equivalent Features in Earlier Layers
If you need specific features but a later layer has poor reconstruction, search for similar concepts in earlier layers where reconstruction is better.

### Option 4: Accept Noise and Add Controls
Run experiments anyway with appropriate controls:
- Steered vs unsteered (baseline)
- Steered vs random-feature-steered (specificity control)
- Steered vs roundtrip-only (reconstruction noise control)

If target steering differs from all controls, that's evidence it works despite noise.

### Option 5: Use Direct Activation Addition
Skip SAE entirely - compute steering vector from contrastive examples and add directly to activations. No reconstruction error, but less interpretable.

---

## Alternative Approaches When SAE Steering Fails

### 1. Direct Activation Addition (No SAE)

Instead of using SAE features, directly add a "steering vector" to activations:

```python
steering_vector = mean(activations_positive_examples) - mean(activations_negative_examples)
steered_activation = original_activation + scale * steering_vector
```

**Pros:** No SAE reconstruction error
**Cons:** Less interpretable, steering vector may be entangled

### 2. Contrastive Activation Addition (CAA)

Compute steering vector from contrastive pairs:

```python
# Positive: prompts that elicit desired behavior
# Negative: prompts that elicit opposite behavior
steering_vector = mean(pos_activations) - mean(neg_activations)
```

**Reference:** Rimsky et al., "Steering Llama 2 via Contrastive Activation Addition"

### 3. Representation Engineering

Train a probe to identify directions in activation space, then steer along those directions.

### 4. Fine-tuning

If steering doesn't work, the behavior may require weight updates rather than activation interventions.

---

## Quick Reference Card

```
STEERING VIABILITY CHECKLIST
============================

□ SAE roundtrip cosine sim > 0.95?
□ Target feature fires on your prompts?
□ Ablating feature changes behavior?
□ Baseline behavior in moveable range?
□ Tried scales from 0x to 10x?
□ Sample size > 50?
□ Random feature control is negative?

If all YES → steering should work
If any NO → investigate that item first
```

---

## References

- Templeton et al. (2024) — "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"
- Lieberum et al. (2024) — "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2"
- Rimsky et al. (2023) — "Steering Llama 2 via Contrastive Activation Addition"
- Turner et al. (2023) — "Activation Addition: Steering Language Models Without Optimization"
- Conmy et al. (2023) — "Towards Automated Circuit Discovery for Mechanistic Interpretability"
