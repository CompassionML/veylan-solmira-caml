# Activation Steering Diagnostic Checklist

Use this checklist to diagnose why steering experiments might be failing and determine if steering is viable for your use case.

---

## Pre-Flight Checks

Before attempting any steering, answer these questions:

### Model & Infrastructure
- [ ] What model are you using? (Name, size, architecture)
- [ ] What is the hidden dimension? (e.g., 4096 for 7B, 8192 for 70B)
- [ ] Are you using an SAE? If so, which one and what width?
- [ ] What layer(s) are you targeting?

### Concept & Data
- [ ] What concept/behavior are you trying to steer?
- [ ] Do you have contrastive pairs (positive/negative examples)?
- [ ] How many pairs? (Minimum 500 recommended)
- [ ] Are pairs semantically close (tight phrasing)?

### Baseline Measurement
- [ ] What is the model's current behavior rate on this concept?
- [ ] Is there room to move? (30-70% is ideal, >90% or <10% is problematic)

---

## Phase 1: SAE Quality Check

**Goal**: Verify the SAE can faithfully reconstruct activations.

### Test
```python
# Roundtrip test
original = model.get_activations(prompt, layer)
encoded = sae.encode(original)
decoded = sae.decode(encoded)
cosine_sim = cosine_similarity(original, decoded)
relative_l2_error = torch.norm(original - decoded) / torch.norm(original)
```

### Evaluation
| Cosine Similarity | L2 Error | Assessment |
|-------------------|----------|------------|
| > 0.95 | < 25% | PASS - High confidence |
| 0.90 - 0.95 | 25-40% | MARGINAL - Requires larger effects |
| < 0.90 | > 40% | FAIL - Too noisy for reliable steering |

### If FAIL:
- [ ] Try a larger-width SAE (65k vs 16k)
- [ ] Try earlier layers (they often have better reconstruction)
- [ ] Consider direct activation addition (no SAE)

**Result**: [ ] PASS [ ] MARGINAL [ ] FAIL

---

## Phase 2: Feature Validation

**Goal**: Verify the selected feature fires on your prompts and is causally relevant.

### Test 2a: Feature Activation
```python
# Does feature fire on target prompts?
for prompt in target_prompts:
    activation = model.get_activations(prompt, layer)
    feature_value = sae.encode(activation)[feature_idx]
    print(f"Feature fires: {feature_value > 0}, magnitude: {feature_value:.4f}")
```

**Evaluation**: Feature should fire (>0) on most target prompts.

- [ ] Feature fires on >80% of target prompts
- [ ] Average magnitude is meaningful (not near-zero)

### Test 2b: Ablation Test
```python
# Does zeroing the feature change behavior?
original_output = model.generate(prompt)
ablated_output = model.generate(prompt, ablate_feature=feature_idx)
behavior_change = measure_difference(original_output, ablated_output)
```

**Evaluation**: Ablating should produce measurable behavior change.

- [ ] Behavior changes when feature is ablated
- [ ] Change is in expected direction

### If FAIL:
- [ ] Try different feature selection method
- [ ] Use features that fire more broadly
- [ ] Consider the feature may be "input-relevant" not "output-relevant"

**Result**: [ ] PASS [ ] FAIL

---

## Phase 3: Baseline Assessment

**Goal**: Verify there's room for steering to have an effect.

### Test
```python
# Measure current behavior rate
baseline_rate = measure_behavior(model, test_prompts)
print(f"Baseline behavior rate: {baseline_rate:.2%}")
```

### Evaluation
| Baseline Rate | Assessment |
|---------------|------------|
| 30-70% | IDEAL - Room to move both directions |
| 10-30% or 70-90% | MARGINAL - Limited room in one direction |
| <10% or >90% | PROBLEMATIC - Saturated, little room to move |

### If PROBLEMATIC:
- [ ] Try steering in the opposite direction
- [ ] Select a different concept that's less saturated
- [ ] Accept that steering may show limited effect

**Result**: [ ] IDEAL [ ] MARGINAL [ ] PROBLEMATIC

---

## Phase 4: Scale Sweep

**Goal**: Find the effective steering scale range.

### Test
```python
scales = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
results = []
for scale in scales:
    steered_rate = measure_behavior(model, test_prompts, steering_scale=scale)
    fluency = measure_fluency(model, test_prompts, steering_scale=scale)
    results.append({
        'scale': scale,
        'behavior_rate': steered_rate,
        'fluency': fluency,
        'delta': steered_rate - baseline_rate
    })
```

### Evaluation
Look for:
- [ ] Monotonic increase in target behavior at low scales
- [ ] Peak effect at some scale (often 1.5-3x)
- [ ] Fluency degradation at high scales

**Effective range found**: Scale _____ to _____

### If No Effect at Any Scale:
- [ ] Feature may not be causally relevant to output
- [ ] Try different layers
- [ ] Try additive instead of multiplicative (or vice versa)

**Result**: [ ] Found effective range [ ] No meaningful effect

---

## Phase 5: Control Experiments

**Goal**: Verify the effect is real and not an artifact.

### Test 5a: Random Feature Control
```python
# Steer with random feature - should show no/minimal effect
random_effect = steer_with_random_feature(scale=best_scale)
target_effect = steer_with_target_feature(scale=best_scale)
assert target_effect > random_effect * 2, "Effect may be noise"
```

### Test 5b: Roundtrip-Only Control
```python
# Encode → decode without steering - should show minimal change
roundtrip_effect = encode_decode_without_steering()
assert target_effect > roundtrip_effect * 2, "Effect may be SAE artifact"
```

### Evaluation
- [ ] Target effect significantly exceeds random feature effect
- [ ] Target effect significantly exceeds roundtrip-only effect

**Result**: [ ] PASS (real effect) [ ] FAIL (may be artifact)

---

## Decision Tree

Based on your results:

```
Phase 1 FAIL (SAE quality < 0.90)
    → Try larger SAE, earlier layer, or direct activation addition

Phase 2 FAIL (feature doesn't fire or no ablation effect)
    → Try different feature selection, concept may not be linearly represented

Phase 3 PROBLEMATIC (baseline saturated)
    → Limited room for improvement, try different concept

Phase 4 FAIL (no effect at any scale)
    → Feature may not be output-relevant, consider probing instead

Phase 5 FAIL (effect same as controls)
    → Effect is likely noise/artifact, not real steering

ALL PASS → Steering is viable! Optimize for your use case.
```

---

## When to Pivot to Probing

Consider pivoting if:
- [ ] Multiple phases fail
- [ ] Time investment exceeds expected value
- [ ] Detection alone would provide sufficient value for your application
- [ ] The concept appears fundamentally unsteerable

**Probing advantages**:
- More reliable and well-established
- Better respected in the field
- Provides detection without causal intervention
- Simpler to implement and validate

---

## Quick Reference: Common Failure Modes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| No effect at any scale | Feature not output-relevant | Try different feature or layer |
| Reverse effect | Anti-steerable concept | Use negative steering or different approach |
| Effect but high variance | Feature redundancy | Model routes around; try ablation |
| Fluency degrades badly | Steering too strong | Use lower scale, later layer |
| Works on some prompts only | Input-dependent steerability | Normal; document limitations |
| SAE roundtrip poor | Wrong layer or SAE | Try earlier layer, larger SAE |
