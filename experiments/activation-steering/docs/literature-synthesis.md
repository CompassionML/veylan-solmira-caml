# Activation Steering Literature Synthesis

A comprehensive synthesis of research on activation steering, when it works, when it fails, and practical guidance.

---

## Executive Summary

**Core Finding**: There is a fundamental gap between detection (probing) and steering (intervention). Detection works reliably (0.93+ AUROC), but steering is highly variable and fails under many conditions.

**Research Consensus**:
- Steering works under "very precise conditions" that are often impractical
- Detection/probing is more reliable and better-respected
- However, steering CAN work with proper conditions (see PersonaVectors example)

---

## Key Papers and Findings

### Contrastive Grounding for Hallucination Detection

**Key Method**: Compute "truthfulness direction" as normalized mean difference between factual and hallucinated hidden states:
```
v = (μ_F - μ_H) / ||μ_F - μ_H||
```

**Detection Results** (Mistral-7B):
- Single layer AUROC: 0.82-0.87
- 4-layer ensemble: 0.933 AUROC
- Cross-dataset transfer: 0.925 mean AUROC

**Critical Finding on Steering**:
> "Causal intervention (adding λ·v to hidden states during generation) did not yield truthfulness gains on TruthfulQA"

**Why Detection Works but Steering Doesn't**:
- The direction optimizes for *separating* representations, not *steering* generation
- High baseline truthfulness leaves limited room for improvement
- Different layers encode different aspects (syntax → semantics → output)

**Recommended Calibration**: 500-1000 balanced pairs

---

### PersonaVectors — A Success Case

**Critical**: This example shows steering CAN work under the right conditions.

**Setup**:
- Model: Llama-3.1-70B-Instruct (4-bit quantization)
- Concept: Compassion
- Method: Forward hooks at specific layer

**Results**:
- Baseline compassion score: 3.33/5
- Steered compassion score: 4.33/5
- **+1.0 point improvement**

**Success Factors**:
1. **Systematic layer selection**: Tested layers 8-14, selected Layer 9
2. **External evaluation**: Different model as judge (not self-evaluation)
3. **Forward hooks**: At specific layer, not global steering
4. **Temperature control**: 0.3 for stability
5. **Large model**: 70B with proper layer targeting

**Code Pattern**:
```python
def steering_hook(module, input, output):
    hidden_states = output[0] if isinstance(output, tuple) else output
    vector_tensor = torch.tensor(vector, dtype=hidden_states.dtype, device=hidden_states.device)
    steered_hidden_states = hidden_states + coefficient * vector_tensor
    return (steered_hidden_states,) + output[1:] if isinstance(output, tuple) else steered_hidden_states
```

---

### SAE Quality Requirements

**Empirical Results** (GemmaScope on Gemma 2 2B):

| Layer | Cosine Similarity | Relative L2 Error | Pass (>0.95)? |
|-------|-------------------|-------------------|---------------|
| 0 | 0.973 | 23.8% | **PASS** |
| 4 | 0.929 | 37.8% | FAIL |
| 15 | 0.922 | 45.9% | FAIL |

**Diagnostic Protocol**:
1. SAE roundtrip test (cosine similarity)
2. Feature activation check (does it fire?)
3. Ablation test (causal relevance)
4. Scale sweep (0.5x to 10x)
5. Control experiments (random feature, roundtrip-only)

**Failure Modes Identified**:
- Poor SAE quality
- Feature doesn't fire
- Feature redundancy (routing around)
- Wrong layer
- Saturated baseline
- Wrong intervention type (multiplicative vs additive)

---

## Key Papers from Published Research

### Steering Llama 2 via Contrastive Activation Addition (Rimsky et al., 2024)
- ACL 2024 publication
- Demonstrates CAA method
- Limitations: Not robust across inputs, many concepts unsteerable

### A Sober Look at Steering Vectors for LLMs (Alignment Forum)
- Comprehensive evaluation of steering reliability
- Finding: Steerability highly variable across inputs
- Some concepts are "anti-steerable" (reverse effect)

### Improving Steering Vectors by Targeting SAE Features (2024)
- SAE-Targeted Steering (SAE-TS)
- Measures causal effect of steering vectors
- Minimizes unintended side effects

### SAEs Are Good for Steering – If You Select the Right Features (2025)
- Distinction between "input features" and "output features"
- Input features capture patterns but don't affect output
- Output features have human-understandable effect on generation

### Use SAEs to Discover Unknown Concepts, Not to Act on Known Concepts (2025)
- Position paper reconciling conflicting narratives
- SAEs underperform baselines on known concept detection
- SAEs valuable for *discovering* unknown concepts

---

## The Detection-Steering Gap

### Why Detection Works
- Linear separability in representation space
- Contrastive directions maximize d' (sensitivity index)
- Strong mathematical foundation (optimal under Gaussian assumption)

### Why Steering Often Fails
1. **Different requirements**: Separating ≠ Controlling
2. **Feature redundancy**: Model routes around interventions
3. **Input vs output features**: Detection features may not affect generation
4. **Layer dynamics**: Later layers can "correct" for interventions
5. **Baseline effects**: Limited room if behavior already present

### Theoretical Insight
The optimal direction for *detecting* a concept is not necessarily the optimal direction for *inducing* that concept. Detection optimizes for separation; steering requires causal influence on the generation process.

---

## When Steering Works (Conditions)

### Model Requirements
- Larger models show stronger signals (70B > 7B)
- SAE reconstruction quality > 0.95
- Proper layer selection (often middle layers, 50-75% depth)

### Feature Requirements
- Feature must actually fire on target prompts
- Feature must be causally relevant (ablation changes behavior)
- Feature should be "output-relevant" not just "input-relevant"

### Behavioral Requirements
- Baseline behavior in 30-70% range (room to move)
- Concept must be linearly represented
- Not saturated (not already at ceiling/floor)

### Intervention Requirements
- Careful scale selection (sweet spot often narrow)
- Late-layer intervention for fluency preservation
- Additive may work where multiplicative fails (or vice versa)

---

## When Steering Fails (13 Failure Modes)

### Technical Failures
1. Poor SAE quality (reconstruction noise overwhelms intervention)
2. Feature doesn't fire (scaling 0 × N = 0)
3. Feature redundancy (model routes around)
4. Wrong layer (later layers correct for intervention)
5. Anti-steerability (reverse effect)

### Methodological Failures
6. Wrong intervention type (multiplicative vs additive)
7. Hyperparameter sensitivity (layer/scale selection)
8. Fluency degradation (excessive strength)
9. Multi-property failure (vector composition fails)
10. Transferability limits (doesn't generalize)

### Fundamental Limitations
11. Detection ≠ Causation
12. Input vs output features
13. High baseline problem (no room to improve)

---

## Practical Recommendations

### When to Use Steering
- Simple, human-interpretable concepts (sentiment, style)
- With systematic layer selection and external evaluation
- When you can verify causality via ablation
- With large models and high-quality SAEs

### When to Use Probing Instead
- When detection alone provides sufficient value
- When steering consistently fails diagnostics
- For safety-critical applications requiring reliability
- When time investment exceeds expected value

### Best Practices
1. Run full diagnostic protocol before committing to steering
2. Use external evaluation (different model as judge)
3. Test multiple layers and intervention types
4. Include proper controls (random feature, roundtrip-only)
5. Document which conditions/prompts work vs fail

---

## References

### Papers
- Rimsky et al. (2024) — "Steering Llama 2 via Contrastive Activation Addition" (ACL 2024)
- Templeton et al. (2024) — "Scaling Monosemanticity"
- Lieberum et al. (2024) — "Gemma Scope: Open Sparse Autoencoders"

### Code & Notebooks
- [nrimsky/CAA](https://github.com/nrimsky/CAA) - Original CAA implementation
- [IBM/activation-steering](https://github.com/IBM/activation-steering) - IBM's implementation

### Community Resources
- [A Sober Look at Steering Vectors](https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/a-sober-look-at-steering-vectors-for-llms)
- [SAE-Targeted Steering](https://arxiv.org/abs/2411.02193)
