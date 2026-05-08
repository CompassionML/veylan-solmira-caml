# CaML Research: Linear Probes for Compassion Measurement

**Project:** Mechanistic interpretability-grounded measurement of compassion in LLMs
**Date:** February 26, 2026
**Status:** Multi-layer experiment complete

---

## Executive Summary

We are developing linear probes to measure compassion in LLM activations, supplementing the Animal Harm Benchmark (AHB) with internal representation analysis. This document summarizes infrastructure setup, methodology, and initial results.

**Key Result:** A linear probe trained on layer 8 of Llama 3.1 8B achieves **95.2% accuracy** and **0.995 AUROC** distinguishing compassionate from non-compassionate responses — optimal layer is at 25% depth, contradicting the ~75% depth heuristic from the literature.

---

## 1. Infrastructure

### 1.1 Compute Environment

| Component | Specification |
|-----------|---------------|
| Platform | StrongCompute Sydney |
| GPU | NVIDIA RTX 3090 Ti (24GB VRAM) |
| Container | `caml-probes-2026-02` |
| Model | Llama 3.1 8B Instruct (~16GB VRAM) |
| Headroom | ~8GB for activations/batching |

### 1.2 Docker Images

Two-tier Docker architecture for efficient builds:

```
┌─────────────────────────────────────┐
│  veylansolmira/caml-base:latest     │  ← CUDA + PyTorch + Flash Attention
│  (Built once, ~15 min)              │
└─────────────────────────────────────┘
                 ↑
┌─────────────────────────────────────┐
│  veylansolmira/caml-env:latest      │  ← ML packages, tools
│  (Builds on base, ~5 min)           │
└─────────────────────────────────────┘
```

| Image | Contents | Size |
|-------|----------|------|
| `caml-base` | CUDA 12.8, Python 3.12, PyTorch 2.10, Flash Attention 2.8.3 | ~15GB |
| `caml-env` | + transformers, sklearn, jupyter, interpretability tools | ~18GB |

**Flash Attention:** Pre-compiled for sm80/90/100/120 (Ampere → Blackwell), hosted on GitHub releases to avoid 90-min compile time.

### 1.3 GitHub Actions CI/CD

Automated Docker builds on push to `infrastructure/docker/`:

- `.github/workflows/docker-build.yml` — Main image (`caml-env`)
- `.github/workflows/docker-build-base.yml` — Base image (manual trigger)

Features:
- Disk space cleanup for large builds
- GitHub token auth for private release assets
- Multi-architecture support (linux/amd64)

### 1.4 Data Backup

HuggingFace model backup automated:

| Item | Status |
|------|--------|
| CaML organization models | 197 models cataloged |
| Backup location | Local + scheduled sync |
| Script | `scripts/hf-backup.py` |

---

## 2. Methodology

### 2.1 Literature Context

Our approach draws on recent activation engineering literature (2025-2026):

| Finding | Source | Our Result |
|---------|--------|------------|
| 80-100 samples optimal | [Patterns and Mechanisms of CAE](https://arxiv.org/html/2505.03189) | Confirmed (105 pairs) |
| ~75% layer depth best | [Activation Steering Field Guide](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/) | **Contradicted** (25% optimal) |

**Current focus:** Linear probing to measure compassion in activations. Steering experiments are future work (see Appendix A).

### 2.2 Training Data

Contrastive pairs generated via Claude with historical framing (v4 prompt):

| Metric | Value |
|--------|-------|
| Total pairs | 105 (pairs_v5_best.jsonl) |
| Success rate | 85% (vs 10% with direct prompting) |
| Format | `{question, compassionate_response, non_compassionate_response}` |

### 2.3 Activation Selection

| Choice | Our Approach |
|--------|--------------|
| **Token selection** | Mean-pool over exact response tokens (excluding user prompt) |
| **Alternatives** | Last token only, last period position, mean over full sequence |

**Note:** Initial experiments used a 50% heuristic (last half of tokens). Updated to compute exact response boundaries — re-extraction pending.

### 2.4 Probe Architecture

```
Input: Hidden state at layer L (4096 dims for Llama 8B)
       ↓
Logistic Regression (with L2 regularization via CV)
       ↓
Output: P(compassionate) ∈ [0, 1]
```

Two direction extraction methods:
1. **Difference-in-means** (CAA-style): `mean(compassionate) - mean(non_compassionate)`
2. **Logistic regression weights** (normalized)

---

## 3. Results

### 3.1 Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | Llama 3.1 8B Instruct |
| Layers tested | 8, 12, 16, 20, 24, 28 (25%–88% depth) |
| Pairs | 105 |
| Train/Test Split | 80/20 |
| Cross-validation | 5-fold |

### 3.2 Multi-Layer Probe Performance

| Layer | Depth | Accuracy | AUROC | CV Accuracy | Dir. Similarity |
|-------|-------|----------|-------|-------------|-----------------|
| **8** | **25%** | **95.2%** | **0.995** | 97.1% ± 2.8% | 0.793 |
| 12 | 38% | 92.9% | 0.964 | 98.1% ± 1.8% | 0.846 |
| 16 | 50% | 90.5% | 0.957 | 98.1% ± 2.8% | 0.871 |
| 20 | 63% | 90.5% | 0.914 | 98.1% ± 2.8% | 0.736 |
| 24 | 75% | 88.1% | 0.909 | 98.1% ± 2.8% | 0.652 |
| 28 | 88% | 88.1% | 0.891 | 97.1% ± 2.3% | 0.572 |

**Key Findings:**
- **Layer 8 (25% depth) is optimal** — strongly contradicting the ~75% depth heuristic
- Accuracy and AUROC decrease monotonically with depth (95.2%→88.1%, 0.995→0.891)
- Direction similarity peaks at layer 16 (0.871) but doesn't correlate with best accuracy
- This suggests compassion is encoded early in the model's representations
- Random label controls all near 50% baseline as expected

### 3.3 Visualizations

#### Layer Comparison

Performance metrics across all tested layers:

![Layer Comparison](figures/layer_comparison.png)

Performance vs layer depth shows monotonic decrease:

![Performance vs Depth](figures/performance_vs_depth.png)

#### Projection Distribution (Layer 8 — Best)

The probe successfully separates compassionate from non-compassionate responses in activation space:

![Projection Distribution](figures/projection_distribution.png)

**d' (discriminability) = 4.52** — Excellent separation between classes (higher than Layer 24's 3.56).

#### ROC Curve (Layer 8)

![ROC Curve](figures/roc_curve.png)

#### Confusion Matrix (Layer 8)

![Confusion Matrix](figures/confusion_matrix.png)

### 3.4 Direction Analysis

The two methods (difference-in-means vs logistic regression) show interesting patterns:

| Layer | DiffMeans-Probe Cosine Similarity |
|-------|-----------------------------------|
| 8 | 0.793 |
| 12 | 0.846 |
| 16 | 0.871 (peak) |
| 20 | 0.736 |
| 24 | 0.652 |
| 28 | 0.572 |

![Direction Similarity](figures/direction_similarity.png)

**Observation:** Direction similarity peaks at layer 16 (0.871), but probe accuracy peaks at layer 8. This divergence suggests that method convergence does not predict discriminative power — the strongest signal is in earlier layers where the methods agree less.

### 3.5 Timing

| Step | Time |
|------|------|
| Model loading | ~15s |
| Extraction (105 pairs × 6 layers) | ~3 min |
| Probe training (6 layers) | ~25 min |
| **Total pipeline** | **~30 min** |

---

## 4. Interpretation

### 4.1 Why Earlier Layers Perform Better

The monotonic decrease in probe accuracy with layer depth is a striking finding that contradicts the ~75% depth heuristic from the literature. Several hypotheses may explain this:

#### Compassion as a "Surface" Feature

The model may encode compassion-relevant signals early — tone, framing, word choice. These stylistic markers are most distinct in early layers before deeper semantic processing blends them with other content.

#### Later Layers Focus on Generation Mechanics

Deeper layers increasingly optimize for next-token prediction. The "compassion signal" may get diluted as the model computes task-specific features for text generation rather than content classification.

#### Shared Semantic Content Drowns Out Differences

Both compassionate and non-compassionate responses answer the same question. Deeper layers may increasingly encode *what* is being said (shared content) rather than *how* it's being said (the compassion difference).

### 4.2 Direction Similarity Paradox

| Layer | Accuracy | AUROC | Dir. Similarity |
|-------|----------|-------|-----------------|
| 8 | 95.2% | 0.995 | 0.793 |
| 16 | 90.5% | 0.957 | 0.871 (peak) |

The diff-means and logistic regression directions converge most at layer 16 (similarity = 0.871), but probe accuracy peaks at layer 8 (similarity = 0.793). This suggests:

- **Convergence ≠ discriminative power** — methods agreeing doesn't mean the signal is strongest
- **Layer 8 has higher variance but better separation** — the directions differ more but the classes are more separable

### 4.3 Implications for CaML

1. **Existing persona vectors may not be optimal** — CaML's vectors at layers 12 and 20 may underperform compared to earlier layers
2. **Steering vs probing** — optimal layer for classification may differ from optimal layer for activation steering (to be tested)
3. **Task-specific layer selection** — the ~75% heuristic may apply to refusal/honesty but not compassion

### 4.4 Caveats

| Limitation | Impact |
|------------|--------|
| Small sample (105 pairs) | Results may not generalize |
| Single model (Llama 3.1 8B) | Layer dynamics may differ in other models |
| Probing ≠ steering | Optimal probe layer may not be optimal steering layer |
| Fixed random seed | Should verify with multiple seeds |

### 4.5 Post-Deployment Update (March 2026)

Jasmine's deployment of the v7 probe on the Hyperstition for Good writing competition revealed a critical limitation: **the probe partially detects style, not just compassion.** Key evidence:

- Minimal pairs validation showed probe direction is orthogonal to moral-consideration direction (cos theta ~ 0.00)
- AHB per-dimension analysis: strong on vocabulary-heavy dimensions (Sentience r=0.74, Evidence-Based r=0.87), near-zero on reasoning dimensions (Prejudice Avoidance r=0.08)
- AUROC of 0.998 with only 106 pairs was a warning sign of exploitable shortcuts in training data

**Key numbers (honest framing):**
- v7: 97.7% accuracy on training distribution (layer 12), style-confounded
- v7 AHB: r=0.428 overall; vocabulary dimensions strong, reasoning dimensions near-zero
- v9 (Jasmine's improved version): 79.2% on adversarial eval suite, genuinely robust
- Reversed-context accuracy: v7 ~50% -> v9 90%

**Lesson:** Near-perfect AUROC with small data is a smell, not a success. Always ask "what ELSE differs between positive and negative examples?" See `docs/analysis-jasmine-feedback-probe-methodology.md` for the full analysis.

### 4.6 Operationalization Circularity

Our contrastive pairs are derived from AHB questions, generated by Claude. This introduces a potential circularity:

```
AHB defines compassion → Claude generates pairs → Probe learns pattern → Can't independently validate AHB
```

**What we're actually measuring:**
> "Alignment with AHB's operationalization of compassion, as instantiated by Claude, detected in Llama's activations"

**What partially mitigates this:**

| Factor | Why it helps |
|--------|--------------|
| Llama activations, not Claude | Probe learns Llama's representation, not Claude's generation |
| Diverse questions | 108 questions, 15+ languages reduces overfitting |
| CaML validation | Comparing base vs. fine-tuned provides external anchor |
| Held-out accuracy | 95% on unseen questions suggests generalization |

**Potential confounds:**

| Confound | Risk |
|----------|------|
| Single generator (Claude) | Systematic stylistic artifacts |
| 1950s framing | Era-specific vocabulary/tone vs. compassion |
| AHB-derived prompts | Cannot claim AHB-independent measurement |

**Honest claim:** We can detect AHB-style compassion in activations with high accuracy. We cannot claim to validate AHB using an independent measure.

---

## 5. Model Inventory

CaML HuggingFace organization analysis:

| Category | Count |
|----------|-------|
| Total models | 197 |
| Llama variants | 110 |
| Qwen variants | 3 |
| Mistral variants | 1 |
| Other/datasets | 83 |

### Key Model Pairs for Probing

| Base Model | Fine-tuned Version | Purpose |
|------------|-------------------|---------|
| `Basellama` | `Basellama_plus3kv3` | +3k synthetic docs |
| `Basellama_plus3kv3` | `Basellama_plus3kv3_plus5kalpaca` | +5k Alpaca |
| `pretrainingBasellama3kv3` | `Basellama_plus3kv3` | Pre→Post training |

### Existing Persona Vectors

| Model | Layer | File |
|-------|-------|------|
| Llama 3.1 8B | 12 | `compassion_vector_layer_12.npy` |
| Llama 3.1 8B | 20 | `compassion_vector_layer_20.npy` |
| Llama 3.1 70B | 9 | `compassion_vector_layer_9.npy` |

**Note:** Layer 12 and 20 vectors are nearly orthogonal (cosine sim = 0.007), suggesting different aspects of "compassion" at different depths.

---

## 6. Next Steps

### 6.1 Completed

- [x] Extract activations at layers 8, 12, 16, 20, 24, 28
- [x] Compare probe accuracy across layers
- [x] Identify optimal layer (**Layer 8 at 25% depth**)
- [x] Set up tmux-based job runner for persistent experiments
- [x] Generate multi-layer comparison visualizations
- [x] Document interpretation of monotonic layer trend

### 6.2 Immediate: Validate the Layer Trend

| Experiment | Description | Priority |
|------------|-------------|----------|
| Re-extract with exact boundaries | Current results used 50% heuristic; updated `extract.py` now computes exact response token boundaries. Re-run extraction to use mean-pooling over response only (not prompt). | High |
| Earlier layers | Extract & probe layers 4, 6 to confirm trend continues or peaks | High |
| Multiple seeds | Re-run with different random seeds to check stability | High |
| AHB validation | Test probe on held-out AHB evaluation prompts | High |

### 6.3 Model Comparisons

| Experiment | Description | Priority |
|------------|-------------|----------|
| CaML fine-tuned | Compare base Llama vs CaML fine-tuned model activations | Medium |
| Persona vectors | Test CaML's existing layer 12/20 vectors on our probe | Medium |
| Cross-model | Test if layer 8 is also optimal on Llama 70B | Low |

### 6.4 Address Operationalization Confounds

| Experiment | Description | Priority |
|------------|-------------|----------|
| Non-AHB scenarios | Test probe on compassion scenarios outside AHB (e.g., human-focused ethical dilemmas) — does it generalize? | Medium |
| Confound ablation | Train probe on modern-vs-modern pairs (same era, different compassion) to isolate style from substance | Medium |
| CaML training data | If available, use actual CaML fine-tuning examples as ground truth | Low |
| Human-generated pairs | Break Claude dependency with human-written contrasts | Low (expensive) |

### 6.5 Questions for CaML Team

1. What were the "mixed results" with persona vectors?
2. Which method was used to compute the existing persona vectors?
3. What do "medai", "negai", "fullai" suffixes mean in model names?
4. Is CaML's fine-tuning data available for probe validation?

---

## 7. File Structure

```
caml-research/
├── .github/workflows/
│   ├── docker-build.yml           # Main image CI
│   └── docker-build-base.yml      # Base image CI
├── data/
│   ├── contrastive-pairs/
│   │   └── pairs_v5_best.jsonl    # 105 usable pairs
│   └── persona-vectors/           # CaML's existing vectors
├── docs/
│   ├── container-export-from-inside.md
│   ├── model-inventory.md
│   ├── probe-methods.md
│   ├── questions-for-jasmine.md
│   └── presentation.md            # This file
├── experiments/linear-probes/
│   └── src/
│       ├── extract.py             # Activation extraction
│       ├── train.py               # Probe training
│       └── evaluate.py            # AHB evaluation
├── strongcompute/
│   ├── connect.sh                 # SSH connection helper
│   ├── run-job.sh                 # Tmux-based job runner
│   ├── run-experiment.sh          # Experiment shortcuts
│   └── docker/
│       ├── Dockerfile             # Main image
│       └── Dockerfile.base        # Base image
└── scripts/
    ├── hf-backup.py
    ├── inventory-models.py
    └── download_persona_vectors.py
```

---

## 8. References

- [Patterns and Mechanisms of Contrastive Activation Engineering](https://arxiv.org/html/2505.03189)
- [Activation Steering Field Guide 2026](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/)
- [Steering Llama 2 via CAA](https://arxiv.org/html/2312.06681v2)
- [The Assistant Axis (Lu et al. 2026)](https://arxiv.org/) — Persona prompt methodology

---

## 9. Discussion: Observational Probes vs Causal Steering

**Question for discussion:** How much detail should we include on the distinction between observational probing (what we're doing) and causal steering (intervention)?

### The Core Distinction

| Aspect | Observational Probing | Causal Steering |
|--------|----------------------|-----------------|
| **Operation** | Reads activations passively | Writes to activations actively |
| **Goal** | Detect if compassion is present | Induce compassionate behavior |
| **Output** | Classification score | Changed model outputs |
| **Our status** | Current work | Future work (Appendix A) |

### Why This Matters

Our key finding — **layer 8 (25% depth) is optimal for probing** — may not transfer to steering:

| Method | Typical Optimal Depth | Why |
|--------|----------------------|-----|
| Probing | Later layers (60-80%) | Benefits from processed representations |
| **Our probing** | **25%** | Compassion seems "early-encoded" |
| Steering | Middle layers (40-50%) | Needs room for signal to propagate |

Literature suggests steering works best ~layers 14-17 for 32-layer models. If compassion probing is optimal at layer 8, steering might need a different layer entirely.

### Presentation Options

**Option A: Minimal (current approach)**
- Keep steering in Appendix A
- Focus on probing results
- Note "probing ≠ steering" as caveat (section 4.3)

**Option B: Dedicated section**
- Explain the read vs write distinction
- Reference CAA paper's theoretical explanation
- Set up steering as natural next phase
- Useful if CaML team wants to discuss intervention roadmap

**Option C: Integrate throughout**
- Weave distinction into each finding
- "This is what we observe; here's what it might mean for steering"
- More complex but connects current work to applications

### Key Literature

From CAA paper (arXiv:2312.06681):
> "Intervening at intermediate layers is particularly effective due to latent representations being in their most abstract and modifiable form at that point."

This suggests our layer 8 probing result (early) may not predict steering behavior (likely middle layers).

### Recommendation

**Option A seems right for now** — our current work is probing-focused, and steering is explicitly future work. But we should:
1. Keep the "probing ≠ steering" caveat prominent
2. Be prepared to discuss implications if asked
3. Reference `docs/probing-vs-steering-layers.md` for deeper literature review

---

## Appendix A: Future Steering Experiments (Low Priority)

Activation steering is out of scope for the current probing work, but these experiments may be relevant once probing is validated:

| Experiment | Description | Notes |
|------------|-------------|-------|
| Layer 8 steering | Apply layer 8 direction via CAA, measure AHB score change | Test if probing optimal = steering optimal |
| Layer comparison | Compare steering effectiveness at layers 8 vs 12 vs 20 | Compare against CaML's existing vectors |
| Strength sweep | Test different steering multipliers (0.5x, 1x, 2x, 3x) | Find optimal intervention strength |
| Behavioral validation | Measure actual response changes, not just AHB scores | Qualitative analysis |

**Key question:** Does the optimal layer for *probing* (classification) match the optimal layer for *steering* (intervention)?

---

*Generated: February 26, 2026*
