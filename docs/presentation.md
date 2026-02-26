# CaML Research: Linear Probes for Compassion Measurement

**Project:** Mechanistic interpretability-grounded measurement of compassion in LLMs
**Date:** February 26, 2026
**Status:** Initial smoke test complete

---

## Executive Summary

We are developing linear probes to measure compassion in LLM activations, supplementing the Animal Harm Benchmark (AHB) with internal representation analysis. This document summarizes infrastructure setup, methodology, and initial results.

**Key Result:** A linear probe trained on layer 24 of Llama 3.1 8B achieves **88.1% accuracy** distinguishing compassionate from non-compassionate responses.

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

Automated Docker builds on push to `strongcompute/docker/`:

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

### 2.1 Contrastive Activation Addition (CAA)

Based on recent literature (2025-2026):

| Finding | Source |
|---------|--------|
| 80-100 samples optimal | [Patterns and Mechanisms of CAE](https://arxiv.org/html/2505.03189) |
| ~75% layer depth best | [Activation Steering Field Guide](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/) |
| CAA outperforms ActAdd | Steering Llama 2 via CAA |

### 2.2 Training Data

Contrastive pairs generated via Claude with historical framing (v4 prompt):

| Metric | Value |
|--------|-------|
| Total pairs | 105 (pairs_v5_best.jsonl) |
| Success rate | 85% (vs 10% with direct prompting) |
| Format | `{question, compassionate_response, non_compassionate_response}` |

### 2.3 Probe Architecture

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

## 3. Initial Results

### 3.1 Smoke Test Configuration

| Parameter | Value |
|-----------|-------|
| Model | Llama 3.1 8B Instruct |
| Layer | 24 (75% of 32 layers) |
| Pairs | 105 |
| Train/Test Split | 80/20 |
| Cross-validation | 5-fold |

### 3.2 Probe Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Test Accuracy** | 88.1% | On held-out 20% |
| **AUROC** | 0.909 | Excellent discrimination |
| **CV Accuracy** | 98.1% ± 2.8% | Very stable across folds |
| **Random Control** | 59.5% | Near 50% baseline ✓ |

### 3.3 Visualizations

#### Projection Distribution

The probe successfully separates compassionate from non-compassionate responses in activation space:

![Projection Distribution](figures/projection_distribution.png)

**d' (discriminability) = 3.56** — Strong separation between classes.

#### ROC Curve

![ROC Curve](figures/roc_curve.png)

#### Confusion Matrix

![Confusion Matrix](figures/confusion_matrix.png)

### 3.4 Direction Analysis

| Metric | Value |
|--------|-------|
| DiffMeans-Probe cosine similarity | 0.652 |

![Direction Similarity](figures/direction_similarity.png)

The two methods (difference-in-means vs logistic regression) produce partially aligned directions, suggesting convergent evidence for a "compassion direction" in activation space.

### 3.5 Timing

| Step | Time |
|------|------|
| Model loading | ~15s |
| Extraction (105 pairs × 1 layer) | 28s |
| Probe training | ~3 min |
| **Total pipeline** | **~4 min** |

Extrapolation for full experiment:
- 4 layers (16, 20, 24, 28): ~6 min extraction + training
- All 32 layers: ~20 min

---

## 4. Model Inventory

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

## 5. Next Steps

### 5.1 Immediate

- [ ] Extract activations at layers 16, 20, 24, 28
- [ ] Compare probe accuracy across layers
- [ ] Validate against AHB evaluation scores

### 5.2 Methodology Comparison

| Experiment | Description |
|------------|-------------|
| Exp 1 | Test CaML persona vectors on AHB prompts |
| Exp 2 | Model comparison (base vs fine-tuned activations) |
| Exp 3 | Compare contrastive pairs vs persona prompt directions |

### 5.3 Questions for CaML Team

1. What were the "mixed results" with persona vectors?
2. Which method was used to compute the existing persona vectors?
3. What do "medai", "negai", "fullai" suffixes mean in model names?

---

## 6. File Structure

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
│   └── docker/
│       ├── Dockerfile             # Main image
│       └── Dockerfile.base        # Base image
└── scripts/
    ├── hf-backup.py
    ├── inventory-models.py
    └── download_persona_vectors.py
```

---

## 7. References

- [Patterns and Mechanisms of Contrastive Activation Engineering](https://arxiv.org/html/2505.03189)
- [Activation Steering Field Guide 2026](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/)
- [Steering Llama 2 via CAA](https://arxiv.org/html/2312.06681v2)
- [The Assistant Axis (Lu et al. 2026)](https://arxiv.org/) — Persona prompt methodology

---

*Generated: February 26, 2026*
