# CaML Capstone Project Proposal

**Program:** ElectricSheep Capstone
**Start:** February 9, 2026

---

## Goal

Create a high quality, mechanistic interpretability-grounded measurement of compassion in AI systems.

The existing Animal Harm Benchmark (AHB) covers this partially, but only on model outputs. We want to supplement with internal representations to:

- Detect and quantify compassion strength in LLMs
- Compare compassion strength across models
- Identify correlated and anti-correlated values
- Supplement AHB findings for robustness

---

## Methodology

**Linear probes** (primary approach) after exploring and rejecting steering vectors.

### Why Not Steering Vectors?

Multiple independent researchers found:

- Results are inconsistent even for detection tasks
- Steering does not reliably capture values in its current state
- Higher steering strength degrades accuracy and increases variance
- Even Anthropic's own papers hedge significantly on reliability

Key insight: *"Contrastive directions remain very strong for detection/probing, but hidden-state steering doesn't seem to be a causal control knob."*

### Why Linear Probes?

- Contrastive pairs work well for detection/probing
- More straightforward methodology
- Better suited for measurement (our goal) vs. control

### Possible Future: SAEs

Sparse Autoencoders may be worth exploring (natural language feature descriptions via Neuronpedia/GemmaScope).

---

## Research Questions

1. Can we reliably detect compassion-related representations in LLM hidden states?
2. How does compassion strength vary across models?
3. What values correlate/anti-correlate with compassion?
4. In which layers does compassion "live"?

---

## Approach

1. Define contrastive pairs for compassion (compassionate vs. non-compassionate responses)
2. Extract hidden states from target layers
3. Train linear probes to classify/score compassion
4. Validate against AHB outputs
5. Compare across models

---

## SAE Resources for Llama

| Resource | Model | Coverage | Source |
|----------|-------|----------|--------|
| [Goodfire SAEs](https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50) | Llama 3.3 70B | Layer 50 | [Announcement](https://www.goodfire.ai/blog/sae-open-source-announcement) |
| [Goodfire SAEs](https://huggingface.co/Goodfire) | Llama 3.1 8B | Layer 19 | [Announcement](https://www.goodfire.ai/blog/sae-open-source-announcement) |
| [Llama Scope](https://arxiv.org/abs/2410.20526) | Llama 3.1 8B | All layers (256 SAEs, 32K-128K features) | [Paper](https://arxiv.org/abs/2410.20526) |
| [qresearch SAE](https://huggingface.co/qresearch/DeepSeek-R1-Distill-Llama-70B-SAE-l48) | DeepSeek-R1-Distill-Llama-70B | Layer 48 | HuggingFace |

**Recommended approach:** Prototype on 8B with richer Llama Scope features (all layers), then extend to 70B using Goodfire SAEs.

---

## Related Work

- [Animal Harm Benchmark](https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/ahb/) — output-based measurement
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) (Anthropic) — foundational SAE paper
- [Neuronpedia](https://neuronpedia.org/) — Natural language feature descriptions for SAE features
