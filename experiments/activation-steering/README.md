# Activation Steering

Activation steering for trait expression using contrastive activation addition (CAA).

## Status

**On hold** — Focus is currently on linear probes (detection). Steering may be revisited after probe validation.

## Overview

Extract and apply steering vectors to modify LLM behavior toward specific traits (e.g., compassion, empathy). This experiment explores both the potential and limitations of activation steering.

### The Detection-Steering Gap

| Aspect | Detection (Probing) | Steering (Intervention) |
|--------|---------------------|------------------------|
| Success rate | High (0.93+ AUROC) | Highly variable |
| Requirements | Linear separability | Causal relevance to output |
| Field status | Well-established | Known reliability issues |

## Usage

```bash
python -m persona_vectors --help
```

## Documentation

| Document | Description |
|----------|-------------|
| [Literature Synthesis](docs/literature-synthesis.md) | Research findings on when steering works |
| [Diagnostic Checklist](docs/diagnostic-checklist.md) | Step-by-step diagnostic protocol |
| [SAE Diagnostics](docs/sae-diagnostics.md) | SAE quality and intervention diagnostics |
| [Remediation Checklist](docs/persona-vectors-remediation-checklist.md) | Fixing common issues |

## Notebooks

- [PersonaVectors_FINAL.ipynb](notebooks/PersonaVectors_FINAL%20(2).ipynb) — Working example (Llama-3.1-70B, +1.0 compassion improvement)

## Code Structure

```
activation-steering/
├── __init__.py
├── __main__.py
├── main.py          # Main entry point
├── cli.py           # Command-line interface
├── config.py        # Configuration
├── model.py         # Model loading
├── extraction.py    # Vector extraction
├── steering.py      # Activation steering
├── evaluation.py    # Evaluation metrics
├── controls.py      # Experimental controls
├── stats.py         # Statistics utilities
├── artifacts.py     # Artifact management
├── docs/            # Documentation
└── notebooks/       # Jupyter notebooks
```

## When Steering Works

1. **Large models** (70B shows results where smaller models fail)
2. **Proper layer selection** (middle layers, 50-75% depth)
3. **SAE reconstruction >0.95**
4. **Feature actually fires** on target prompts
5. **Baseline behavior 30-70%** (room to move)
6. **External evaluation** (different model as judge)

## Resources

- [Steering Llama 2 via CAA](https://arxiv.org/abs/2312.06681) - ACL 2024
- [A Sober Look at Steering Vectors](https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/a-sober-look-at-steering-vectors-for-llms)
- [nrimsky/CAA GitHub](https://github.com/nrimsky/CAA)
