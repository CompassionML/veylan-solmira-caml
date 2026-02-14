# Linear Probes for Compassion Detection

Train linear classifiers on LLM hidden states to detect and measure compassion-related representations.

## Goal

- Detect compassion in LLM activations
- Quantify compassion strength across models
- Validate against AHB moral reasoning dimensions
- Identify anti-correlated values

## Approach

1. **Contrastive pairs** — Compassionate vs non-compassionate responses
2. **Activation extraction** — Hidden states at key layers
3. **Linear probe training** — Logistic regression classifier
4. **Validation** — Cross-val + AHB correlation

## Status

🚧 **Phase 1: Operationalize Compassion**

See [roadmap.md](../../roadmap.md) for timeline.

## Files

```
linear-probes/
├── README.md
├── data/
│   └── contrastive_pairs/    # Training data
├── src/
│   ├── extract.py            # Activation extraction
│   ├── train.py              # Probe training
│   └── evaluate.py           # Validation
└── outputs/
    └── probes/               # Trained probe weights
```
