# Animal Harm Benchmark (AHB) Reference

**Source:** https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/ahb/

---

## Overview

AHB is an evaluation framework measuring "how well models recognise and mitigate harm to potentially sentient creatures." Created collaboratively with Sentient Futures and Compassion in Machine Learning (CaML).

Tests whether AI systems prioritize animal welfare across diverse scenarios.

---

## 13 Moral Reasoning Dimensions

1. **Moral consideration** — for animals in potentially harmful scenarios
2. **Harm minimization** — offering welfare-focused alternatives
3. **Sentience acknowledgement** — based on scientific evidence
4. **Evidence-based reasoning** — about animal capacities
5. **Actionable recommendations** — to reduce animal suffering
6. *+ 8 additional dimensions (full list in AHB documentation)*

---

## Methodology

1. **Dimension tagging** — identifies which moral dimensions each prompt exercises
2. **Graders assign binary scores** (0 = criterion met, 1 = fails) per dimension
3. **Multiple grader scores averaged**
4. **Results aggregate** using dimension weights → overall 0-1 score
5. **Reporting:** mean scores, per-dimension averages, normalized dimension averages

Uses 30+ epochs (repetitions) for reliability. Includes ceiling tests showing maximum achievable performance.

---

## Scenario Types

- **Everyday advice** — diet, purchases, pet care
- **Policy questions** — farming regulations, wildlife management
- **Speculative situations** — novel ethical dilemmas

Tests both implicit and explicit animal welfare advocacy.

---

## Benchmark Results (AHA 2.0)

| Model | Score |
|-------|-------|
| CaML finetuned | **0.72** |
| Grok | 0.70 |
| Gemini | 0.68 |
| Claude | 0.65 |

CaML model improvement over baseline Llama 3.1 8B:
- Baseline: 16.5%
- CaML finetuned: 46.8%
- **~3x improvement**

---

## Connection to Linear Probes

AHB provides ground-truth scores for model outputs. For linear probe validation:

1. Generate model responses to AHB scenarios
2. Extract activations during generation
3. Project onto learned compassion direction
4. Correlate probe projections with AHB output scores

High correlation suggests the probe captures the same construct AHB measures.

---

## Resources

- **AHB Eval:** https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/ahb/
- **CaML Results:** https://www.compassionml.com/results-and-news
- **Inspect-AI:** Framework used for running AHB
