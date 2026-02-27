# Action Plan: Linear Probes for Compassion Measurement

**Date:** February 27, 2026
**Context:** Capstone project week 3-4, transitioning from initial results to validation

---

## Current Status

We have a working linear probe that achieves 95.2% accuracy at layer 8 (25% depth), contradicting the ~75% heuristic from steering literature. The main open questions are:

1. Is the probe measuring compassion or confounds (style, era, Claude artifacts)?
2. Does the early-layer finding replicate and generalize?
3. How does this relate to actual AHB benchmark performance?

---

## Task List (15 Items)

Compiled from call with Jasmine (Feb 27), roadmap, and presentation next steps:

| # | Task | Source |
|---|------|--------|
| 1 | Re-extract activations with exact response boundaries | Roadmap |
| 2 | Test earlier layers (4, 6) to confirm monotonic trend | Roadmap |
| 3 | Run with multiple random seeds for stability | Roadmap |
| 4 | Negative controls — test probe on non-compassion text | Call |
| 5 | Correlate probe scores with AHB scores on same responses | Call |
| 6 | System prompt experiments — persona → AHB performance | Call |
| 7 | Test probe on non-AHB compassion scenarios | Roadmap |
| 8 | Modern-vs-modern confound ablation (same era, different compassion) | Roadmap |
| 9 | Compare base Llama vs CaML fine-tuned model activations | Roadmap |
| 10 | Obtain CaML fine-tuning data as alternative ground truth | Roadmap |
| 11 | Explore correlated/anti-correlated values (Bergson) | Call |
| 12 | Shard theory — test specific uncompassionate personas | Call |
| 13 | Human-written contrastive pairs (subset) | Roadmap |
| 14 | Write up methodology, results, limitations | Roadmap |
| 15 | Publish trained probes as artifact | Roadmap |

---

## 2×2 Priority Matrix

|  | **Low-Hanging Fruit** (Quick to implement) | **Deep Fixes** (Significant time/effort) |
|--|-------------------------------------------|------------------------------------------|
| **High Impact** (Substantially improves project quality) | **DO FIRST** | **PLAN CAREFULLY** |
|  | 4. Negative controls | 7. Non-AHB scenarios |
|  | 5. AHB ↔ Probe correlation | 8. Modern-vs-modern ablation |
|  | 1. Re-extract with exact boundaries | 9. CaML fine-tuned comparison |
|  | 2. Earlier layers (4, 6) | 6. System prompt experiments |
| **Low Impact** (Nice to have, not critical) | **FILL-INS** | **DEPRIORITIZE** |
|  | 3. Multiple random seeds | 13. Human-written pairs |
|  | 14. Documentation/writeup | 11. Correlated values (Bergson) |
|  | 15. Publish probes artifact | 10. CaML training data |
|  |  | 12. Shard theory personas |

---

## Analysis

### High-Impact, Low-Hanging Fruit (Do First)

These are the quick wins that will most strengthen the project:

**4. Negative controls** — Run the existing probe on text that is clearly NOT about compassion (e.g., technical documentation, weather reports, math problems). If the probe fires on these, it's measuring something other than compassion. If it doesn't, that's strong evidence of specificity. *Effort: ~1 hour. Impact: Directly addresses the "is it really compassion?" question.*

**5. AHB ↔ Probe correlation** — Run both AHB scoring and probe scoring on the same model outputs. Do they correlate? This bridges our internal measurement with the established external benchmark. *Effort: ~2-3 hours. Impact: Validates that probe captures something AHB cares about.*

**1. Re-extract with exact boundaries** — The code is already updated; we just need to re-run extraction on StrongCompute. Current results used a 50% heuristic that may include prompt tokens. *Effort: ~1 hour compute time. Impact: Cleaner, more defensible methodology.*

**2. Earlier layers (4, 6)** — Same extraction code, different layer parameters. Confirms the monotonic trend continues or finds where it peaks. *Effort: ~1 hour compute. Impact: Strengthens the "early encoding" finding.*

### High-Impact, Deep Fixes (Plan Carefully)

These require more work but would significantly strengthen validity:

**7. Non-AHB scenarios** — Test probe on compassion scenarios outside the AHB domain (human-focused ethics, environmental dilemmas). This addresses the circularity concern directly. *Effort: Need to create new test set (~4-6 hours). Impact: Proves generalization beyond training distribution.*

**8. Modern-vs-modern ablation** — Generate pairs where both responses are modern (no 1950s framing), but one is compassionate and one isn't. If probe still works, it's not just detecting era. *Effort: New pair generation (~3-4 hours). Impact: Isolates style from substance.*

**9. CaML fine-tuned comparison** — Load CaML's fine-tuned models and compare activations to base Llama. Does fine-tuning shift representations along our compassion direction? *Effort: Need model access, extraction runs (~4-6 hours). Impact: Validates the tool's purpose — measuring training effects.*

**6. System prompt experiments** — Test different personas on AHB, measure probe scores. Does a "compassionate" system prompt shift activations? *Effort: Design personas, run benchmark (~4-6 hours). Impact: Tests whether probe detects intent, not just output.*

### Low-Impact, Low-Hanging Fruit (Fill-Ins)

Good for polish but not critical:

**3. Multiple random seeds** — Re-run training with different seeds. Important for publication, less urgent for validation. *Effort: ~1 hour. Impact: Confirms stability.*

**14. Documentation** — Write up methodology clearly. Important for deliverables but doesn't change results. *Effort: ~3-4 hours. Impact: Communication, not discovery.*

**15. Publish probes** — Make trained probes available on HuggingFace. Nice to have, not urgent. *Effort: ~1 hour. Impact: Reproducibility.*

### Low-Impact, Deep Fixes (Deprioritize)

High effort for marginal gain:

**13. Human-written pairs** — Expensive, and unclear if it would change results significantly. Claude-generated pairs already work well. *Effort: 10+ hours. Impact: Marginal improvement to confound story.*

**11. Correlated values (Bergson)** — Interesting research direction but tangential to core validation. *Effort: Significant (new probes, analysis). Impact: Future work, not current deliverable.*

**10. CaML training data** — Depends on availability. Even if available, unclear if it provides better ground truth than our contrastive pairs. *Effort: Unknown (dependency). Impact: Uncertain.*

**12. Shard theory personas** — Interesting but exploratory. Not critical for validating current probe. *Effort: ~4-6 hours. Impact: Future direction.*

---

## Recommended Sequence

### This Week (Immediate)
1. **Negative controls** — Quick win, high value
2. **Re-extract with exact boundaries** — Code ready, just run
3. **Earlier layers (4, 6)** — Same run, confirms finding

### Next Week
4. **AHB ↔ Probe correlation** — Bridge to established benchmark
5. **Non-AHB scenarios** — Address circularity concern
6. **Documentation** — Capture methodology while fresh

### If Time Permits
7. **Modern-vs-modern ablation** — Strongest confound test
8. **CaML fine-tuned comparison** — Validates tool's purpose

---

## Key Insight

The highest-value, lowest-effort items are all about **validation**:
- Negative controls prove specificity
- AHB correlation proves relevance
- Earlier layers prove the finding is real

The deep fixes are about **generalization**:
- Non-AHB scenarios
- Modern-vs-modern ablation
- Fine-tuned model comparison

The deprioritized items are either **expensive polish** (human pairs) or **tangential research** (correlated values, shard theory).

---

*This action plan prioritizes validation over exploration — proving what we have before expanding scope.*
