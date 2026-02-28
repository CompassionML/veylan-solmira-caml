# The Cutting Room Floor

**Project:** Linear Probes for Compassion Measurement

---

## Candidates for Cutting (5 elements)

### 1. Infrastructure details (StrongCompute, Docker, Flash Attention)

*Why cut:* Technical implementation details that don't affect the core finding. The audience cares that we measured compassion in activations, not that we used a custom Docker container with Flash Attention 2. Already relegated to full presentation; absent from pitch.

### 2. The layer 8 finding / steering heuristic contradiction

*Why cut:* The main goal is measuring compassion, not challenging the ~75% depth heuristic from steering literature. This finding is a tangent — interesting to interpretability researchers but not essential for CaML's goal of validating compassion training.

### 3. Direction similarity analysis (diff-means vs logistic regression convergence)

*Why cut:* Methodological detail showing that two direction-extraction methods converge at layer 16 but diverge elsewhere. Interesting for probe methodology nerds, but doesn't affect the practical result (probe works at 95% accuracy).

### 4. Multi-layer comparison beyond the optimal

*Why cut:* We tested 6 layers (8, 12, 16, 20, 24, 28) but only layer 8 matters for the tool. The monotonic decline is intellectually interesting but for practical use, we just need to know "use layer 8." The full table could go to an appendix.

### 5. Operationalization circularity discussion

*Why cut:* Spending significant space on limitations (AHB-derived pairs, Claude generation, potential confounds) could undermine confidence in the result. A shorter "limitations" note might suffice. Audiences may not care about methodological purity if the probe works.

---

## Defense of 2 Keepers

### Keeping: The layer 8 finding

*Defense:* This finding is not a tangent — it's the most actionable insight for CaML. If CaML has been applying compassion vectors at layers 12 and 20 (where their existing vectors live), this result suggests they may be intervening suboptimally. Layer 8 encodes compassion most cleanly. This directly informs where to measure AND where to intervene. Cutting it would reduce the project to "we built a probe" without the insight that makes it useful. The contradiction with steering literature also signals that compassion may behave differently from other concepts (refusal, honesty) — worth knowing for anyone doing value alignment.

### Keeping: The operationalization circularity discussion

*Defense:* Acknowledging limitations upfront builds credibility, not undermines it. A technical audience will immediately spot that our pairs are AHB-derived and Claude-generated — if we don't address it, they'll assume we haven't thought about it. By naming the circularity explicitly and explaining what we CAN and CANNOT claim, we demonstrate rigor. This also sets up the validation roadmap (negative controls, non-AHB scenarios) as motivated next steps rather than afterthoughts. Cutting the limitations would make the project look naive; keeping them makes it look honest and well-designed.

---

## Summary

| Element | Cut or Keep | Reason |
|---------|-------------|--------|
| Infrastructure details | Cut | Implementation, not insight |
| Layer 8 finding | **Keep** | Most actionable insight for CaML |
| Direction similarity | Cut | Methodological detail, not essential |
| Full layer table | Cut (to appendix) | Only layer 8 matters practically |
| Circularity discussion | **Keep** | Builds credibility, motivates roadmap |
