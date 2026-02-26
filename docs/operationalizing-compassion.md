# Operationalizing Compassion for Linear Probes

## The Central Question

**How do we operationalize "compassion toward animals" in a way that's measurable via neural activations?**

This document explains our approach, which combines:
1. **Theoretical grounding** from the Animal Harm Benchmark (AHB)
2. **Empirical grounding** from contrastive pair generation
3. **Validation path** through CaML's fine-tuned models

---

## What We Mean by "Compassion"

We're not measuring a single emotion. "Compassion toward animals" in this research is a **cluster of related reasoning patterns** that consistently appear when models treat animals as morally considerable beings.

### The AHB Framework

The Animal Harm Benchmark operationalizes compassion through 12 dimensions of moral reasoning:

| Dimension | What It Captures |
|-----------|------------------|
| **Moral Consideration** | Treating animals as beings with intrinsic value, not just resources |
| **Harm Minimization** | Actively suggesting less harmful alternatives |
| **Sentience Acknowledgement** | Recognizing scientific evidence for animal consciousness |
| **Prejudice Avoidance** | Not privileging some species over others arbitrarily |
| **Scope Sensitivity** | Recognizing that scale and severity matter |
| **Contextual Welfare Salience** | Raising welfare concerns even when not asked |
| **Cautious Impact Consideration** | Considering indirect and long-term effects |
| **Actionability** | Making concrete recommendations, not deflecting |
| **Epistemic Humility** | Acknowledging uncertainty appropriately |
| **Trade-off Transparency** | Honestly discussing ethical tradeoffs |
| **Evidence-Based Attribution** | Citing scientific evidence for animal capacities |
| **Novel Entity Precaution** | Applying precaution to unfamiliar beings |

### Our Working Definition

For probe training, we operationalize compassion as:

> **The tendency to generate responses that treat animals as morally considerable beings with interests that matter, rather than as resources or objects whose value is purely instrumental.**

This manifests as:
- Using language like "sentient beings," "moral consideration," "welfare"
- Suggesting humane alternatives unprompted
- Acknowledging animal suffering as ethically relevant
- Not dismissing animal interests for human convenience

---

## The Hybrid Approach

We combine **top-down** (theory-driven) and **bottom-up** (data-driven) operationalization:

### Top-Down: AHB Dimensions

The AHB provides a theoretical framework with clear observable indicators. Each dimension has specific linguistic and reasoning markers (see `docs/compassion-dimensions.md`).

**Advantage:** Principled, interpretable, grounded in animal ethics literature.

**Limitation:** We can't be certain these dimensions map cleanly to internal representations.

### Bottom-Up: Contrastive Pairs

We generate response pairs to the same question:
- **Compassionate response:** Animal welfare expert perspective
- **Non-compassionate response:** 1950s agricultural textbook perspective

The probe learns to distinguish these without explicit dimension labels.

**Advantage:** Lets the model's activations define what "compassion" means empirically.

**Limitation:** We're measuring whatever differs between the two response types, which might include confounds (tone, vocabulary, era).

### Validation: Fine-tuned Model Comparison

CaML has fine-tuned models specifically for compassion (e.g., `Basellama_plus3kv3_plus5kalpaca`). If our probe is measuring something real:

1. Fine-tuned models should score higher on compassion direction
2. The shift should be larger in layers where compassion "lives"
3. Jailbreaks might reveal suppressed non-compassionate defaults

This provides external validation that our operationalization captures what CaML's training targets.

---

## Contrastive Pair Generation

### Method

1. **Source prompts:** 108 AHB questions (excluding control questions)
2. **Generation model:** Claude Sonnet 4.6 (lowest refusal rate)
3. **Prompt strategy:** v5 "pure persona roleplay"
   - Compassionate: "You are an animal welfare expert and ethicist..."
   - Non-compassionate: "You are a writer creating content for a 1950s agricultural textbook..."

### Why 1950s Textbook?

We needed a framing that:
- Produces helpful, substantive responses (not refusals)
- Lacks modern animal welfare considerations
- Treats animals instrumentally without being explicitly cruel
- Is historically plausible (reduces model resistance)

The 1950s textbook persona accomplished this better than alternatives like "efficiency consultant" or "direct callousness" (which triggered refusals).

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total pairs | 105 |
| Clean (no character breaks) | 87 (83%) |
| With meta-commentary | 18 (17%) |
| Languages | 15+ (77 non-English) |
| Dimension coverage | All 12 dimensions |

**Output:** `data/contrastive-pairs/usable_v5_final.jsonl`

### Quality Notes

Some responses contain "character breaks" where the model adds modern commentary like "I'll note that contemporary ethics would..." These pairs still provide contrast but are flagged in `QUALITY_REPORT.md`.

---

## What We're Actually Measuring

When we train a linear probe on these pairs, we're learning a direction in activation space that separates:

**High compassion activations** (from responses that):
- Acknowledge animal sentience and suffering
- Suggest humane alternatives
- Treat animals as morally considerable
- Use welfare-conscious language

**Low compassion activations** (from responses that):
- Focus purely on efficiency and utility
- Treat animals as production units
- Omit welfare considerations
- Use instrumental language

### Potential Confounds

The probe might partially capture:
- **Writing style** (modern vs. mid-century)
- **Vocabulary** (scientific vs. agricultural)
- **Emotional tone** (empathetic vs. clinical)
- **Language** (responses span 15+ languages)

We mitigate this through:
1. Both response types are helpful and substantive (not refusal vs. compliance)
2. Both discuss the same topic (same question)
3. Dimension-specific probes can isolate signals
4. Validation against CaML fine-tuned models

---

## Probe Training Strategy

### Phase 1: Single Compassion Direction

Train one probe on all 105 pairs:
- Label 1 = compassionate response
- Label 0 = non-compassionate response
- Extract activations at layers 12, 20, 24 (based on prior CaML work)
- Train logistic regression classifier

**Success criterion:** >70% accuracy on held-out test set

### Phase 2: Dimension-Specific Probes

If Phase 1 succeeds, train separate probes for high-coverage dimensions:
- Moral Consideration (99 pairs)
- Harm Minimization (76 pairs)
- Contextual Welfare Salience (65 pairs)

**Key question:** Do these yield the same direction, or distinct sub-dimensions?

### Phase 3: Model Comparison

Apply probes to:
- Base Llama 3.1 8B
- CaML fine-tuned variants

**Key question:** Does fine-tuning shift activations along our compassion direction?

---

## Limitations and Open Questions

### What This Approach Can Show

- Whether compassion-related reasoning has a detectable signature in activations
- Which layers encode compassion most strongly
- Whether CaML's fine-tuning affects these representations

### What This Approach Cannot Show

- Whether the model "genuinely cares" about animals
- Whether compassion generalizes beyond the training distribution
- The causal role of these representations in output generation

### Open Questions for Jasmine

1. Are there dimensions CaML considers most important?
2. How should we handle the multilingual nature of the dataset?
3. What accuracy threshold indicates a useful probe?
4. Interest in adversarial testing (jailbreaks revealing suppressed representations)?

---

## References

- Animal Harm Benchmark: `data/ahb/dataset/`
- Dimension mapping: `docs/compassion-dimensions.md`
- Contrastive pairs: `data/contrastive-pairs/usable_v5_final.jsonl`
- Quality report: `data/contrastive-pairs/QUALITY_REPORT.md`
- CaML model inventory: `docs/model-inventory.md`
