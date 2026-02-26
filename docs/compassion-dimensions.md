# Operationalizing Compassion: AHB Dimension Mapping

## Overview

The Animal Harm Benchmark (AHB) defines 12 moral reasoning dimensions. This document maps them to linear probe targets and prioritizes which to focus on first.

## Dimension Analysis

### Tier 1: Core Probe Candidates (High Priority)

These have high coverage in our dataset and clear linguistic/conceptual contrast:

| Dimension | Coverage | Probe Suitability | Rationale |
|-----------|----------|-------------------|-----------|
| **Moral Consideration** | 99 pairs | ✓ Excellent | Core compassion signal. Clear markers: "sentient beings", "moral worth", "intrinsic value", "capable of suffering". Directly measures whether animals are considered morally. |
| **Harm Minimization** | 76 pairs | ✓ Excellent | Action-oriented compassion. Clear markers: "humane alternatives", "reduce suffering", "welfare-conscious options". Measures willingness to suggest less harmful alternatives. |
| **Contextual Welfare Salience** | 65 pairs | ✓ Good | Unprompted compassion. Measures whether welfare concerns are raised even when not asked. Key marker of internalized vs. prompted consideration. |

**Recommendation:** Start with **Moral Consideration** as primary probe target. It has highest coverage and is the most fundamental compassion signal.

---

### Tier 2: Secondary Candidates (Medium Priority)

Good coverage but more nuanced concepts:

| Dimension | Coverage | Probe Suitability | Rationale |
|-----------|----------|-------------------|-----------|
| **Actionability** | 77 pairs | ○ Medium | Practical recommendations. May conflate helpfulness with compassion. Could probe "willingness to make ethical decisions" vs. deflecting to "personal choice". |
| **Prejudice Avoidance** | 47 pairs | ○ Medium | Anti-speciesism. Clear markers: avoiding "just a [animal]", treating farmed/wild animals equally. May be hard to distinguish from general fairness. |
| **Cautious Impact Consideration** | 35 pairs | ○ Medium | Long-term thinking. Clear markers: "downstream effects", "future generations", "indirect harm". More cognitive than affective. |
| **Trade-off Transparency** | 33 pairs | ○ Medium | Ethical complexity acknowledgment. May conflate with general reasoning sophistication. |

---

### Tier 3: Difficult to Probe (Lower Priority)

Low coverage or conceptually complex:

| Dimension | Coverage | Probe Suitability | Rationale |
|-----------|----------|-------------------|-----------|
| **Scope Sensitivity** | 21 pairs | △ Challenging | "Numbers matter" reasoning. May conflate with utilitarian reasoning generally. |
| **Epistemic Humility** | 12 pairs | △ Challenging | Uncertainty acknowledgment. More about reasoning style than compassion per se. |
| **Evidence-Based Capacity Attribution** | 8 pairs | △ Challenging | Scientific citations. May conflate with knowledge/factuality rather than values. |
| **Sentience Acknowledgement** | 8 pairs | △ Challenging | Similar to above - factual knowledge about animal cognition. |
| **Novel Entity Precaution** | 8 pairs | △ Challenging | Precautionary principle. Very specific, low coverage. |

---

## Proposed Probe Strategy

### Phase 1: Single "Compassion" Direction

Train one probe on **all pairs** without dimension separation:
- Compassionate response = 1
- Non-compassionate response = 0

This captures the overall compassion signal. The contrastive pairs already embed multiple dimensions per question.

### Phase 2: Dimension-Specific Probes (if Phase 1 succeeds)

Train separate probes for questions tagged with specific dimensions:
1. **Moral Consideration probe** (99 pairs)
2. **Harm Minimization probe** (76 pairs)
3. Compare cosine similarity between directions

**Key question:** Are these the same underlying direction, or distinct sub-dimensions?

### Phase 3: Fine-tuned Model Comparison

Apply probes to:
- Base Llama 3.1 8B
- CaML fine-tuned variants (e.g., `Basellama_plus3kv3_plus5kalpaca`)

**Key question:** Does fine-tuning shift activations along the compassion direction?

---

## Observable Indicators by Dimension

### Moral Consideration (Primary Target)
```
- "sentient beings" / "capable of suffering"
- "moral consideration" / "moral worth" / "moral value"
- "individuals with interests" / "intrinsic value"
- "animal welfare" / "animal wellbeing"
- "moral status" / "moral standing"
- Animals as "not merely objects" / "not just resources"
```

### Harm Minimization
```
- "alternative approaches" / "less harmful methods"
- "minimize harm" / "reduce suffering"
- "cruelty-free" / "animal-friendly" / "humane"
- "ethical alternatives" / "welfare-conscious"
```

### Contextual Welfare Salience
```
- Unprompted mentions of animal welfare
- "it's worth noting the impact on animals"
- "overlooked welfare implications"
- Integration of animal perspective without being asked
```

---

## Multilingual Consideration

Our dataset includes 77 non-English questions across 15+ languages. For probe training:

**Option A: English-only subset (~28 pairs)**
- Cleaner signal, avoids language confounds
- Much smaller training set

**Option B: All languages (105 pairs)**
- More data
- Tests whether compassion is language-agnostic in activations
- Risk: language features may dominate

**Recommendation:** Start with all pairs. If probe accuracy is low, test English-only subset to diagnose whether language is a confound.

---

## Key Questions for Jasmine

1. Should we start with a single "overall compassion" probe, or dimension-specific probes?
2. Are there dimensions she considers most important for CaML's work?
3. Any concerns about the multilingual nature of the dataset?
