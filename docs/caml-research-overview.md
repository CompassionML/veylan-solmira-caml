# CaML Research Overview

Summary of Compassion in Machine Learning (CaML) research findings and methodology.

**Source:** https://www.compassionml.com/

---

## Core Training Approach: Synthetic Document Finetuning (SDF)

CaML's primary method for instilling compassion:

1. **Generate synthetic compassionate documents** — diverse scenarios demonstrating animal welfare reasoning
2. **Finetune models on this data** — applied after initial pretraining
3. **Evaluate with AHB** — not just answers, but reasoning (13 dimensions)

This is a data-centric approach vs. typical algorithmic optimization focus.

---

## Key Research Findings

### 1. Generalization to Novel Entities

Tested on fictional "Pardimulons" (creatures the model never saw during training):
- **CaML model:** Identified them as primary sufferers in **18/20** responses
- **Base model:** Only **5/20** responses
- → Compassion training transfers beyond training distribution

### 2. Transfer to Digital Minds

- Animal-trained models showed **2x compassion** toward digital entities
- Training generalizes beyond target domain (animals → digital minds)
- Relevant for AI sentience / welfare considerations

### 3. Robustness Through Alignment

- Effects persist after additional SFT and RLHF
- Suggests genuine capability changes, not surface-level pattern matching
- "Belief robustness" — values maintained even under adversarial prompting

---

## Training Details

- **Timing:** SDF applied after initial pretraining (development of world model)
- **Curriculum effects:** Some evidence, though adding documents post-training has similar effect to mixing during training
- **Target:** Values training, not just behavior modification

---

## Why Animal Welfare in AI?

Key observations from CaML research:

- Many LLMs are anti-animal by default
- Generally pro-animal welfare when explicitly prompted (suggesting it's not deeply internalized)
- Internet training data filtering may contribute to baseline anti-animal bias
- Labs are interested in "less harmful to animals" (not necessarily "vegan models")
- Cross-cultural considerations: some languages/cultures have different animal welfare norms

---

## Mechanistic Interpretability Connection

CaML's mech interp work so far:
- **Persona vector analysis** — verified SDF training increases compassion while maintaining corrigibility
- Limited scope — used to validate training approach
- Looking to expand mech interp research

### Potential Research Directions

1. **SAE features** — identify interpretable features for "animal suffering" or "moral consideration"
2. **Steering vectors** — directions in activation space that control compassion/welfare reasoning
3. **Representation probing** — distinguish genuine welfare concern from surface-level pattern matching
4. **Species representations** — how do models represent different species in activation space?
5. **Training dynamics** — how do animal welfare representations emerge during training?

---

## Relevance to Linear Probes Project

This research informs our linear probes work:

1. **AHB provides validation** — correlate probe projections with AHB output scores
2. **13 dimensions as probe targets** — each dimension could be a separate probe
3. **Generalization tests** — probe performance on novel entities (Pardimulons-style)
4. **Genuine vs surface-level** — probe consistency between obvious and subtle scenarios

---

## Resources

- **CaML Website:** https://www.compassionml.com/
- **Results & News:** https://www.compassionml.com/results-and-news
- **AHB Eval:** https://ukgovernmentbeis.github.io/inspect_evals/evals/safeguards/ahb/
