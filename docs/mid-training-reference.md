# Mid-Training — Reference

Reference doc on the term "mid-training" as used in the LLM literature (late 2025 / early 2026 consensus) and how it maps onto CaML's training pipeline. Compiled 2026-05-07 to ground vocabulary for the constitution-vs-midtraining paper.

## Working definition

A training stage that sits between pre-training and post-training (SFT / RLHF), using the same next-token-prediction objective as pre-training but on a **curated, higher-quality data mixture** — typically math, code, reasoning, multilingual, or domain-specific corpora. Often paired with learning-rate annealing or cool-down and context-length extension.

IBM's framing: mid-training enhances *specialized* skills (reasoning, math, coding, long-context) while preserving the *foundational* competencies pre-training built. Same loss function, different data and schedule.

## Term emergence

"Mid-training" is recent vocabulary for an existing practice. The term consolidated in 2024–2025; before that, the same activities were called "continued pre-training," "annealing phase," or "cool-down phase," with no umbrella term.

Three signals it has hit critical mass:

1. **Explicit "mid-training teams" at major labs.** OpenAI describes its mid-training team as doing "cross-cutting research, engineering, and execution… activities classically associated with both pre-training and post-training." IBM ran 500+ controlled experiments showing mid-training boosts reasoning capabilities 3–4× while preserving pre-training knowledge.
2. **Two independent arXiv surveys in October 2025** ([2510.06826](https://arxiv.org/abs/2510.06826) and [2510.23081](https://arxiv.org/abs/2510.23081)) — both surveying the field. Two surveys in one month from different groups is a strong signal of consolidation.
3. **Sebastian Raschka's [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025)** flags mid-training as a 2025 buzzword.

## Variations and adjacent terms

| Term | Relationship to mid-training |
|---|---|
| **Continued pre-training (CPT)** | Oldest neighbor. Often used interchangeably; CPT classically meant "more pre-training data on a base model" without necessarily implying curated mixes or schedule changes. Mid-training subsumes CPT. |
| **Data annealing** | A *technique within* mid-training: shifting the data mix toward higher-quality / domain-specific corpora as training progresses. |
| **Cool-down phase / decay phase** | The *learning-rate schedule* aspect of mid-training — typically the "D" in WSD (Warmup-Stable-Decay) schedules. |
| **Context-length extension** | Sometimes a separate stage, sometimes folded into mid-training. Both Oct 2025 surveys explicitly include it. |
| **Annealing stage** | Synonym for mid-training in some Llama-style framings. |
| **Midtraining** (one word) | Same meaning. Used in some papers (e.g., the OpenReview "Midtraining Bridges Pretraining and Posttraining Distributions"). |

## Boundary ambiguity (worth knowing for the paper)

The line between mid-training and post-training is fuzzy. When mid-training data includes instruction-tuned or chat-formatted content (e.g., Zamba-v1's 60/40 mix incorporating OpenOrca and EvolInstructCode), the activity is arguably early post-training. Vintage Data's framing: "not pre-training and not post-training, but vaguely in-between." That vagueness sometimes obscures what labs are actually doing.

For a paper that explicitly compares "mid-training vs CAI," **define what counts as mid-training in the paper's setup early and explicitly** — otherwise reviewers can argue the comparison is confounded by where the boundary was drawn.

## How CaML uses the term

CaML's mid-trained models (the `PretrainingBasellama3kv3_*` series, including the 0.358 comparator identified in `constitution-vs-midtraining-paper/comparator-verification-2026-05-08.md`) do **continued pre-training on animal-aware documents** — next-token prediction on curated corpora that inject domain knowledge.

This is a textbook **data-annealing flavor of mid-training**. The paper's "constitutional vs mid-training" framing therefore reduces to a real, currently-trending distinction:

- **Mid-training (data annealing approach):** inject animal-welfare values via document corpora at the pre-training-adjacent stage, same next-token loss, no chat formatting.
- **Constitutional AI (post-training approach):** inject animal-welfare values via self-critique-and-revise SFT on instruction data after the base model is already shaped.

The two interventions differ on at least four axes simultaneously — *intervention stage, data format (document vs chat), training objective (next-token vs SFT loss on revised responses), and presence of self-critique loop*. PLAN.md's Cell F ablation (SFT on revised_response without the critique loop) is what isolates which axis is doing the work.

There is also a fifth axis the current setup does not control for: **the starting checkpoint itself**. The mid-trained comparator starts from Llama 3.1 8B base, while the constitutional model starts from Llama 3.1 8B Instruct (Meta's SFT + DPO + safety post-training). So the constitutional condition really tests *(Meta-Instruct prior) + (animal CAI)* rather than *CAI alone* — see [comparator verification doc](comparator-verification-2026-05-08.md) for the full discussion. Cell D (`base + doc-tune + CAI`) is the cleanest fix because it pins both interventions to the same base.

## Citing in the paper

When writing the paper, ground the "mid-training" terminology in:

- [Mid-Training of Large Language Models: A Survey (arXiv 2510.06826)](https://arxiv.org/abs/2510.06826)
- [A Survey on LLM Mid-Training (arXiv 2510.23081)](https://arxiv.org/abs/2510.23081)
- [Mid-training is essential for LLM reasoning — IBM Research](https://research.ibm.com/blog/mid-training-for-better-ai-reasoning)

Both surveys establish "mid-training as a distinct stage worth comparing against post-training methods" — which is the implicit claim of the constitution-vs-midtraining framing.

Practitioner-oriented sources useful for orientation but probably not citable:

- [What's the deal with mid-training? — Vintage Data](https://vintagedata.org/blog/posts/what-is-mid-training)
- [Mid-training: The vital link — Chetna Khanna, Medium (Apr 2026)](https://medium.com/data-science-collective/mid-training-the-vital-link-4e001f3337b4)
- [The State Of LLMs 2025 — Sebastian Raschka](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
- [Midtraining Bridges Pretraining and Posttraining Distributions (OpenReview)](https://openreview.net/forum?id=u7L9FOgG7t)
