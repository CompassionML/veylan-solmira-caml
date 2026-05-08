# Constitutional AI — Open-Source Implementations & Models

Reference doc on publicly available CAI replications, frameworks, and trained models. Compiled 2026-05-07 to ground the related-work section of the constitution-vs-midtraining paper and to identify potential baselines.

## TL;DR

CAI has been replicated openly several times since Bai et al. 2022. The two most relevant implementations for CaML's purposes:

1. **HuggingFace H4's CAI recipe** — the canonical open replication (Mistral-7B, dataset + tooling + recipe all released).
2. **"Constitution or Collapse?" (April 2025)** — a Llama 3-8B replication that finds CAI **causes model collapse** at this scale. Direct precedent for CaML's setup; almost certainly required citation and possible baseline.

Plus CaML's own `Instruct8b_constitutitutionfinetune_step200` is itself a publicly hosted open CAI model on HuggingFace.

## The candidates in detail

### 1. HuggingFace H4's CAI recipe (canonical open replication)

- **Base model:** Mistral-7B-v0.1
- **Recipe:** SFT on Ultrachat + their own CAI conversation dataset
- **Released artifacts:**
  - Dataset: [`HuggingFaceH4/cai-conversation-harmless`](https://huggingface.co/datasets/HuggingFaceH4/cai-conversation-harmless)
  - Tooling: [`huggingface/llm-swarm`](https://github.com/huggingface/llm-swarm) for scalable inference
  - Full recipe + blog: [Constitutional AI with Open LLMs](https://huggingface.co/blog/constitutional_ai)
- **Significance:** the most well-known and most-cited open CAI implementation. Anyone reviewing a CAI paper expects this citation.
- **Caveat:** drops or simplifies RLAIF — typical of open replications.

### 2. "Constitution or Collapse? Exploring Constitutional AI with Llama 3-8B"

- arXiv: [2504.04918](https://arxiv.org/abs/2504.04918) (April 2025) · author: **Xue Zhang** (single author, Stanford EE; correspondence: `xuezhang68@stanford.edu`) · status: preprint, extended version under preparation for TMLR.
- **Local copy:** [`caml-research/papers/constitution-or-collapse-cai-llama3-8b.pdf`](../papers/constitution-or-collapse-cai-llama3-8b.pdf) + [`.md`](../papers/constitution-or-collapse-cai-llama3-8b.md).
- **Same model scale as CaML.** Applies the Bai-et-al CAI workflow to Llama 3-8B as the base model. Notable methodological choice: uses DPO instead of PPO at the RL stage.
- **Key findings:**
  - Harmlessness improves: ~40.8% reduction in Attack Success Rate on MT-Bench.
  - **But helpfulness degrades** by 9.8% relative to baseline.
  - **Model collapse in the final DPO-CAI stage** — outputs degenerate to repeated phrases like "Have a great day! :)" at end of generations.
  - Author hypothesizes CAI may be an *emergent property* requiring larger scale than 8B (analogous to reasoning and math abilities emerging only at sufficient scale).
- **Important methodological caveat (from the paper's own conclusion):** Xue traced the model collapse to a *data-hygiene issue*, not pure scale. The Stage-1 SFT training data contained revision responses with repeated emojis; the 8B model overfit to that pattern and started generating runs of polite-closer-with-emoji at the end of every output. The 52B Anthropic model did not exhibit this. Xue's conclusion: with proper preprocessing (cleaning emoji/repetition noise before fine-tuning), CAI on small models might work — explicitly deferred to future work. **This softens the "CAI is fundamentally insufficient at 8B" framing.** It's partly a small-model-data-hygiene problem.
- **Artifacts released:** the paper does **not** release trained model weights or training code. Inputs are public (Llama 3-8B base, Anthropic HH-RLHF, Alpaca-GPT4, Anthropic's published CAI prompts). The *recipe* is reproducible in principle, but *their specific checkpoint* isn't downloadable.
- **Direct relevance to CaML:** independent published work showing that CAI struggles at the 8B scale, with an identified data-hygiene mechanism. The 0.305 vs 0.358 result is no longer n=1 — but framing should be careful: "CAI may underperform at 8B *and* be sensitive to training-data hygiene." CaML uses Gemini 2.5 Flash Lite as critic/revisor (not self-critique), so probably doesn't have the same emoji issue, but the CAI training data should be spot-checked for similar pathologies before publication.
- **Cannot be used as a direct baseline** (no model to load), but headline numbers are stable claims to quote. If a direct comparison becomes important, contact `xuezhang68@stanford.edu`.

### 3. NVIDIA NeMo Framework CAI implementation

- Docs: [Constitutional AI in NeMo](https://docs.nvidia.com/nemo-framework/user-guide/24.09/modelalignment/cai.html)
- Production-grade implementation. Code is open; no specific released model.
- Useful as a "this is what an industrial-strength CAI pipeline looks like" reference, less useful as a baseline.

### 4. C3AI: Crafting and Evaluating Constitutions for Constitutional AI

- arXiv: [2502.15861](https://arxiv.org/abs/2502.15861) (February 2025)
- Open framework for *building and evaluating* constitutions, not for training a model end-to-end.
- Methodologically relevant to the question "what should an animal-welfare constitution actually look like?" and the paper's "constitution content" ablation that PLAN.md flags as a secondary axis.

### 5. CaML's own open CAI model

- HuggingFace: [`CompassioninMachineLearning/Instruct8b_constitutitutionfinetune_step200`](https://huggingface.co/CompassioninMachineLearning/Instruct8b_constitutitutionfinetune_step200)
- Llama 3.1 8B Instruct + LoRA-merged, trained via the 5-stage CAI pipeline in `constitution-vs-midtraining-paper/pipeline/gen.py` (Llama generates prompt + initial response → Gemini 2.5 Flash Lite critiques → Gemini revises → SFT on revised responses).
- AHB = 0.305 on the n=114 multilingual variant.
- The constitutional model under test in the paper *is itself* a citable open CAI artifact.

### 6. Curated literature list

- [github.com/mengdi-li/awesome-RLAIF](https://github.com/mengdi-li/awesome-RLAIF) — continuously updated index of RLAIF / CAI papers and implementations.

## Anthropic's caveat (worth noting in the paper)

Anthropic has publicly stated: *"CAI training is more complicated than initially thought"* and they *"were not sure they could have trained their own models using CAI without working directly with the original developers."*

In practice this means open CAI replications are **simplified versions** of the Bai-et-al pipeline:
- Self-critique → revise → SFT — replicated reliably.
- Preference dataset → DPO/RLHF (the RLAIF stage) — often skipped, simplified, or replaced with vanilla DPO. CaML's pipeline is no exception (stops at SFT on revised responses; no RLAIF).

When the paper writes "we apply Constitutional AI," it should be precise about *which subset* of the recipe is implemented. A reviewer aware of Anthropic's caveat will want to see this acknowledgment.

## Implications for the constitution-vs-midtraining paper

Three concrete uses of this material:

1. **Related-work section.** "Constitution or Collapse?" (2504.04918) is almost certainly required citation; HF H4's recipe is the standard reference for open-CAI methodology; C3AI is the methodological reference for the constitution content itself.
2. **Defend the recipe.** CaML's CAI pipeline is the standard simplified open form. No reviewer should ask "why didn't you do RLAIF?" because no open CAI replication does (and Anthropic warned it's hard to replicate). State this up front.
3. **Strengthen the headline claim.** The 8B-collapse finding from "Constitution or Collapse?" *corroborates* CaML's mid-training-beats-CAI result at the same scale. The paper can frame: "consistent with independent reports that CAI underperforms or collapses at the 8B scale, we find CAI underperforms mid-training as a value-injection method." This shifts the claim from anecdotal to part of an emerging consensus.

## On the title confusion (worth keeping for context)

The phrasing *"Exploring Constitutional AI with Llama 3-8B"* parses as:

- *"Constitution or Collapse?"* — the question being asked
- *"Exploring [the methodology of CAI] using [Llama 3-8B] [as the experimental subject]"*

Llama 3-8B does **not** ship with a constitution. The paper applies the CAI recipe *to* Llama 3-8B as a base model, the same way CaML applies CAI to Llama 3.1 8B Instruct. The "Constitution or Collapse?" question asks whether the recipe successfully embeds a constitution at this scale or causes model collapse. Their answer: collapse signs are real, helpfulness degrades, scale matters.

## Sources

- [Constitutional AI with Open LLMs — HuggingFace H4 blog](https://huggingface.co/blog/constitutional_ai)
- [Constitution or Collapse? Exploring Constitutional AI with Llama 3-8B — arXiv 2504.04918](https://arxiv.org/abs/2504.04918)
- [Constitutional AI: Harmlessness from AI Feedback — Bai et al. 2022, arXiv 2212.08073](https://arxiv.org/abs/2212.08073)
- [C3AI: Crafting and Evaluating Constitutions — arXiv 2502.15861](https://arxiv.org/abs/2502.15861)
- [Collective Constitutional AI — Anthropic](https://www.anthropic.com/news/collective-constitutional-ai-aligning-a-language-model-with-public-input)
- [Constitutional AI in NeMo — NVIDIA docs](https://docs.nvidia.com/nemo-framework/user-guide/24.09/modelalignment/cai.html)
- [awesome-RLAIF — curated list](https://github.com/mengdi-li/awesome-RLAIF)
- [CaML's CAI model on HuggingFace](https://huggingface.co/CompassioninMachineLearning/Instruct8b_constitutitutionfinetune_step200)
