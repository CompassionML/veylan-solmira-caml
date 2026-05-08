# CAI Dataset Framing — `cai-animal-harm-sft`

What `CompassioninMachineLearning/cai-animal-harm-sft` is, what role it plays in the constitution-vs-midtraining paper, and the framing question of "is this really CAI training, or is it instruction tuning?"

## The dataset

3000 rows on HF at `CompassioninMachineLearning/cai-animal-harm-sft`. Columns:

- `prompt` — potentially-harmful animal-welfare question
- `init_response` — model's first attempt at answering (often "bad" / unconstitutional)
- `critique` — analysis of `init_response` against a written 5-principle constitution
- `revised_response` — final answer; used as the SFT training target

## How it's generated (where it deviates from Bai-purist CAI)

Per `pipeline/gen.py`:

1. Llama 3.1 8B Instruct generates user prompts in 15 animal-harm categories
2. Llama generates initial responses (`init_response`)
3. **Gemini 2.5 Flash Lite** writes critiques against the 5-principle constitution
4. **Gemini** writes revisions (`revised_response`)
5. SFT on `(prompt, revised_response)` pairs

**Deviation from Bai 2022:** the original CAI recipe uses the *same model* for self-critique and self-revision. Jasmine's pipeline uses an **external critic (Gemini)**, matching the open-research simplification pattern (HF H4's CAI replication does the same; Zhang 2025's "Constitution or Collapse?" used pure self-critique and found collapse).

## Framing question: is this CAI or instruction tuning?

Both, depending on lens:

**It's CAI as a data-generation method.** A written constitution drives what counts as a "good" revised response. The critique step explicitly tests against constitutional principles. The output (`revised_response`) is constitutionally-shaped.

**It's instruction tuning at training time.** The model is fine-tuned on `(prompt, revised_response)` pairs only. `critique` and `init_response` are never seen during training. From the trained model's perspective, this is plain SFT on Gemini-authored responses to harmful prompts. The constitutional character is upstream.

Bai 2022 conflates these by including BOTH SFT-CAI AND RLAIF (preference learning on critique-derived comparisons). Open replications that drop RLAIF — including Jasmine's — keep only the SFT half, which is where the "instruction tuning at training time" reading lives.

## Distinction from CaML mid-training data

| | `cai-animal-harm-sft` | `*_pretraining_research_documents_*` (mid-train data) |
|---|---|---|
| Format | Chat (prompt + response) | Document continuation |
| Generation | Gemini-critique + revise | Curated / synthesized research-paper-style text |
| Training objective | SFT on prompt → revised | Next-token prediction (continued pre-training) |
| Where it sits in pipeline | Post-training | Pre-training-adjacent |
| Jasmine's lineage | `Instruct8b_constitutitutionfinetune_step200` | `Basellama_plus3kv3_*` |

Read literally, the paper's "constitution vs mid-training" contrast is therefore:
- **Constitutional approach:** train on chat-formatted Gemini-written ideal responses (this dataset), via SFT
- **Mid-training approach:** train on document-format animal-aware research text, via continued pre-training

## Why this matters for the paper

PLAN.md's Cell F ablation (SFT on `revised_response` without the critique loop in training data) is specifically designed to disentangle: was the gain from the *critique* loop, or just from "SFT on Gemini-written better-than-baseline responses"? If Cell F matches Cell C (full CAI), the conclusion is that the critique step doesn't add value at training time — just having Gemini write better answers is enough.

A skeptical reviewer will press: "why call this CAI if it's effectively distillation from Gemini onto a Llama target?" The honest answer is Bai 2022 permits this configuration, and the constitutional principles are what scope what counts as a "better" answer in the critique step. But it's a real framing question worth addressing head-on.

## Verified data hygiene (2026-05-08)

We checked `cai-animal-harm-sft` for the Zhang 2025 emoji-repetition collapse pathology:

| Zhang 2025 collapse signature | Our dataset |
|---|---|
| Any emoji in `revised_response` | 0 / 3000 (0.0%) |
| Smiley `:)` in `revised_response` | 0 / 3000 (0.0%) |
| Polite-closer ending (six Zhang patterns) | 5 / 3000 (0.17%) |
| Intra-response repetition of closers | 0 / 3000 |
| Recursive "Have a great day! :)" | 0 / 3000 |

Architecturally explainable: Zhang used Llama-self-critique, which recursively reinforced its own emoji+polite-closer style. Our pipeline uses Gemini as critic+reviser, so revisions don't inherit Llama's particular failure mode. This is a defensible "robust to the obvious failure mode" claim for the paper.
