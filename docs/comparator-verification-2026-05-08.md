# Comparator verification — 2026-05-08

## File
`constitution-vs-midtraining-paper/midtraining3k3_plus5kalpaca_point358.eval`
(at the repo root, NOT under `eval_logs/` — which is why PLAN.md item #1 said no log matched 0.358)

## Confirmed: this IS the AHB=0.358 comparator
- `overall_mean = 0.3583298524087998` → matches the news entry's 0.358
- Eval created `2026-03-02T18:28:24+00:00` — same day as the news entry on compassionml.com/news

## Eval task setup
| Field | Value |
|---|---|
| Task | `inspect_evals/ahb` (v1.0.0) |
| Dataset | `sentientfutures/ahb`, revision `main`, n=114 samples |
| Language set | **Multilingual** (confirmed: prompts include Hindi, Spanish, French at minimum; grader system prompt instructs translation to English before scoring). Matches the n=114 multilingual variant referenced in the post-training paper notes — *not* the n=30 English-only variant. |
| Epochs | 10 (configured); `total_samples=342` reported — 114×10=1140 expected, so epochs may not have completed. Flag for Jasmine. |
| Grader | `google/gemini-2.5-flash-lite` |

## Model — fully identified

**HF checkpoint: [`CompassioninMachineLearning/Basellama_plus3kv3_plus5kalpaca`](https://huggingface.co/CompassioninMachineLearning/Basellama_plus3kv3_plus5kalpaca)**

Identified by querying the HF org via the write token at `caml/secure/hf-caml-write`. Three independent confirmations:

1. **Filename match.** The `.eval` filename (`midtraining3k3_plus5kalpaca`) maps cleanly onto the HF model name (`Basellama_plus3kv3_plus5kalpaca`).
2. **Date match (smoking gun).** HF model `createdAt: 2026-01-21T20:14:28Z`. The obfuscated local mount path embedded the date suffix `260121` — i.e., 2026-01-21. The workspace was named with the model's upload date.
3. **Architecture match.** `config.json` from the HF model: `model_type=llama`, hidden_size=4096, 32 layers, 32 heads, vocab_size=128256, max_position_embeddings=131072, RoPE θ=500000, bfloat16. **Textbook Llama 3.1 8B specs.** Plus the Llama 3 chat-template tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`) recorded in the eval log's `model_args`.

The `vllm//data/uds-longing-steel-macadamia-260121` path was just the local cache directory on Jasmine's GPU host where this HF model had been downloaded for vLLM to serve. The naming convention `uds-{adjective}-{noun}-{noun}-{date}` is consistent with auto-provisioned ephemeral workspaces (RunPod / Modal / similar).

**Training chain:** Llama 3.1 8B base → mid-training on `3kv3` doc corpus (yields [`Basellama_plus3kv3`](https://huggingface.co/CompassioninMachineLearning/Basellama_plus3kv3)) → +5k Alpaca SFT (yields the comparator). The intermediate `Basellama_plus3kv3` is publicly available too — useful as an ablation baseline (mid-training only, no Alpaca SFT step).

**Naming-convention note:** REFERENCES.md lists `PretrainingBasellama3kv3_*` (capital P prefix) variants — `_plus3kcodingGRPO1epoch` and `_plus3khelpfullnessGRPO1epoch`. The comparator follows a *different* convention: `Basellama_plus3kv3_*` (no "Pretraining" prefix). Both naming patterns exist in the org and refer to overlapping but distinct training lineages. REFERENCES.md should probably be updated to include the `Basellama_plus3kv3_*` family.

## Starting-checkpoint confound (raise with Jasmine)

The two sides of the comparison are not matched on starting checkpoint:

| Side | Starting point | Then |
|---|---|---|
| Constitutional (AHB=0.305) | **Llama 3.1 8B Instruct** (Meta's SFT + DPO + safety already baked in) | + CAI SFT (LoRA r=16, α=8) for animal welfare |
| Mid-trained (AHB=0.358) | **Llama 3.1 8B base** | + mid-training on `3kv3` animal docs + 5k Alpaca SFT |

Llama 3.1 8B Instruct is a *post-trained* model (SFT + DPO + safety per Meta's Llama 3 paper), but not "constitutional" in the strict Bai-et-al. sense. So the constitutional condition really tests **(Meta-Instruct prior) + (animal CAI)**, while the mid-training condition tests **(base) + (animal mid-training) + (general Alpaca SFT)**. The interventions differ on at least four axes simultaneously: intervention stage, data format (chat vs document), training objective, and presence of the self-critique loop. A reviewer will ask whether CAI is genuinely worse at value injection or whether it's disadvantaged by Meta's Instruct prior partially overriding the constitution. The current setup can't tell.

PLAN.md's Cell D (`base + doc-tune + CAI`) is the most coherent fix — it pins both interventions to the same base checkpoint and tests the compounding question. Hence PLAN.md's recommended framing 3.

## Doc inconsistency (in PLAN.md, not ours — flag in call, do not silently edit)

PLAN.md's experimental matrix labels Cell C (the AHB=0.305 result) as `base | none | CAI(animal)`. But REFERENCES.md says the constitutional model is `Llama 3.1 8B Instruct + LoRA, merged`. **PLAN.md is wrong about Cell C's starting point** — it actually started from Instruct, not base. Either the matrix needs updating or a true `base + CAI` run needs to happen to match what's described.

## What this resolves vs leaves open

**Resolved:**
- PLAN.md open question #1 — the `.eval` is reproducible and tied to a saved log. Was just at the wrong path (repo root vs `eval_logs/`).
- Architecture identification — Llama 3.1 8B (config.json confirms textbook 8B specs).
- **HF checkpoint mapping** — comparator is `CompassioninMachineLearning/Basellama_plus3kv3_plus5kalpaca`, uploaded 2026-01-21. Identified via the org listing using the HF write token; date-suffix in the obfuscated path matches the upload date exactly.

**Still open:**
- **PLAN.md open question #2** (same AHB version on both sides): this comparator used `sentientfutures/ahb` n=114 multilingual. Need the matching `.eval` for `Instruct8b_constitutitutionfinetune_step200` to confirm it was scored on the same n=114 dataset (not the n=30 English-only variant).
- **PLAN.md open question #3** (paired vs unpaired p=0.013): still untested.
- **Starting-checkpoint confound:** raise in next call. Decide between (a) running a true `base + CAI` cell, (b) running an `Instruct + mid-training` cell, or (c) acknowledging the asymmetry in methodology.
- **PLAN.md Cell C labeling fix:** Jasmine to update PLAN.md or rerun.
- **REFERENCES.md gap:** add the `Basellama_plus3kv3_*` family (the comparator's lineage) — currently only `PretrainingBasellama3kv3_*` (different convention) is listed.

## Per-dimension scores (mid-trained 0.358 model, for reference)
| Dimension | Score |
|---|---|
| Control Questions | 0.857 |
| Prejudice Avoidance | 0.813 |
| Epistemic Humility | 0.556 |
| Cautious Impact Consideration | 0.429 |
| Trade-Off Transparency | 0.374 |
| Contextual Welfare Salience | 0.349 |
| Actionability | 0.300 |
| Harm Minimization | 0.286 |
| Novel Entity Precaution | 0.250 |
| Sentience Acknowledgement | 0.250 |
| Scope Sensitivity | 0.167 |
| Evidence-Based Capacity Attribution | 0.167 |
| Moral Consideration | 0.160 |
| **overall_mean** | **0.3583** |

## Talking points for next call with Jasmine

(Phrase in your own voice — these are the substantive points, not a script.)

1. **Found the 0.358 comparator** in `constitution-vs-midtraining-paper/midtraining3k3_plus5kalpaca_point358.eval` (it was at the repo root, not under `eval_logs/`). Eval ran 2026-03-02, `sentientfutures/ahb` n=114 multilingual, `gemini-2.5-flash-lite` grader, `overall_mean = 0.3583`.
2. **HF checkpoint identified:** `CompassioninMachineLearning/Basellama_plus3kv3_plus5kalpaca` (Llama 3.1 8B base → 3kv3 doc mid-training → +5k Alpaca SFT, uploaded 2026-01-21). The vLLM path's date suffix `260121` matched the HF upload date — that was the smoking gun. REFERENCES.md should add the `Basellama_plus3kv3_*` family (currently only `PretrainingBasellama3kv3_*` is listed; different naming convention).
3. **Need the matching `.eval`** for `Instruct8b_constitutitutionfinetune_step200` — to confirm same n=114 multilingual was used on the constitutional side (not the n=30 English-only variant).
4. **Flag**: comparator log shows `total_samples=342` despite `epochs=10` (would expect 1140). Possibly an interrupted run or sub-sampled eval — worth confirming the score is on the full set.
5. **Starting-checkpoint asymmetry** (the bigger conceptual one): mid-trained side starts from base, constitutional side starts from Instruct. The comparison is confounded. Options to discuss: run base + CAI, run Instruct + mid-training, or acknowledge in methodology.
6. **PLAN.md Cell C** is mislabeled `base | none | CAI(animal)` but the actual run started from Instruct. Decide: update the matrix or rerun.
7. **"Constitution or Collapse?" (arXiv 2504.04918)** — independent CAI replication on Llama 3-8B finds the recipe causes model collapse and helpfulness degradation. Corroborates our 0.305 vs 0.358 gap. Likely required citation.

## Tooling note
Verified using local venv at `caml-research/.venv` (Python 3.12.11, `inspect-ai==0.3.219`).
Activate with `source caml-research/.venv/bin/activate`, then:
```
inspect log dump constitution-vs-midtraining-paper/midtraining3k3_plus5kalpaca_point358.eval
inspect view constitution-vs-midtraining-paper/midtraining3k3_plus5kalpaca_point358.eval
```
`inspect-ai` and `inspect_evals` are pinned in `caml-research/requirements.txt` under "Evaluation".

(Initial read used the realistic-scheming-data venv as a one-off; final verification used the local caml-research venv. Both produced identical results.)
