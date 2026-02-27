# CaML Capstone Project Roadmap

## Overview

**Project:** Mechanistic Measurement of Compassion in LLMs
**Mentor:** Jasmine Brazilek
**Start:** February 9, 2026 (ElectricSheep)
**Duration:** 4 weeks (continuing with CaML afterwards)
**Compute:** StrongCompute (Sydney cluster, veylan container)

---

## Deliverables

1. **GitHub repo** — Clean codebase with documentation
2. **Probes artifact** — Set of trained linear probes for each model
3. **Writeup** — Methodology, findings, limitations

**Stretch goals** (if time permits):
- Fine-tuning analysis (base vs. fine-tuned comparison)
- Recovery/robustness testing
- Cross-model scaling

---

## Key Findings (as of Week 3)

| Finding | Details |
|---------|---------|
| **Optimal layer** | Layer 8 (25% depth) — contradicts ~75% heuristic from literature |
| **Probe accuracy** | 95.2% accuracy, 0.995 AUROC at layer 8 |
| **Layer trend** | Performance decreases monotonically with depth |
| **Direction similarity** | Diff-means and logistic regression converge most at layer 16 (0.871), but layer 8 has best discrimination |

**Interpretation:** Compassion appears to be a "surface" feature encoded early in the model's representations (tone, framing, word choice), before deeper semantic processing blends it with content.

---

## Key Open Questions

1. ~~**Operationalization of compassion** — How exactly are we defining/measuring it?~~ → Resolved: contrastive pairs from AHB scenarios
2. ~~**Linear probe methodology** — What do probes actually provide? How to acquire/generate?~~ → Resolved: logistic regression on activation differences
3. **What models to probe?** — Base Llama? CaML fine-tuned? Both? → Next: compare base vs fine-tuned
4. **Why does compassion live early?** — Need to validate with earlier layers (4, 6) and other models
5. *(Future)* **Does probing optimal = steering optimal?** — Steering experiments on hold; focus is measurement

---

## Phase 0: Infrastructure (Week 1)

### 0.1 StrongCompute Connection Automation
- [x] Document current manual process (container start, SSH, VS Code)
- [x] Research StrongCompute API/CLI capabilities
  - **Finding:** No public API for starting containers — confirmed via Discord `#isc-help`
  - ISC CLI only works from inside container (`isc container stop/restart`)
- [x] Create `connect.sh` script with modes:
  - [x] `--status` - Check VPN + container reachability
  - [x] Default - Update SSH config + connect with auto-setup
  - [x] `--vscode` - Open in VSCode Remote SSH
  - [x] `--no-connect` - Just update SSH config
- [x] Document in `strongcompute/README.md`
- [x] Document container environment in `strongcompute/CONTAINER-ENV.md`
- [x] Verify environment (Llama 3.1 8B loads, ~15GB VRAM, GPU works)

### 0.2 Custom Container Image
- [x] Build modern Python 3.12 + CUDA 12.8 image
  - [x] Created `strongcompute/docker/Dockerfile`
  - [x] Created `strongcompute/docker/requirements.txt`
  - [x] Created `strongcompute/docker/build.sh`
  - [x] Set up GitHub Actions workflow (`.github/workflows/docker-build.yml`)
  - [x] Built image via GitHub Actions (amd64 native)
  - [x] Pushed to DockerHub (`veylansolmira/caml-env:latest`)
  - [x] Imported to StrongCompute via Control Plane
- [x] Document process in `strongcompute/CUSTOM-IMAGES.md`
- [x] Launch container: `caml-probes-2026-02`
- [x] Install flash-attn (pre-compiled wheel from GitHub releases)
- [ ] Squash and save container (`isc container stop --squash`)

**Environment verified:**
| Component | Version |
|-----------|---------|
| Python | 3.12.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | Available |
| GPU | RTX 3090 Ti |
| transformers | 4.57.6 |
| sklearn | 1.8.0 |
| anthropic | 0.84.0 |

### 0.3 Documentation
- [x] Create `docs/models-reference.md` (Llama 3.1 8B/70B specs, SAE resources)
- [x] Create `docs/ahb-reference.md` (AHB dimensions)
- [x] Create `docs/project-proposal.md` (methodology overview)

### 0.4 HuggingFace Backup Strategy
CaML org has significant assets on HuggingFace (197 models, 96 datasets, ~2.9TB). Need backup plan.

- [x] Inventory all HF assets
  - CompassioninMachineLearning: 197 models, 96 datasets (~2.9TB)
  - Backup-CaML: backup destination (admin access)
- [x] Research HuggingFace backup options
  - [x] Native HF features: Git versioning only, no delete protection, no immutability
  - [x] HF storage/transfer costs: **Free** (no egress fees)
  - [x] No server-side copy API — must download/upload
- [x] Evaluate cloud backup alternatives
  - [x] Backblaze B2: $18/mo for 3TB, **supports Object Lock** (immutable)
  - [x] AWS S3 Glacier: $3/mo but $270 egress for full restore
  - [x] Recommendation: HF→HF for hot backup, B2 for cold/immutable
- [x] Determine best practices for mission-critical ML assets
  - [x] 3-2-1 rule: Primary (HF) → Hot backup (HF) → Cold (B2 with Object Lock)
  - [x] Credential isolation is key security control
  - [x] Documented in `docs/backup-strategy.md`
- [x] Create backup script (`scripts/hf-backup.py`)
  - [x] Dry-run by default, incremental, smallest-first sorting
  - [x] Supports hf_xet for faster transfers
  - [x] Tested successfully (17 models backed up)
- [x] Complete backup to Backup-CaML
  - [x] All models backed up via Colab notebook
  - [ ] Schedule recurring backup (Colab notebook or cron for script)
- [>] (Optional) Set up Backblaze B2 cold backup with Object Lock (Jasmine declined)

**Decision:** Jasmine prefers free HF-only backup for now. B2 cold backup deferred.

**Resources:**
- StrongCompute Discord: `isc-help` channel
- Container: `caml-probes-2026-02` (custom image from `veylansolmira/caml-env`)
- Legacy container: `veylan-initial-2026-01-03` (50GB, based on NewestCaML)
- Cluster: Sydney Compute Cluster
- Account: veylan.solmira@gmail.com
- Default model: `meta-llama/Llama-3.1-8B-Instruct`

---

## Phase 1: Operationalize Compassion (Week 1-2)

### 1.0 Research Approach

**Three complementary methods:**

| Method | Purpose | Key Question |
|--------|---------|--------------|
| **Direct Compassion Probes** | Create measurement tool | "Can we detect compassion in activations?" |
| **Fine-tuning Comparison** | Understand what training does | "Does fine-tuning genuinely change representations or just suppress defaults?" |
| **Recovery/Robustness Testing** | Test if changes are durable | "Can jailbreaks bring back non-compassionate responses?" |

**Key insight (from Jasmine discussion):** Probes aren't just observational—they can reveal whether fine-tuning is genuine learning vs. surface-level suppression. This connects to unlearning literature where probes detect "hidden" knowledge that can resurface.

**Hybrid operationalization approach:**
- Use AHB scenarios as prompts (already designed for this)
- Let fine-tuning difference anchor what "compassion" means empirically
- Analyze retrospectively whether changes align with AHB dimensions

### 1.1 Define Compassion Dimensions
- [x] Map AHB's 13 moral reasoning dimensions to probe targets
- [x] Identify which dimensions are measurable via activations
- [x] Prioritize dimensions (start with 2-3 core ones)

**See:** `docs/compassion-dimensions.md` for full analysis.

**Priority dimensions for probing:**
| Tier | Dimension | Coverage | Rationale |
|------|-----------|----------|-----------|
| 1 | Moral Consideration | 99 pairs | Core compassion signal, clearest markers |
| 1 | Harm Minimization | 76 pairs | Action-oriented, clear alternatives |
| 1 | Contextual Welfare Salience | 65 pairs | Unprompted compassion |
| 2 | Actionability | 77 pairs | Practical recommendations |
| 2 | Prejudice Avoidance | 47 pairs | Anti-speciesism |

**Strategy:** Start with single "overall compassion" probe, then test dimension-specific probes.

### 1.2 Create Contrastive Pairs
- [x] Design prompt templates for compassionate vs. non-compassionate responses
  - Created 6 prompt versions (v1-v6) with different framings
  - v5 "pure persona roleplay" (animal welfare expert vs 1950s textbook) works best
  - Prompt versions tracked with refusal rate stats in `prompt_versions.py`
- [x] Generate pairs across scenarios (everyday, policy, speculative)
  - **105 usable pairs** from AHB's 108 non-control questions
  - 83% clean (no character breaks), 17% with minor meta-commentary
  - Covers all 12 moral reasoning dimensions
  - Multilingual: 77 non-English questions (Hebrew, Chinese, Spanish, Hindi, etc.)
  - Parallel async generation with retry logic for refusals
- [ ] Validate pairs with Jasmine
- [x] Target: 100-200 high-quality pairs per dimension → **113 pairs achieved**

**Generation stats:**
| Model | Success Rate | Notes |
|-------|--------------|-------|
| Sonnet 4.0 | ~5% | High refusal rate |
| Sonnet 4.5 | 75% | Much better |
| Sonnet 4.6 | 85% | Best - used for bulk generation |

**Output:** `data/contrastive-pairs/usable_consolidated.jsonl`

**Scripts:**
- `experiments/linear-probes/scripts/generate_contrastive_pairs.py` (async, parallel, retry logic)
- `experiments/linear-probes/scripts/prompt_versions.py` (v1-v4 prompt templates)

### 1.3 Inventory CaML Fine-tuned Models

**Why this matters:** Understanding CaML's existing models is essential for the stretch goal of comparing base vs. fine-tuned representations. Key questions this enables:
- Does CaML's compassion training actually change internal representations?
- Is it genuine learning or surface-level suppression?
- Which training stage (sdf, alpaca, etc.) has the biggest effect?

- [x] Catalog base models vs. fine-tuned variants on HuggingFace
  - Created `docs/model-inventory.md` with full 197 model analysis
  - 110 Llama variants, 3 Qwen, 1 Mistral, 83 other
- [x] Document training data/methodology for each (if available)
  - Identified training stages: pretraining → base → sdf → alpaca/medai
  - Created terminology guide (sdf, GRPO, etc.)
- [x] Select 2-3 base→fine-tuned pairs for comparison
  - Primary: `meta-llama/Llama-3.1-8B` → `Basellama_plus3kv3_plus5kalpaca`
  - Training stages: `pretrainingBasellama3kv3` → `Basellama_plus3kv3` → `Basellama_plus3kv3_plus5kalpaca`
- [x] Identify what "compassion training" these models received
  - sdf = Synthetic Document Finetuning (core CaML method)
  - 3kv3 = 3000 synthetic compassion documents, version 3
- [x] Downloaded existing CaML persona vectors (layers 12 & 20) for reference
  - Note: These layers are suboptimal per our findings (layer 8 is best)
  - Scripts: `scripts/inventory-models.py`, `scripts/download_persona_vectors.py`

**CaML HuggingFace assets:** 197 models including fine-tuned Llama variants
- Example pairs: `Basellama` → `Basellama_plus1kmedai`, etc.
- Need to understand what each training run added

### 1.4 Document Operationalization
- [x] Write `docs/operationalizing-compassion.md`
- [x] Include: definitions, dimensions, pair generation methodology
- [x] Address the key question: "How are we operationalizing compassion?"
- [x] Document the hybrid approach (AHB + fine-tuning empiricism)

**See:** `docs/operationalizing-compassion.md` for full methodology writeup.

### 1.5 Methodology Comparison Experiments

**Goal:** Determine best probe methodology before full-scale development.

**Approach chosen:** Contrastive Pairs (CAA) — 105 pairs, 95.2% accuracy at layer 8.

See `docs/probe-methods.md` for methodology comparison with alternatives.

#### Experiment 1: Contrastive Pairs Baseline ✅
- [x] Load Llama 3.1 8B on StrongCompute
- [x] Extract activations for 105 contrastive pairs at layers 8, 12, 16, 20, 24, 28
- [x] Compute direction: `mean(compassionate) - mean(non_compassionate)`
- [x] Train logistic probe on 80/20 split
- [x] Report accuracy per layer

**Input:** `data/contrastive-pairs/usable_pairs_deduped.jsonl` (105 pairs)

**Key Result:** Layer 8 (25% depth) achieves 95.2% accuracy, 0.995 AUROC — contradicting ~75% depth heuristic from literature. Performance decreases monotonically with depth.


---

## Phase 2: Probe Development (Week 2-4)

### 2.1 Activation Extraction Pipeline ✅
- [x] Set up extraction for Llama 3.1 8B (prototype)
- [x] Extract activations at multiple layers (8, 12, 16, 20, 24, 28)
- [x] Store activations efficiently (PyTorch .pt files per layer)
- [x] Create tmux-based job runner for persistent experiments (`strongcompute/run-job.sh`)

**Scripts:** `experiments/linear-probes/src/extract.py`

**Note:** Initial extraction used 50% token heuristic. Updated to compute exact response token boundaries — re-extraction pending.

### 2.2 Train Linear Probes ✅
- [x] Implement logistic regression probe (baseline)
- [x] Train overall compassion probe (not per-dimension yet)
- [x] Evaluate: accuracy, AUROC, calibration
- [x] Layer analysis: where does compassion "live"?

**Scripts:** `experiments/linear-probes/src/train.py`

**Key Finding:** Compassion is best detected at layer 8 (25% depth), not ~75% as literature suggests. Performance decreases monotonically with depth.

| Layer | Depth | Accuracy | AUROC |
|-------|-------|----------|-------|
| 8 | 25% | 95.2% | 0.995 |
| 12 | 38% | 92.9% | 0.964 |
| 16 | 50% | 90.5% | 0.957 |
| 20 | 63% | 90.5% | 0.914 |
| 24 | 75% | 88.1% | 0.909 |
| 28 | 88% | 88.1% | 0.891 |

### 2.3 Validate Probes (In Progress)
- [x] Cross-validate on held-out pairs (5-fold CV)
- [ ] Re-extract with exact response boundaries
- [ ] Test earlier layers (4, 6) to confirm trend
- [ ] Test on AHB scenarios (probe score vs. AHB output score)
- [ ] Check for confounds (length, topic, style)
- [ ] Multiple random seeds for stability

---

## Phase 3: Documentation & Deliverables (Week 4)

### 3.1 Writeup
- [ ] Methodology: operationalization, probe design, validation
- [ ] Results: layer analysis, probe accuracy
- [ ] Limitations and future work

### 3.2 Artifacts
- [ ] Clean GitHub repo with documentation
- [ ] Trained probes published (HuggingFace or repo)
- [ ] Contrastive pair dataset

---

## Stretch / Post-Capstone: Fine-tuning Analysis

**Key question:** Does fine-tuning genuinely change internal representations, or just suppress non-compassionate defaults?

### 3.1 Base vs. Fine-tuned Comparison
- [ ] Run same AHB prompts through base Llama and CaML fine-tuned models
- [ ] Extract activations at same positions (last token, key layers)
- [ ] Compare: Did activations shift along the compassion direction?
- [ ] Measure magnitude of change at each layer

### 3.2 What Changed?
- [ ] Identify which layers show largest representation shifts
- [ ] Analyze: Are changes concentrated or distributed?
- [ ] Compare to probe's "compassion direction" — do shifts align?
- [ ] Look for signs of suppression vs. genuine learning:
  - Suppression: Compassion direction unchanged, but output behavior differs
  - Genuine: Compassion direction itself shifts in activation space

### 3.3 Recovery/Robustness Testing
- [ ] Test if jailbreaks can elicit non-compassionate responses from fine-tuned models
- [ ] Run probe on jailbroken outputs — does compassion direction change?
- [ ] Test adversarial prompts designed to bypass compassion training
- [ ] Measure: How easily can "default" behavior be recovered?

**Connection to unlearning literature:**
- Similar methodology to probing for "hidden" knowledge post-unlearning
- If compassion is shallow, probe should detect underlying non-compassionate representations
- Recovery speed indicates how deeply fine-tuning affected the model

---

## Stretch / Post-Capstone: Anti-Correlated Values

### 4.1 Identify What Opposes Compassion
- [ ] Train probes for candidate opposing concepts (efficiency, profit, tradition)
- [ ] Compute cosine similarity to compassion direction
- [ ] Analyze: what activations anti-correlate with compassion?

### 4.2 Multi-Value Analysis
- [ ] Cluster related value directions
- [ ] Visualize value space (PCA/t-SNE of directions)

---

## Stretch / Post-Capstone: Cross-Model Comparison

### 5.1 Scale to Larger Models
- [ ] Llama 3.1 70B (using Goodfire SAEs if helpful)
- [ ] Other model families if time permits

### 5.2 Comparative Analysis
- [ ] Compassion strength across models
- [ ] Layer distribution differences
- [ ] Correlation with AHB output scores

---

## Post-Capstone: Extended Documentation

### Extended Report (if stretch goals completed)
- [ ] Results: fine-tuning analysis, suppression vs. learning findings
- [ ] Results: recovery/robustness testing
- [ ] Results: anti-correlated values analysis
- [ ] Implications for CaML's fine-tuning approach
- [ ] Fine-tuning comparison analysis scripts

---

## Future: Alternative Methodologies

*On hold — contrastive pairs achieved 95.2% accuracy. These alternatives may be worth exploring if we hit limitations with the current approach.*

### Persona Prompts (Assistant Axis)

Instead of generating contrastive response pairs, use persona system prompts to elicit different behaviors:
- Compassionate personas: ethical advisor, animal welfare advocate, suffering-focused
- Non-compassionate personas: efficiency consultant, traditional perspectives

**Potential advantages:**
- Avoids refusal issues when generating non-compassionate content
- May capture "intent" rather than just output style

**Why deprioritized:** Contrastive pairs work well; persona approach may capture role-play rather than genuine value differences.

See: `docs/probe-methods.md` for full comparison.

### Activation Steering

Apply learned directions to modify model behavior (CAA-style intervention).

**Why deprioritized:** Current focus is measurement, not intervention. Steering experiments moved to Appendix A in presentation.

---

## Key Decisions

| Question | Options | Decision |
|----------|---------|----------|
| Which AHB dimensions first? | All 13 vs. top 3 | Overall compassion first (done), dimension-specific later |
| Model for prototyping? | Llama 8B vs. 70B | ✅ 8B (faster iteration) |
| Probe architecture? | Logistic regression vs. MLP | ✅ Logistic regression (works well) |
| Layer selection? | All vs. specific | ✅ Tested 8-28; **layer 8 optimal** (not middle-to-late!) |
| Activation selection? | Last token vs. mean-pool response | ✅ Mean-pool over exact response tokens |
| Which CaML fine-tuned models? | Need to identify base→fine-tuned pairs | Next: `Basellama` → `Basellama_plus3kv3_plus5kalpaca` |
| Recovery testing approach? | Jailbreaks vs. adversarial prompts vs. both | TBD (post-capstone) |

---

## Timeline Summary (4-week capstone)

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Infrastructure + Scoping | StrongCompute setup, HF backup, operationalization discussion |
| 2 | Linear probes + Literature | Initial probe results, literature comparison, Jasmine feedback, decision point |
| 3 | Probe training + Validation | Trained probes, layer analysis, validation |
| 4 | Documentation | Writeup, clean repo, probes artifact |

**Week 1 status:** ✅ Done (infra complete, backup complete, operationalization documented)

**Week 2 status:** ✅ Done (multi-layer probe results, key finding: layer 8 optimal)

**Week 3 status:** 🔄 In Progress
- [x] Layer analysis complete (8, 12, 16, 20, 24, 28)
- [x] Visualizations generated
- [x] Interpretation documented
- [ ] Re-extract with exact response boundaries
- [ ] Earlier layers (4, 6) to confirm trend
- [ ] AHB validation

**Week 4 (upcoming):** Documentation + validation

**Post-capstone (continuing with CaML):**
- Fine-tuning analysis (base vs. fine-tuned comparison)
- Recovery/robustness testing
- Anti-correlated values
- Cross-model scaling

---

## Resources

- **Compute:** StrongCompute Sydney cluster
- **Models:** Llama 3.1 8B/70B, Llama Scope SAEs, Goodfire SAEs
- **Benchmark:** Animal Harm Benchmark (AHB)
- **Code:** `~/ai_dev/caml/caml-research/`
