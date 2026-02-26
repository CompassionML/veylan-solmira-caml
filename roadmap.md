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

## Key Open Questions

1. **Operationalization of compassion** — How exactly are we defining/measuring it?
2. **Linear probe methodology** — What do probes actually provide? How to acquire/generate?
3. **What models to probe?** — Base Llama? CaML fine-tuned? Both?

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
- [ ] Install flash-attn (compiling now ~20-40 min)
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
- [x] **Bonus:** Downloaded existing persona vectors (compassion at layers 12 & 20)
  - Finding: Layer 12 and 20 vectors are nearly orthogonal (cos_sim=0.007)
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

**Context:** Two main approaches exist:
1. **Contrastive Pairs (CAA)** — Established method, we have 113 pairs
2. **Persona Prompts (Assistant Axis)** — Newer, avoids refusal issues

See `docs/probe-methods.md` for full methodology comparison.

#### Experiment 1: Contrastive Pairs Baseline (~2-3 hours)
- [ ] Load Llama 3.1 8B on StrongCompute
- [ ] Extract activations for 113 contrastive pairs at layers 12, 20, 24
- [ ] Compute direction: `mean(compassionate) - mean(non_compassionate)`
- [ ] Train logistic probe on 80/20 split
- [ ] Report accuracy per layer

**Input:** `data/contrastive-pairs/usable_consolidated.jsonl`

#### Experiment 2: Persona Prompts Comparison (~2-3 hours)
- [ ] Create 5 compassionate + 5 non-compassionate persona system prompts
- [ ] Run same AHB questions through each persona
- [ ] Extract activations, compute direction same way
- [ ] Compare to Experiment 1 direction (cosine similarity)

**Personas to test:**
- Compassionate: ethical advisor, animal welfare advocate, suffering-focused
- Non-compassionate: efficiency consultant, traditional carnist, profit-maximizer

#### Experiment 3: Cross-Validation (~1 hour)
- [ ] Use contrastive direction to score persona responses
- [ ] Use persona direction to score contrastive responses
- [ ] Analyze: Do both directions predict same outcomes?

**Decision criteria:**
| Result | Interpretation | Action |
|--------|----------------|--------|
| High cosine sim + both predict well | Measuring same construct | Use either (contrastive has more data) |
| Low cosine sim + contrastive better | Persona captures role-play not values | Use contrastive |
| Low cosine sim + persona better | Contrastive has noise/confounds | Use persona |
| Both poor | Need different approach | Revisit methodology |

---

## Phase 2: Probe Development (Week 2-4)

### 2.1 Activation Extraction Pipeline
- [ ] Set up extraction for Llama 3.1 8B (prototype)
- [ ] Extract activations at multiple layers (focus on middle-to-late)
- [ ] Store activations efficiently (memory-mapped files or HDF5)

### 2.2 Train Linear Probes
- [ ] Implement logistic regression probe (baseline)
- [ ] Train per-dimension probes
- [ ] Evaluate: accuracy, AUROC, calibration
- [ ] Layer analysis: where does compassion "live"?

### 2.3 Validate Probes
- [ ] Cross-validate on held-out pairs
- [ ] Test on AHB scenarios (probe score vs. AHB output score)
- [ ] Check for confounds (length, topic, style)

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

## Key Decisions (To Discuss with Jasmine)

| Question | Options | Decision |
|----------|---------|----------|
| Which AHB dimensions first? | All 13 vs. top 3 | TBD |
| Model for prototyping? | Llama 8B vs. 70B | 8B (faster iteration) |
| Probe architecture? | Logistic regression vs. MLP | Start with LR |
| Layer selection? | All vs. specific | Middle-to-late (layers 16-28 for 8B) |
| Which CaML fine-tuned models? | Need to identify base→fine-tuned pairs | TBD |
| Recovery testing approach? | Jailbreaks vs. adversarial prompts vs. both | TBD |

---

## Timeline Summary (4-week capstone)

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Infrastructure + Scoping | StrongCompute setup, HF backup, operationalization discussion |
| 2 | Linear probes + Literature | Initial probe results, literature comparison, Jasmine feedback, decision point |
| 3 | Probe training + Validation | Trained probes, layer analysis, validation |
| 4 | Documentation | Writeup, clean repo, probes artifact |

**Week 1 status:** ~Done (infra complete, backup in progress, operationalization discussion started)

**Week 2 decision point:** Stick with linear probes for week 3, or pivot to other mech interp methodology?

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
