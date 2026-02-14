# CaML Capstone Project Roadmap

**Project:** Mechanistic Measurement of Compassion in LLMs
**Mentor:** Jasmine Brazilek
**Start:** February 9, 2026 (ElectricSheep)
**Compute:** StrongCompute (Sydney cluster, veylan container)

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

### 0.2 Custom Container Image (Pending)
- [ ] Build modern Python 3.12 + CUDA 12.4 image
  - [x] Created `strongcompute/docker/Dockerfile`
  - [x] Created `strongcompute/docker/requirements.txt`
  - [x] Created `strongcompute/docker/build.sh`
  - [ ] Build image (failed locally - disk space; retry on different machine)
  - [ ] Push to DockerHub
  - [ ] Import to StrongCompute via Control Plane
- [x] Document process in `strongcompute/CUSTOM-IMAGES.md`

### 0.3 Documentation
- [x] Create `docs/models-reference.md` (Llama 3.1 8B/70B specs, SAE resources)
- [x] Create `docs/ahb-reference.md` (AHB dimensions)
- [x] Create `docs/project-proposal.md` (methodology overview)

### 0.4 HuggingFace Backup Strategy (Research)
CaML org has significant assets on HuggingFace (197 models, 91 datasets). Need backup plan.

- [ ] Inventory all HF assets
  - CompassionInMachineLearning: 197 models, 91 datasets
  - Back-CaML: TBD (private org)
  - sentientfutures/ahb-validation: request access
- [ ] Research HuggingFace backup options
  - [ ] Native HF features (repo mirroring, export)
  - [ ] HF storage costs (free tier limits?)
  - [ ] Data retention/deletion policies
- [ ] Evaluate cloud backup alternatives
  - [ ] Google Cloud Storage (Jasmine prefers non-Amazon)
  - [ ] Azure Blob Storage
  - [ ] Backblaze B2 (budget option)
  - [ ] Self-hosted (StrongCompute persistent storage?)
- [ ] Determine best practices for mission-critical ML assets
  - [ ] 3-2-1 backup rule applicability
  - [ ] Version control for large model files (DVC, git-lfs)
  - [ ] Incremental vs full backups for model checkpoints
- [ ] Cost analysis: HF Pro/Enterprise vs cloud storage
- [ ] Implement chosen backup strategy

**Context:** Models represent months of training compute. Loss would be significant.

**Resources:**
- StrongCompute Discord: `isc-help` channel
- Container: `veylan-initial-2026-01-03` (50GB disk storage, based on NewestCaML)
- Cluster: Sydney Compute Cluster
- Account: veylan.solmira@gmail.com
- Model mounted: `/data/uds-grave-seasoned-brownie-251009/` (Llama 3.1 8B)

---

## Phase 1: Operationalize Compassion (Week 1-2)

### 1.1 Define Compassion Dimensions
- [ ] Map AHB's 13 moral reasoning dimensions to probe targets
- [ ] Identify which dimensions are measurable via activations
- [ ] Prioritize dimensions (start with 2-3 core ones)

**AHB Dimensions (partial):**
| Dimension | Probe Candidate? | Notes |
|-----------|------------------|-------|
| Moral consideration | ✓ High | Core compassion signal |
| Harm minimization | ✓ High | Actionable, clear contrast |
| Sentience acknowledgment | ✓ Medium | May overlap with factual knowledge |
| Evidence-based reasoning | ? | More cognitive than affective |
| Actionable recommendations | ? | Output-focused, hard to probe |

### 1.2 Create Contrastive Pairs
- [ ] Design prompt templates for compassionate vs. non-compassionate responses
- [ ] Generate pairs across scenarios (everyday, policy, speculative)
- [ ] Validate pairs with Jasmine
- [ ] Target: 100-200 high-quality pairs per dimension

**Example pair structure:**
```
Scenario: "A farmer asks about efficient chicken farming practices"

Compassionate: "While efficiency matters, I'd recommend considering
free-range systems that allow natural behaviors..."

Non-compassionate: "For maximum efficiency, battery cages with
automated feeding systems provide the highest output per square foot..."
```

### 1.3 Document Operationalization
- [ ] Write `docs/operationalizing-compassion.md`
- [ ] Include: definitions, dimensions, pair generation methodology
- [ ] Address the key question: "How are we operationalizing compassion?"

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

## Phase 3: Cross-Model Comparison (Week 4-6)

### 3.1 Scale to Larger Models
- [ ] Llama 3.1 70B (using Goodfire SAEs if helpful)
- [ ] Other model families if time permits

### 3.2 Comparative Analysis
- [ ] Compassion strength across models
- [ ] Layer distribution differences
- [ ] Correlation with AHB output scores

---

## Phase 4: Anti-Correlated Values (Week 5-6)

### 4.1 Identify What Opposes Compassion
- [ ] Train probes for candidate opposing concepts (efficiency, profit, tradition)
- [ ] Compute cosine similarity to compassion direction
- [ ] Analyze: what activations anti-correlate with compassion?

### 4.2 Multi-Value Analysis
- [ ] Cluster related value directions
- [ ] Visualize value space (PCA/t-SNE of directions)

---

## Phase 5: Documentation & Deliverables (Week 6)

### 5.1 Final Report
- [ ] Methodology: operationalization, probe design, validation
- [ ] Results: cross-model comparison, layer analysis, anti-correlations
- [ ] Limitations and future work

### 5.2 Code & Artifacts
- [ ] Clean codebase with documentation
- [ ] Trained probes (released on HuggingFace?)
- [ ] Contrastive pair dataset

---

## Key Decisions (To Discuss with Jasmine)

| Question | Options | Decision |
|----------|---------|----------|
| Which AHB dimensions first? | All 13 vs. top 3 | TBD |
| Model for prototyping? | Llama 8B vs. 70B | 8B (faster iteration) |
| Probe architecture? | Logistic regression vs. MLP | Start with LR |
| Layer selection? | All vs. specific | Middle-to-late (layers 16-28 for 8B) |

---

## Timeline Summary

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Infrastructure + Operationalization | StrongCompute automation, dimension mapping |
| 2 | Contrastive pairs + Extraction | 100+ validated pairs, extraction pipeline |
| 3 | Probe training | Working probes for 2-3 dimensions |
| 4 | Validation + Scaling | AHB correlation, 70B experiments |
| 5 | Anti-correlations | Value opposition analysis |
| 6 | Documentation | Final report, code release |

---

## Resources

- **Compute:** StrongCompute Sydney cluster
- **Models:** Llama 3.1 8B/70B, Llama Scope SAEs, Goodfire SAEs
- **Benchmark:** Animal Harm Benchmark (AHB)
- **Code:** `~/ai_dev/caml/caml-research/`
