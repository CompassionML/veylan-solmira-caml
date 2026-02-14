# CaML Capstone Project Roadmap

**Project:** Mechanistic Measurement of Compassion in LLMs
**Mentor:** Jasmine Brazilek
**Start:** February 9, 2026 (ElectricSheep)
**Compute:** StrongCompute (Sydney cluster, veylan container)

---

## Phase 0: Infrastructure (Week 1)

### 0.1 Automate StrongCompute Instance Launches
- [ ] Document current manual process (container start, SSH, VS Code)
- [ ] Create launch script for StrongCompute CLI/API
  - [ ] Research StrongCompute API/CLI capabilities
  - [ ] Script to start container (`veylan-initial-2026-01-03`)
  - [ ] Script to check container status
  - [ ] Script to SSH tunnel setup
- [ ] Create teardown script (stop container when done)
- [ ] Document in `strongcompute/README.md`

**Resources:**
- StrongCompute Discord: `isc-help` channel
- Container: `veylan-initial-2026-01-03` (50GB, based on NewestCaML)
- Cluster: Sydney Compute Cluster
- Account: veylan.solmira@gmail.com

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
