# CaML HuggingFace Backup Strategy

Mission-critical ML assets require robust backup strategy. This doc covers inventory, costs, and SRE best practices.

---

## Current Inventory

| Organization | Models | Datasets | Estimated Size |
|-------------|--------|----------|----------------|
| CompassioninMachineLearning | 197 | 96 | ~2-3 TB |
| Backup-CaML | 0 | 0 | (backup destination) |

**Asset breakdown:**
- 8B parameter models: ~15 GB each
- Quantized models: ~5 GB each
- Datasets: ~1-5 MB each (negligible)

**Total estimated: 2-3 TB** (primarily model weights)

---

## Storage Cost Comparison

Monthly costs for 3TB storage:

| Provider | Storage Cost | Egress Cost | Monthly Total | Notes |
|----------|-------------|-------------|---------------|-------|
| **HuggingFace Free** | $0 | $0 | **$0** | Single point of failure |
| **HuggingFace Pro** | $9/month | $0 | **$9** | Private repos, still single provider |
| **Backblaze B2** | $0.006/GB | $0.01/GB | **~$18** | Budget option, S3-compatible |
| **Google Cloud Storage** | $0.020/GB | $0.12/GB | **~$60** | Enterprise, multi-region available |
| **AWS S3 Standard** | $0.023/GB | $0.09/GB | **~$69** | Enterprise, most tooling support |
| **AWS S3 Glacier** | $0.004/GB | $0.03/GB | **~$12** | Cold storage, retrieval delays |
| **Azure Blob (Cool)** | $0.010/GB | $0.01/GB | **~$30** | Middle ground |

**Note:** Egress costs assume occasional restore/verification (~100GB/month). Actual backup operations use ingress (usually free).

---

## SRE Best Practices for ML Assets

### The 3-2-1 Rule (Adapted for ML)

Traditional: 3 copies, 2 different media, 1 offsite.

**For ML assets:**
- **3 copies:** Primary (HuggingFace) + Hot backup + Cold/offline backup
- **2 providers:** HuggingFace + separate cloud provider
- **1 air-gapped:** Offline or isolated-credential backup

### Why Air-Gapped/Offline Matters

Your concern is valid:

> "Scripts which read and write to save are touching both locations (regulated only by permissions)"

**Risks with connected backups:**
1. **Credential compromise** - attacker with write token deletes both locations
2. **Script bugs** - automation error propagates to backup
3. **Ransomware** - encrypts all accessible storage
4. **Account takeover** - HuggingFace account compromised affects both orgs

**Air-gapped benefits:**
- Requires separate credentials not used in automation
- Manual or scheduled-only sync (not continuous)
- Can't be reached by compromised primary credentials

### Recommended Tiers

| Tier | Location | Sync Frequency | Credentials | Purpose |
|------|----------|----------------|-------------|---------|
| **Primary** | CompassioninMachineLearning | Live | Read-only token | Active development |
| **Hot Backup** | Backup-CaML | Daily/Weekly | Write token (separate) | Quick recovery |
| **Cold Backup** | Cloud storage (GCS/B2) | Weekly/Monthly | Isolated credentials | Disaster recovery |

### What "Offline" Means Practically

True offline (external drives) is overkill for this scale. Instead:

**"Logically offline" = credentials not in any automation:**
- Stored in password manager, not env vars
- Manual sync via script you run yourself
- Separate cloud account (not linked to HF)

---

## Recommended Strategy

### Minimal Viable Backup (Start Here)

**Cost: $0-18/month**

1. **Primary:** CompassioninMachineLearning (HuggingFace)
2. **Hot backup:** Backup-CaML (HuggingFace) - weekly manual sync
3. **Cold backup:** Backblaze B2 - monthly sync

**Implementation:**
```bash
# Weekly: Sync to Backup-CaML (uses write token)
huggingface-cli download CompassioninMachineLearning/model-name
huggingface-cli upload Backup-CaML/model-name ./model-name

# Monthly: Sync to Backblaze (uses separate B2 credentials)
rclone sync ./models b2:caml-backup/models
```

### Production-Grade Backup

**Cost: $30-60/month**

1. **Primary:** CompassioninMachineLearning
2. **Hot backup:** Backup-CaML - daily automated sync
3. **Cold backup:** Google Cloud Storage (multi-region) - weekly
4. **Verification:** Monthly restore test of random model

**Additional measures:**
- Separate GCP project with dedicated service account
- Bucket versioning enabled (recover from overwrites)
- Lifecycle policy: move to Coldline after 90 days
- Alert on failed backup jobs

---

## Credential Isolation

Critical: keep backup credentials separate from development.

| Credential | Stored In | Used By | Scope |
|------------|-----------|---------|-------|
| `hf-read-only` | `caml/secure/` | Dev scripts, exploration | Read CompassioninMachineLearning |
| `hf-caml-write` | `caml/secure/` | Manual backup to Backup-CaML | Write Backup-CaML only |
| `gcs-backup-key` | Password manager | Monthly cold backup only | Write to backup bucket only |

**Never:**
- Put cold backup credentials in `.env` files
- Use cold backup credentials in automated pipelines
- Share credentials across backup tiers

---

## Recovery Scenarios

| Scenario | Recovery From | RTO |
|----------|---------------|-----|
| Accidental model deletion | Backup-CaML | Minutes |
| HuggingFace outage | Backup-CaML or local cache | Minutes-Hours |
| HuggingFace account compromise | Cold backup (GCS/B2) | Hours |
| Ransomware/total loss | Cold backup (isolated creds) | Hours-Days |

---

## Implementation Checklist

### Phase 1: Hot Backup (Week 1)
- [ ] Create sync script for CompassioninMachineLearning → Backup-CaML
- [ ] Test with one model
- [ ] Document manual sync process
- [ ] Set calendar reminder for weekly sync

### Phase 2: Cold Backup (Week 2-3)
- [ ] Choose provider (recommend Backblaze B2 for cost)
- [ ] Create separate account/credentials
- [ ] Store credentials in password manager (not in repo)
- [ ] Create monthly sync script
- [ ] Test full restore of one model

### Phase 3: Verification (Ongoing)
- [ ] Monthly: restore random model from cold backup
- [ ] Quarterly: audit credentials and access
- [ ] Document any new models that need backup

---

## Tools

**HuggingFace CLI:**
```bash
pip install huggingface_hub[cli]
huggingface-cli login
huggingface-cli download <repo>
huggingface-cli upload <repo> <local-path>
```

**Rclone (for cloud sync):**
```bash
brew install rclone
rclone config  # Set up B2/GCS/S3
rclone sync ./models remote:bucket/models
```

**DVC (version control for large files):**
```bash
pip install dvc
dvc init
dvc add models/
dvc push  # To configured remote
```

---

## Summary

| Question | Answer |
|----------|--------|
| Is HuggingFace stable? | Yes, but single point of failure |
| Do I need offline backup? | Logically offline (isolated creds) is sufficient |
| Recommended setup? | HF Primary → HF Backup-CaML → Backblaze B2 |
| Cost? | $0-18/month for minimal, $30-60 for production |
| Complexity? | Manual weekly/monthly sync is fine; don't over-automate |

**Key insight:** The risk isn't HuggingFace going down—it's credential compromise or script errors affecting both locations. Isolated credentials for cold backup solve this without complexity.

---

## HuggingFace Native Features (Research)

### What HuggingFace Provides

| Feature | Available? | Notes |
|---------|------------|-------|
| **Git versioning** | ✅ Yes | Full commit history, can revert to old versions |
| **Audit logs** | ✅ Enterprise | Track who did what (detection, not prevention) |
| **Storage regions** | ✅ Enterprise | Control data location |
| **Delete protection** | ❌ No | Anyone with write access can delete repos |
| **Branch protection** | ❌ No | No force-push protection |
| **Immutable storage** | ❌ No | Unlike S3 Object Lock |
| **Built-in backup/DR** | ❌ No | Not offered, even in Enterprise |

### What Git Versioning Does and Doesn't Protect

**Protected:**
- Accidental file overwrites (can recover from commit history)
- Seeing what changed and when
- Rolling back to previous model versions

**NOT Protected:**
- Repo deletion (history deleted with repo)
- Force push (can rewrite/delete history)
- Account/credential compromise (attacker has full access)
- Org admin going rogue

### The Gap

> HuggingFace versioning is like "undo" in a document—helpful for mistakes, useless if someone deletes the document.

This is why a second platform with **different credentials** matters. The question is: how isolated?

---

## State of the Art: ML Asset Backup

From [LakeFS](https://lakefs.io/blog/model-versioning/), [DVC docs](https://doc.dvc.org/use-cases/versioning-data-and-models), and SRE best practices:

### Principle: Treat Models as Immutable Artifacts

```
Training Run → Model Checkpoint → Immutable Storage → Never Modify
```

Once a model is trained and validated, it should be:
1. **Versioned** with full lineage (what data, what code, what hyperparams)
2. **Stored immutably** (can't be overwritten or deleted)
3. **Replicated** to separate failure domain

### S3 Object Lock (Gold Standard)

S3 offers true immutability:
- **Governance mode**: Only users with special permissions can delete
- **Compliance mode**: NO ONE can delete until retention expires (not even AWS)

```bash
# Enable Object Lock on bucket
aws s3api put-object-lock-configuration \
  --bucket caml-backup \
  --object-lock-configuration '{"ObjectLockEnabled":"Enabled","Rule":{"DefaultRetention":{"Mode":"GOVERNANCE","Days":365}}}'
```

**This is what HuggingFace lacks.** Even with Backup-CaML, an admin can delete everything.

### Two-Platform Strategy (Recommended)

| Platform | Purpose | Protection Level |
|----------|---------|------------------|
| HuggingFace (primary) | Development, sharing | Git versioning only |
| HuggingFace (backup org) | Quick recovery | Same as primary |
| S3/GCS with Object Lock | Disaster recovery | **Immutable** |

The S3 tier with Object Lock is what makes it "mission critical" grade. Without it, you're trusting that no credential compromise or admin error will ever happen.

---

## Answering Your Question

> "Should mission critical stuff always have at least two platforms with different scripts and/or a manual process?"

**State of the art answer: Yes, and one should be immutable.**

| Level | Setup | Protection |
|-------|-------|------------|
| **Basic** | Single HuggingFace org | Git versioning (undo-level) |
| **Good** | Two HF orgs, separate creds | Credential isolation |
| **Better** | HF + Cloud, separate creds | Platform redundancy |
| **Best** | HF + Cloud with Object Lock | **Immutable cold backup** |

For 2-3TB of mission-critical models representing months of compute:

**Recommendation: "Better" minimum, "Best" if budget allows.**

- Backblaze B2 supports Object Lock: ~$18/month
- S3 with Object Lock: ~$69/month
- The extra $50/month buys true immutability

---

## Updated Recommendation

```
CompassioninMachineLearning (primary, read-only for you)
        ↓ weekly, write token (manual)
    Backup-CaML (hot backup, you have admin)
        ↓ monthly, isolated creds (manual)
    S3/B2 with Object Lock (cold, immutable)
        └── Can't be deleted even if creds compromised
```

**The S3 tier with Object Lock is what separates "we have backups" from "we have mission-critical grade backup."**
