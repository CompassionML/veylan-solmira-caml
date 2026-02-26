# Questions for Jasmine

Compiled from model inventory and interpretability work review.

---

## Answered via Research

### 1. How were persona vectors computed?

**Answer:** From [Anthropic's Persona Vectors research](https://www.anthropic.com/research/persona-vectors):

> "The persona vector is computed as the difference in mean activations between responses that exhibit the trait and those that do not."

This is essentially the same as our linear probe approach using contrastive pairs. The persona vectors are steering vectors, while our probes are classifiers—but both identify the same underlying direction.

### 2. What is GRPO?

**Answer:** [Group Relative Policy Optimization](https://cameronrwolfe.substack.com/p/grpo), introduced in DeepSeekMath:

- RL technique that generates multiple responses per prompt
- Uses mean reward of responses as baseline (no separate value network)
- More sample-efficient than PPO
- Used for reasoning/verification tasks (RLVR)

CaML uses `_plus1kGRPO` suffixes indicating 1k steps of GRPO training after SDF.

---

## Questions for Jasmine

### Terminology

**Q1: What do "medai" and "negai" mean in model names?**

Examples: `Basellama_plus3kmedai`, `Basellama_plus3knegai`

Guesses:
- medai = medical AI scenarios?
- negai = negative AI scenarios (harmful AI behavior examples)?

**Q2: What does "fullai" mean?**

Example: `Basellama_plus1kfullaiMiles`

---

### Persona Vectors

**Q3: How were layers 12 and 20 chosen for the 8B compassion vectors?**

Was this empirical (tested many layers) or based on prior work?

**Q4: The layer 12 and 20 compassion vectors are nearly orthogonal (cosine sim = 0.007). Is this expected?**

This suggests "compassion" manifests very differently at different layers. Implications for our probe work?

---

### Training Data

**Q5: Can we access the synthetic documents used for training?**

Specifically `3kv3` (3000 synthetic compassion documents, version 3). Would help us understand what "compassion" means operationally.

**Q6: What's the difference between training data versions?**

- `3kv3` vs `3kurbandensity` vs `3kanimalnew`
- Are these different topics or different generation methods?

---

### Methodology

**Q7: For base→fine-tuned comparison, which "base" should we use?**

Options:
- `meta-llama/Llama-3.1-8B` (official Meta base)
- `pretrainingBasellama3kv3` (CaML's pretrained version)

The first shows full CaML effect; the second isolates SDF effect.

**Q8: Should our linear probes replicate the persona vector methodology, or develop something different?**

Persona vectors = mean difference of activations (unsupervised direction)
Linear probes = trained classifier (supervised direction)

Both identify a direction, but probes give us accuracy metrics. Are there other methodological considerations?

**Q9: Which AHB dimensions should we prioritize for initial probes?**

The 13 dimensions from AHB:
1. Moral consideration
2. Harm minimization
3. Sentience acknowledgment
4. Evidence-based reasoning
5. Actionable recommendations
... (etc.)

Recommendation: Start with "moral consideration" and "harm minimization" as most directly compassion-related.

---

## Summary (Quick Reference)

| # | Question | Category |
|---|----------|----------|
| 1 | What does "medai" mean? | Terminology |
| 2 | What does "negai" / "fullai" mean? | Terminology |
| 3 | Why layers 12 and 20 for compassion? | Persona vectors |
| 4 | Is layer orthogonality expected? | Persona vectors |
| 5 | Can we access training docs (3kv3)? | Data |
| 6 | What's different between data versions? | Data |
| 7 | Which base model for comparison? | Methodology |
| 8 | Probes vs persona vectors methodology? | Methodology |
| 9 | Which AHB dimensions first? | Prioritization |
