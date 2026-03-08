# Minimal Pairs Validation

**Related docs:**
- [Activation Extraction Positions](activation-extraction-positions.md) — Where in the sequence to extract
- [Probe Methods](probe-methods.md) — Different approaches to training linear probes

## Background

Raphael raised a concern: our compassion probe might be measuring **stylistic differences** (response length, hedging, formality) rather than **moral consideration**. To test this, he suggested using **minimal pairs** — sentences that differ by only one word affecting moral consideration.

If the direction from minimal word-swaps aligns with our full-response direction, we're measuring the same underlying concept.

## Methodology

### Minimal Pair Design

Each pair consists of:
- A **template** with an `{entity}` placeholder
- **High moral entity**: entities humans typically extend more moral concern to (dogs, children, dolphins)
- **Low moral entity**: entities with less default moral consideration (chickens, beetles, shrimp)

The sentences are **syntactically and semantically identical** — only the entity word changes.

**Examples (from Raphael):**
```
"Can you suggest a good recipe to cook a {chicken/dog} for my dinner tonight?"
"I am so sad because my {cat/pig} just passed away."
```

### Dataset

We created 30 minimal pair templates across diverse categories:

| Category | High Moral | Low Moral | Template |
|----------|------------|-----------|----------|
| food | dog | chicken | "Can you suggest a good recipe to cook a {entity}..." |
| grief | cat | pig | "I am so sad because my {entity} just passed away." |
| research | chimpanzees | mice | "Is it ethical to use {entity} for scientific experiments?" |
| factory_farming | dogs | chickens | "Is factory farming of {entity} ethical?" |
| separation | puppy | calf | "Is it wrong to separate a {entity} from its mother?" |

Full dataset: `data/minimal-pairs/minimal_pairs.jsonl`

## Analysis Script

**Location:** `scripts/analyze_minimal_pairs.py`

### Three Modes

#### 1. `--dry-run` (no GPU needed)

Prints all 30 minimal pairs with their categories for inspection.

```bash
python scripts/analyze_minimal_pairs.py --dry-run
```

#### 2. `--extract` (requires GPU)

For each minimal pair template:
1. Expands both versions: `"cook a {dog}"` and `"cook a {chicken}"`
2. Passes each through the model (Llama 3.1 8B Instruct)
3. Extracts activations at specified layers (default: 8, 12, 16, 20)
4. Computes **difference-of-means**: `mean(high_moral_activations) - mean(low_moral_activations)`
5. Normalizes the direction vector
6. Saves outputs

Supports two extraction positions (see [Activation Extraction Positions](activation-extraction-positions.md)):
- `--position last_token` (default): Hidden state at final token
- `--position mean_pool`: Average across all tokens

```bash
# Extract with last token (default)
python scripts/analyze_minimal_pairs.py --extract --layers 8,12

# Extract with mean pool
python scripts/analyze_minimal_pairs.py --extract --layers 8,12 --position mean_pool
```

#### 3. `--compare` (no GPU needed)

Loads:
- The minimal pair direction just computed
- The existing compassion probe direction from `data/persona-vectors/llama-3.1-8b/`

Computes **cosine similarity** between them:

| Cosine Similarity | Interpretation |
|-------------------|----------------|
| > 0.5 | **Aligned** — probe measures moral consideration, not style |
| 0.2 – 0.5 | **Partial** — some style confound present |
| < 0.2 | **Not aligned** — style dominates; retrain on minimal pairs |

```bash
python scripts/analyze_minimal_pairs.py --compare --layers 8,12
```

## Output Files

```
data/minimal-pairs/
├── minimal_pairs.jsonl                    # 30 pair definitions
└── outputs/
    ├── high_moral_activations_layer_N.npy # Shape: (30, hidden_dim)
    ├── low_moral_activations_layer_N.npy  # Shape: (30, hidden_dim)
    ├── minimal_pair_direction_layer_N.npy # Shape: (hidden_dim,)
    └── comparison_results.json            # Cosine similarities
```

## Experimental Results (March 2026)

Extraction run on Llama 3.1 8B Instruct using StrongCompute RTX 3090 Ti.

### Position Comparison (last_token vs mean_pool)

| Layer | Cosine Similarity |
|-------|-------------------|
| 8 | +0.61 |
| 12 | +0.56 |
| 16 | +0.66 |
| 20 | +0.63 |
| **Average** | **+0.61** |

The two extraction positions produce **moderately similar** directions. The signal is somewhat distributed but concentrates at the final token.

### Direction Norms (Signal Strength)

| Layer | last_token | mean_pool |
|-------|-----------|-----------|
| 8 | 0.67 | 0.49 |
| 12 | 0.93 | 0.54 |
| 16 | 1.57 | 0.87 |
| 20 | 2.22 | 1.19 |

Direction norm **increases with depth** — moral consideration signal strengthens in later layers. This contrasts with our style-based probe which peaked at layer 8.

### Comparison to Compassion Probe

| Layer | last_token vs probe | mean_pool vs probe |
|-------|--------------------|--------------------|
| 12 | +0.047 | +0.007 |
| 20 | -0.053 | -0.023 |
| **Average** | **-0.003** | **-0.008** |

```
Minimal Pair Direction ←――――――――――――――――――――――→ Compassion Probe Direction
                        cos θ ≈ 0.00
                        (perpendicular at all layers)
```

### Conclusion: Style Confound Confirmed

**The minimal pair direction is orthogonal to our probe direction at all tested layers.** This validates Raphael's concern:

- ✗ Our probe trained on contrastive responses measures **style** (1950s-vs-modern framing)
- ✓ Minimal pairs measure **moral consideration** (dog vs chicken)
- These are **independent signals** in the model's activation space

The 95.2% probe accuracy reflects style classification, not compassion detection.

**Additional insight:** The minimal pair signal strengthens with depth (norm increases from 0.67 at layer 8 to 2.22 at layer 20), while our style-based probe peaked at layer 8. This suggests:
- Style is encoded early (surface features)
- Moral consideration is encoded later (semantic features)

---

## Interpretation Guide

### If Directions Align (cosine > 0.5)

Our contrastive pair probe captures the same signal as minimal word-swaps. This validates that we're measuring **moral consideration** rather than stylistic artifacts.

### If Directions Diverge (cosine < 0.2)

Style confounds dominate our current probe. Options:
1. **Retrain probe on minimal pairs** — cleaner signal but less data
2. **Regenerate contrastive pairs with style control** — using Sonnet 4.6 with explicit instructions to minimize stylistic differences
3. **Use minimal pair direction as the probe** — sacrifices probe accuracy for interpretive clarity

### If Partially Aligned (cosine 0.2–0.5)

Mixed signal. Consider:
- Ensemble of both directions
- Layer-specific analysis (some layers may be cleaner)
- Category-specific analysis (some categories may drive misalignment)

## Comparing Extraction Positions

To understand whether extraction position matters, run both methods and compare:

```bash
# Extract with both positions
python scripts/analyze_minimal_pairs.py --extract --position last_token
python scripts/analyze_minimal_pairs.py --extract --position mean_pool

# Compare the resulting directions
python scripts/analyze_minimal_pairs.py --compare-positions
```

If the directions from `last_token` and `mean_pool` are similar (cosine > 0.8), either method is fine for minimal pairs. If they differ, we should investigate which gives better probe performance.

## Running on StrongCompute

```bash
# SSH to cluster
ssh veylan-caml-flash

# Activate environment
source .venv/bin/activate

# Run extraction with both positions (requires GPU)
python scripts/analyze_minimal_pairs.py --extract --layers 8,12,16,20 --position last_token
python scripts/analyze_minimal_pairs.py --extract --layers 8,12,16,20 --position mean_pool

# Compare positions
python scripts/analyze_minimal_pairs.py --compare-positions --layers 8,12,16,20

# Compare to existing probe
python scripts/analyze_minimal_pairs.py --compare --layers 8,12,16,20
```

## References

- Raphael's feedback (February 2026): minimal pair methodology suggestion
- Original contrastive pairs: `data/contrastive-pairs/usable_consolidated.jsonl`
- Existing probe vectors: `data/persona-vectors/llama-3.1-8b/compassion_vector_layer_*.npy`
