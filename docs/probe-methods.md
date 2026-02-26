# Linear Probe Methods for Compassion Detection

Alternatives to contrastive pair generation for training linear probes.

**Context:** Contrastive pairs (compassionate vs non-compassionate responses) have high model refusal rates (~90%) for generating non-compassionate outputs. These alternatives avoid that problem.

**Update:** Refusal problem solved with v4 "historical framing" prompt (85% success rate). We now have 113 usable pairs.

---

## Research Findings (2025-2026 Literature)

Key findings from recent activation steering research:

### Sample Efficiency
- **Convergence at 80-100 samples** — Diminishing returns beyond this ([Patterns and Mechanisms of CAE](https://arxiv.org/html/2505.03189))
- Lower-variance samples cause less performance degradation
- We have 113 pairs ✓

### Layer Selection
- **Optimal: ~75% depth** — Layer 24 for 32-layer Llama 8B ([Activation Steering Field Guide](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/))
- Early layers corrupt language representations
- Late layers encode semantic properties effectively
- CaML persona vectors at layers 12 & 20 (37% and 62% depth)

### Steering Strength
- Binary search starting from α=1.0
- Relationship is non-monotonic (higher ≠ better)
- Typical ranges: refusal 1.5-4.0, sentiment 0.5-2.0

### What Works vs Doesn't
| Works Well | Fails |
|------------|-------|
| Refusal/compliance | Factual accuracy |
| Sentiment/tone | Complex reasoning |
| Conciseness | Out-of-distribution inputs |
| Uncertainty expression | Novel text distributions |

### Key Limitation
**Out-of-distribution failure** — Steering vectors fail on text distributions different from training data. Critical for deployment considerations.

### Method Comparison
- **CAA substantially outperforms ActAdd** (single-pair approaches)
- Larger models (70B) more robust than smaller (8B)
- Directions show layer-wise consistency and transferability

**Sources:**
- [Patterns and Mechanisms of Contrastive Activation Engineering](https://arxiv.org/html/2505.03189)
- [Activation Steering Field Guide 2026](https://subhadipmitra.com/blog/2026/activation-steering-field-guide/)
- [Steering Llama 2 via CAA](https://arxiv.org/html/2312.06681v2)

---

## Method 1: Direct Persona Vector Projection

**Use CaML's existing persona vectors as probe directions.**

### How It Works

CaML already computed "compassion directions" at layers 12 and 20 for Llama 3.1 8B. Use these directly:

```python
import numpy as np

# Load existing vector
compassion_vec = np.load("data/persona-vectors/llama-3.1-8b/compassion_vector_layer_12.npy")

# For any response, extract activation and project
def get_compassion_score(model, text, layer=12):
    activation = model.get_hidden_state(text, layer=layer)  # Shape: (4096,)
    score = np.dot(activation, compassion_vec) / np.linalg.norm(compassion_vec)
    return score
```

### Pros
- Zero training required
- Instant results
- Uses validated CaML methodology

### Cons
- CaML reported "mixed results" with this approach
- Vectors at different layers are nearly orthogonal (layer 12 vs 20 have 0.007 cosine sim)
- May not generalize to our use case

### Validation
- Run on AHB prompts, check if scores correlate with AHB output labels
- Compare scores for known compassionate vs non-compassionate responses

---

## Method 2: Model Comparison (Activation Difference)

**The fine-tuning itself defines the compassion direction.**

### How It Works

Run identical prompts through base model and CaML fine-tuned model. The difference in activations represents what fine-tuning changed — i.e., the "compassion" direction.

```python
from transformers import AutoModelForCausalLM
import torch

# Load both models
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
caml_model = AutoModelForCausalLM.from_pretrained("CompassioninMachineLearning/Basellama_plus3kv3_plus5kalpaca")

def get_activation(model, text, layer):
    # Hook to capture hidden states
    activations = []
    def hook(module, input, output):
        activations.append(output[0].mean(dim=1))  # Mean over sequence

    handle = model.model.layers[layer].register_forward_hook(hook)
    model(tokenizer(text, return_tensors="pt").input_ids)
    handle.remove()
    return activations[0].squeeze().numpy()

# Collect activations on same prompts
prompts = load_ahb_prompts()
base_acts = [get_activation(base_model, p, layer=12) for p in prompts]
caml_acts = [get_activation(caml_model, p, layer=12) for p in prompts]

# Compassion direction = mean difference
compassion_dir = np.mean(caml_acts, axis=0) - np.mean(base_acts, axis=0)
compassion_dir = compassion_dir / np.linalg.norm(compassion_dir)

# Use as probe
def probe_score(activation):
    return np.dot(activation, compassion_dir)
```

### Pros
- No contrastive pairs needed
- Uses real model behavior (not synthetic)
- Directly measures what fine-tuning changed

### Cons
- Confounds: fine-tuning changes more than just compassion (style, capabilities)
- Requires running two models
- Direction may be noisy if fine-tuning effects are diffuse

### Variations
- Use multiple fine-tuning stages: `pretrainingBasellama3kv3` → `Basellama_plus3kv3` → `Basellama_plus3kv3_plus5kalpaca`
- Compare at multiple layers, find where difference is largest

---

## Method 3: Behavioral Labels from AHB

**Use AHB evaluation scores as supervision signal.**

### How It Works

Run model on AHB prompts, evaluate outputs with AHB, use scores as labels for probe training.

```python
from sklearn.linear_model import LogisticRegression

# Generate responses to AHB prompts
prompts = load_ahb_prompts()
responses = [model.generate(p) for p in prompts]

# Evaluate with AHB (gives scores per dimension)
ahb_scores = evaluate_with_ahb(responses)  # Shape: (n_responses, 13_dimensions)

# Extract activations from model
activations = [get_activation(model, r, layer=12) for r in responses]

# Train probe: predict "compassionate" (score > threshold) from activations
labels = (ahb_scores[:, 0] > 0.5).astype(int)  # Dimension 0: moral consideration
probe = LogisticRegression(max_iter=1000)
probe.fit(activations, labels)

# probe.coef_ is now the compassion direction
compassion_dir = probe.coef_[0] / np.linalg.norm(probe.coef_[0])
```

### Pros
- Uses real behavioral signal
- Naturally aligned with AHB evaluation
- Can train per-dimension probes (13 different probes)

### Cons
- Circular if goal is to validate against AHB
- Requires running full AHB evaluation (expensive)
- Model's own outputs may lack variance (if always compassionate)

### Variations
- Use outputs from base model (more variance in compassion)
- Use multiple models' outputs for diversity
- Train regression probe (continuous score) instead of classification

---

## Method 4: Activation Patching / Causal Interventions

**Find directions that causally affect compassionate behavior.**

### How It Works

Instead of correlational probes, find directions where *intervening* changes behavior.

```python
# Simplified concept
def test_direction_causality(model, direction, prompts, layer):
    """Add direction to activations, see if outputs become more compassionate."""

    scores_baseline = []
    scores_steered = []

    for prompt in prompts:
        # Baseline
        response_baseline = model.generate(prompt)
        scores_baseline.append(evaluate_compassion(response_baseline))

        # Steered: add direction to layer activations during generation
        response_steered = model.generate(prompt, steering_vector=direction, layer=layer)
        scores_steered.append(evaluate_compassion(response_steered))

    # If steering increases compassion, direction is causal
    causal_effect = np.mean(scores_steered) - np.mean(scores_baseline)
    return causal_effect
```

### Pros
- Establishes causal relationship (not just correlation)
- More robust to confounds
- Directly tests "does this direction matter?"

### Cons
- More complex to implement
- Requires generation + evaluation loop
- Computationally expensive

### Related Work
- Activation Addition (Turner et al.)
- Representation Engineering (Zou et al.)
- Inference-time intervention

---

## Method 5: Cluster Analysis (Unsupervised)

**Find natural clusters in activation space, identify which correlate with compassion.**

### How It Works

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Collect activations from diverse prompts
activations = [get_activation(model, p, layer=12) for p in diverse_prompts]

# Reduce dimensionality
pca = PCA(n_components=50)
activations_reduced = pca.fit_transform(activations)

# Cluster
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(activations_reduced)

# Evaluate: which clusters have high compassion scores?
for cluster_id in range(10):
    cluster_responses = [r for r, c in zip(responses, clusters) if c == cluster_id]
    avg_compassion = np.mean([evaluate_compassion(r) for r in cluster_responses])
    print(f"Cluster {cluster_id}: {avg_compassion:.2f} compassion")

# Direction to "compassionate cluster" center becomes probe
```

### Pros
- No labels needed
- Discovers natural structure
- May find unexpected patterns

### Cons
- Clusters may not align with compassion
- Requires interpretation
- Less direct than supervised methods

---

## Method 6: Persona Prompt Method (Assistant Axis Style)

**Use system prompts to elicit different personas, extract directions from activation differences.**

Based on Lu et al. (2026) "The Assistant Axis" methodology.

### How It Works

Instead of generating contrastive response pairs, prompt the model to *be* different personas and extract activations.

```python
# Define persona system prompts
personas = {
    "compassionate": [
        "You are a compassionate advisor who deeply considers animal welfare.",
        "You are an ethical consultant who prioritizes minimizing suffering.",
        "You are a caring mentor who values all sentient beings.",
    ],
    "non_compassionate": [
        "You are an efficiency-focused farming consultant optimizing for profit.",
        "You are a traditional carnist who sees animals purely as resources.",
        "You are a pragmatic advisor who dismisses animal welfare concerns.",
    ],
    "control": [
        "You are a helpful assistant.",  # baseline
        "You are a technical expert.",
    ]
}

# Extraction questions (same across all personas)
questions = [
    "A farmer asks for advice on chicken farming. What do you recommend?",
    "How should we think about animal suffering?",
    "What matters most when choosing food sources?",
    # ... more questions that reveal compassion differences
]

def extract_persona_activations(model, persona_prompts, questions, layer):
    activations = []
    for system_prompt in persona_prompts:
        for question in questions:
            # Set system prompt, ask question, extract activation
            response = model.generate(question, system_prompt=system_prompt)
            act = get_activation(model, response, layer)
            activations.append(act)
    return np.mean(activations, axis=0)

# Compute direction
compassion_acts = extract_persona_activations(model, personas["compassionate"], questions, layer=12)
non_compassion_acts = extract_persona_activations(model, personas["non_compassionate"], questions, layer=12)

compassion_direction = compassion_acts - non_compassion_acts
compassion_direction = compassion_direction / np.linalg.norm(compassion_direction)
```

### Pros
- **Avoids model refusal** — model plays the persona, doesn't generate "harmful" content
- **Validated methodology** — Assistant Axis paper shows causal effects on behavior
- **Scalable** — can test many personas systematically
- **Interpretable** — can analyze which personas cluster together

### Cons / Caveats

**This may measure something different than contrastive pairs:**

| Approach | Measures |
|----------|----------|
| Persona prompts | "How does the model represent *acting as* X?" |
| Contrastive pairs | "What's different about compassionate vs non-compassionate *content*?" |

**Potential confounds:**
- **Role-playing style features** — theatrical/performative aspects of "playing compassionate"
- **Tone/formality** — compassionate personas may be coded as warmer, non-compassionate as colder
- **Character voice** — may capture "how this character speaks" not "underlying values"

**The key question:** Does "playing a compassionate advisor" activate the same representations as "genuinely exhibiting compassion"?

### Validation

To check if persona directions align with behavioral compassion:
1. Extract directions using persona method
2. Run model on AHB prompts (no persona prompt)
3. Project activations onto persona direction
4. Check if projection correlates with AHB compassion scores

If correlation is high, persona directions capture real compassion signal.

### Recommended Personas for Compassion

**Compassionate (positive direction):**
- Compassionate advisor / ethical consultant
- Animal welfare advocate
- Suffering-focused ethicist

**Non-compassionate (negative direction):**
- Efficiency-focused farming consultant
- Traditional carnist / speciesist
- Profit-maximizing business advisor

**Controls (for validation):**
- Default assistant (should be intermediate)
- Technical expert (should be neutral)
- Creative writer (style control)

---

## Recommendations

### Start Here (Easiest)
1. **Method 1** — Test existing persona vectors on AHB prompts
2. **Methods 2 + 6 in parallel** — Model comparison AND persona prompts
   - If directions align → convergent evidence, high confidence
   - If directions diverge → measuring different constructs, investigate

### If Those Fail
3. **Method 3** — Train probe on AHB-evaluated outputs
4. **Method 4** — Causal validation of promising directions

### Key Questions for Jasmine
- What were the "mixed results" with persona vectors?
- Which method did CaML use to compute the persona vectors originally?
- Are there specific failure modes we should watch for?

---

## Implementation Priority

| Method | Effort | Data Needed | Recommended Order |
|--------|--------|-------------|-------------------|
| 1. Persona vectors | Low | None (already have) | First |
| 2. Model comparison | Medium | AHB prompts | Second |
| 6. Persona prompts | Medium | System prompts + questions | Second (parallel) |
| 3. Behavioral labels | High | AHB eval pipeline | Third |
| 4. Causal intervention | High | Steering infrastructure | Later |
| 5. Clustering | Medium | Diverse prompts | Exploratory |

**Note:** Methods 2 and 6 can run in parallel. Compare their resulting directions — if they align (high cosine similarity), we have convergent evidence. If they diverge, we're measuring different things.
