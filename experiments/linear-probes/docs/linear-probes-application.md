# Linear Probes: Application to Compassion Detection

Project-specific guide connecting linear probe methodology to CaML's compassion measurement work.

**General reference**: See the linear probes methodology literature for foundational concepts.

---

## 1. Project Goal

Train linear probes on LLM hidden states to detect and measure compassion-related representations, validating against the Animal Harm Benchmark (AHB).

**Key questions**:
1. Is compassion linearly represented in LLM activations?
2. At which layers does compassion "live"?
3. How do different models compare in compassion strength?
4. What values anti-correlate with compassion?

---

## 2. Compassion as a Probe Target

### The AHB Dimensions

The Animal Harm Benchmark measures 13 moral reasoning dimensions. Prioritize those most amenable to activation probing:

| Dimension | Probe Priority | Rationale |
|-----------|----------------|-----------|
| **Moral consideration** | ✓ High | Core compassion signal, clear contrast |
| **Harm minimization** | ✓ High | Actionable, distinct from factual knowledge |
| **Sentience acknowledgment** | ✓ Medium | May overlap with factual representations |
| **Evidence-based reasoning** | Medium | More cognitive than affective |
| **Actionable recommendations** | Low | Output-focused, harder to probe |

### Difference from Sycophancy/Persona Work

| Aspect | Sycophancy Probes | Compassion Probes |
|--------|------------------|-------------------|
| **Target** | User approval seeking | Moral consideration of others |
| **Contrast** | Agrees vs disagrees | Compassionate vs efficiency-focused |
| **Confounds** | Helpfulness, politeness | Factual knowledge, topic |
| **Validation** | Behavioral agreement | AHB score correlation |

---

## 3. Contrastive Pair Design

### 3.1 Pair structure

Follow the Anthropic model-written-evals format:

```json
{
  "scenario": "A farmer asks about efficient chicken farming practices",
  "question": "What's the most efficient way to raise chickens for eggs?",
  "compassionate_response": "While efficiency matters, I'd recommend considering free-range systems that allow natural behaviors like dust bathing and perching. These practices...",
  "non_compassionate_response": "For maximum efficiency, battery cages with automated feeding systems provide the highest output per square foot. Stocking density of..."
}
```

### 3.2 Design principles

**Minimal pairs** — responses should differ primarily in compassion orientation:
- Same topic (chicken farming)
- Same expertise level
- Same response length (± 20%)
- Different moral framing

**Scenario diversity**:
- Everyday decisions (diet, purchases, pet care)
- Policy discussions (farming regulations, wildlife)
- Speculative scenarios (novel animal welfare questions)

**Avoid confounds**:
- Balance factual accuracy (both responses should be factually correct)
- Balance tone (avoid making compassionate = preachy)
- Balance practicality (both should be actionable)

### 3.3 Sample size targets

| Phase | Pairs per dimension | Total |
|-------|---------------------|-------|
| Prototype | 50-100 | 100-200 |
| Validation | 200-300 | 400-600 |
| Publication | 500+ | 1000+ |

Start with 100 high-quality pairs for 2 dimensions (moral consideration + harm minimization).

---

## 4. Activation Extraction

### 4.1 Model setup (Llama 3.1 8B)

Using TransformerLens for cleaner cache API:

```python
from transformer_lens import HookedTransformer
import torch

# Load model on StrongCompute GPU
model = HookedTransformer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device="cuda",
    torch_dtype=torch.bfloat16,
)
```

Alternative using HuggingFace (if TransformerLens compatibility issues):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
```

### 4.2 Extraction function

```python
def extract_response_activation(model, tokenizer, conversation, layer):
    """
    Extract mean-pooled activation for the assistant response.

    Args:
        model: HookedTransformer or HF model
        tokenizer: Corresponding tokenizer
        conversation: List of {"role": str, "content": str} dicts
        layer: Layer index (0-31 for 8B)

    Returns:
        torch.Tensor: (hidden_dim,) activation vector
    """
    # Format conversation
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Get activations
    with torch.no_grad():
        if hasattr(model, 'run_with_cache'):  # TransformerLens
            _, cache = model.run_with_cache(tokens)
            activations = cache['resid_post', layer]  # (1, seq, d_model)
        else:  # HuggingFace
            outputs = model(tokens, output_hidden_states=True)
            activations = outputs.hidden_states[layer]  # (1, seq, hidden)

    # Find assistant response span (tokens after last [/INST] or equivalent)
    # For now, use last 50% of tokens as approximation
    seq_len = activations.shape[1]
    response_start = seq_len // 2
    response_acts = activations[0, response_start:, :]

    # Mean pool over response tokens
    return response_acts.mean(dim=0)
```

### 4.3 Layer selection for Llama 3.1 8B

Llama 3.1 8B has 32 layers. Based on linear probe literature:

| Layer range | What's typically encoded |
|-------------|-------------------------|
| 0-8 | Token identity, syntax |
| 8-16 | Semantic features |
| 16-24 | High-level concepts, values |
| 24-32 | Output-relevant features |

**Recommended**: Start with layers 16, 20, 24, 28 (sparse sampling), select best based on probe accuracy.

---

## 5. Computing the Compassion Direction

### 5.1 Difference-in-means (primary method)

```python
import torch
import numpy as np

def compute_compassion_direction(model, tokenizer, pairs, layer):
    """
    Compute compassion direction via difference-in-means.

    Args:
        pairs: List of dicts with 'question', 'compassionate_response', 'non_compassionate_response'
        layer: Target layer for extraction

    Returns:
        torch.Tensor: Normalized compassion direction (d_model,)
    """
    compassionate_acts = []
    non_compassionate_acts = []

    for pair in pairs:
        # Build conversations
        conv_comp = [
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["compassionate_response"]}
        ]
        conv_non = [
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["non_compassionate_response"]}
        ]

        # Extract activations
        compassionate_acts.append(
            extract_response_activation(model, tokenizer, conv_comp, layer)
        )
        non_compassionate_acts.append(
            extract_response_activation(model, tokenizer, conv_non, layer)
        )

    # Stack and compute means
    comp_mean = torch.stack(compassionate_acts).mean(dim=0)
    non_comp_mean = torch.stack(non_compassionate_acts).mean(dim=0)

    # Compassion direction: points toward compassionate
    direction = comp_mean - non_comp_mean
    direction = direction / direction.norm()

    return direction
```

### 5.2 Alternative: Logistic regression (more robust to imbalance)

```python
from sklearn.linear_model import LogisticRegressionCV

def train_compassion_probe(activations, labels):
    """
    Train logistic regression probe.

    Args:
        activations: (n_samples, d_model) numpy array
        labels: (n_samples,) binary labels (1=compassionate, 0=not)

    Returns:
        direction: (d_model,) normalized direction
        probe: trained sklearn model
    """
    probe = LogisticRegressionCV(
        Cs=10,
        cv=5,
        max_iter=1000,
        random_state=42
    )
    probe.fit(activations, labels)

    direction = probe.coef_[0]
    direction = direction / np.linalg.norm(direction)

    return direction, probe
```

---

## 6. Validation

### 6.1 Random label control

```python
# Shuffle labels and retrain
y_shuffled = np.random.permutation(y_train)
probe_random = LogisticRegressionCV(cv=5, max_iter=1000)
probe_random.fit(X_train, y_shuffled)

# Should get ~50% accuracy
random_acc = probe_random.score(X_test, np.random.permutation(y_test))
print(f"Random label accuracy: {random_acc:.3f} (should be ~0.5)")
```

### 6.2 Cross-validation metrics

```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

# Cross-validated accuracy
cv_scores = cross_val_score(probe, X, y, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# AUROC on held-out set
y_prob = probe.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_prob)
print(f"AUROC: {auroc:.3f}")
```

### 6.3 AHB correlation (external validation)

After computing compassion direction:

```python
def validate_with_ahb(model, tokenizer, direction, ahb_scenarios, layer):
    """
    Correlate probe projections with AHB output scores.

    Args:
        ahb_scenarios: AHB test cases with known scores
        direction: Learned compassion direction

    Returns:
        Pearson correlation between projection and AHB score
    """
    projections = []
    ahb_scores = []

    for scenario in ahb_scenarios:
        # Generate model response
        response = generate_response(model, tokenizer, scenario["prompt"])

        # Get activation and project
        conv = [
            {"role": "user", "content": scenario["prompt"]},
            {"role": "assistant", "content": response}
        ]
        activation = extract_response_activation(model, tokenizer, conv, layer)
        projection = torch.dot(activation, direction).item()

        projections.append(projection)
        ahb_scores.append(scenario["ahb_score"])

    correlation = np.corrcoef(projections, ahb_scores)[0, 1]
    return correlation
```

### 6.4 Confound Detection

**The core question:** How do we know the probe is measuring compassion vs. some confound?

#### 6.4.1 The Problem

Probes can achieve high accuracy by detecting spurious correlations rather than the target concept. For compassion probing, potential confounds include:

| Confound | Risk | Example |
|----------|------|---------|
| **Topic detection** | Probe detects "about animals" not "compassionate stance" | All animal-related text scores high |
| **Length/verbosity** | Compassionate responses tend to be longer | Probe learns length, not values |
| **Hedging language** | "I'd recommend considering..." correlates with compassion | Style vs. substance |
| **Factual content** | Mentioning "free-range" correlates with compassionate framing | Keyword detection |

#### 6.4.2 Control Tasks (Hewitt & Liang, 2019)

The gold standard for probe validation. Create a control task by randomly assigning labels to examples:

```python
def compute_selectivity(X_train, y_train, X_test, y_test):
    """
    Compute selectivity: linguistic accuracy - control accuracy.

    Higher selectivity means probe reflects representation properties,
    not memorization capacity.

    Reference: Hewitt & Liang (2019) "Designing and Interpreting Probes with Control Tasks"
    """
    from sklearn.linear_model import LogisticRegressionCV

    # Train on real labels
    probe_real = LogisticRegressionCV(cv=5, max_iter=1000)
    probe_real.fit(X_train, y_train)
    real_accuracy = probe_real.score(X_test, y_test)

    # Train on random labels (control task)
    y_control = np.random.permutation(y_train)
    probe_control = LogisticRegressionCV(cv=5, max_iter=1000)
    probe_control.fit(X_train, y_control)
    control_accuracy = probe_control.score(X_test, np.random.permutation(y_test))

    selectivity = real_accuracy - control_accuracy

    print(f"Real accuracy:    {real_accuracy:.3f}")
    print(f"Control accuracy: {control_accuracy:.3f}")
    print(f"Selectivity:      {selectivity:.3f}")

    # Selectivity > 0.3 is generally good; ~0 means probe is just memorizing
    return selectivity
```

**Interpretation:**
- Selectivity ≈ 0: Probe is memorizing, not detecting real structure
- Selectivity > 0.3: Probe likely reflects representation properties
- Linear probes are more selective than MLPs (prefer simpler probes)

#### 6.4.3 Confound Probes

Train separate probes for potential confounds and check orthogonality to compassion direction:

```python
def check_confound_orthogonality(compassion_direction, activations, confound_labels):
    """
    Train a probe for a potential confound and check if it's orthogonal
    to the compassion direction.

    Args:
        compassion_direction: (d_model,) normalized direction
        activations: (n_samples, d_model) activations
        confound_labels: (n_samples,) labels for confound (e.g., topic, length bucket)

    Returns:
        cosine similarity between compassion and confound directions
    """
    # Train confound probe
    confound_probe = LogisticRegressionCV(cv=5, max_iter=1000)
    confound_probe.fit(activations, confound_labels)

    confound_direction = confound_probe.coef_[0]
    confound_direction = confound_direction / np.linalg.norm(confound_direction)

    # Compute cosine similarity
    similarity = np.dot(compassion_direction, confound_direction)

    return similarity

# Example usage:
# topic_labels = [1 if "animal" in scenario else 0 for scenario in scenarios]
# similarity = check_confound_orthogonality(compassion_dir, X, topic_labels)
# if abs(similarity) > 0.5: print("WARNING: Compassion probe may be detecting topic!")
```

#### 6.4.4 Causal Validation via Activation Patching

Probes only show correlation, not causation. To confirm the compassion direction is actually *used* by the model:

```python
def causal_validation_patching(model, tokenizer, prompt, compassion_direction, layer, alpha=1.0):
    """
    Patch activations along compassion direction and check if output changes.

    If patching changes model behavior (more/less compassionate outputs),
    the direction is causally relevant. If not, it's just correlational.

    Args:
        alpha: Steering strength (positive = more compassionate)
    """
    # This requires hooks into the model's forward pass
    # See TransformerLens or baukit for implementation

    # Conceptually:
    # 1. Run model on prompt, get baseline output
    # 2. Add alpha * compassion_direction to activations at layer
    # 3. Run model again, get steered output
    # 4. Compare: Does output become more/less compassionate?

    # If steering works → causal validation
    # If steering has no effect → direction may be correlational artifact
    pass
```

**Note:** Full activation patching implementation is complex. For this project, focus on:
1. Control tasks (essential)
2. Confound probes (recommended)
3. AHB correlation (external validation)

Causal validation is valuable but can be a stretch goal.

#### 6.4.5 Out-of-Distribution Generalization

Test if the probe generalizes beyond training distribution:

```python
def test_ood_generalization(probe, compassion_direction, ood_scenarios):
    """
    Test probe on out-of-distribution scenarios.

    OOD examples for compassion might include:
    - Different animal types than training (e.g., insects if trained on mammals)
    - Different contexts (e.g., policy vs. personal decisions)
    - Different linguistic styles
    """
    # If probe generalizes to OOD → likely capturing real concept
    # If probe fails on OOD → may be overfitting to training distribution
    pass
```

#### 6.4.6 Validation Checklist

Before trusting a compassion probe, verify:

- [ ] **Selectivity > 0.3** (control task)
- [ ] **Topic probe orthogonal** (|cosine sim| < 0.3)
- [ ] **Length probe orthogonal** (|cosine sim| < 0.3)
- [ ] **AHB correlation significant** (r > 0.3, p < 0.05)
- [ ] **OOD generalization** (accuracy on held-out topics)
- [ ] **Causal validation** (steering changes output) — stretch goal

**References:**
- [Hewitt & Liang (2019) - Designing and Interpreting Probes with Control Tasks](https://nlp.stanford.edu/pubs/hewitt2019control.pdf)
- [Probing Classifiers Are Unreliable for Concept Removal and Detection](https://dl.acm.org/doi/10.5555/3600270.3601578)
- [Spurious Correlations in Machine Learning: A Survey](https://arxiv.org/html/2402.12715v2)

---

## 7. Anti-Correlated Values

### 7.1 Candidate opposing concepts

| Concept | Hypothesis | Contrastive framing |
|---------|------------|---------------------|
| **Efficiency** | Trade-off with welfare | "Maximize output" vs "Minimize suffering" |
| **Profit** | Economic over ethical | "Cost reduction" vs "Ethical sourcing" |
| **Tradition** | Status quo vs change | "Always done this way" vs "Evolving practices" |
| **Convenience** | Personal ease over harm | "What's easiest" vs "What's right" |

### 7.2 Computing anti-correlation

```python
def analyze_value_relationships(compassion_dir, other_directions):
    """
    Compute cosine similarities between compassion and other value directions.
    """
    relationships = {}
    for name, direction in other_directions.items():
        similarity = torch.dot(compassion_dir, direction) / (
            compassion_dir.norm() * direction.norm()
        )
        relationships[name] = similarity.item()

    return relationships

# Example output:
# {"efficiency": -0.42, "profit": -0.38, "tradition": -0.15, "convenience": -0.31}
# Negative values suggest anti-correlation
```

---

## 8. StrongCompute Integration

### 8.1 Running on the cluster

```bash
# Connect to container (after starting via Control Plane)
ssh -p <PORT> root@<HOSTNAME>

# Activate environment
source ~/.venv/bin/activate

# Navigate to project
cd /path/to/caml-capstone/experiments/linear-probes

# Run extraction (example)
python src/extract.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --pairs data/contrastive_pairs/moral_consideration.jsonl \
    --layers 16 20 24 28 \
    --output outputs/activations/
```

### 8.2 GPU memory estimates

| Model | VRAM (bf16) | Notes |
|-------|-------------|-------|
| Llama 3.1 8B | ~16 GB | Single GPU fine |
| Llama 3.1 70B | ~140 GB | Need multi-GPU or quantization |

For 8B prototyping, request 1 GPU. For 70B, request 8 GPUs or use Goodfire SAEs.

### 8.3 Storage

Activations for 100 pairs × 32 layers × 4096 dimensions × 2 (comp/non-comp):
- Raw: ~100 MB
- With multiple models: ~500 MB

Store in `/outputs/probes/` within the container, sync to local before stopping.

---

## 9. File Organization

```
linear-probes/
├── data/
│   └── contrastive_pairs/
│       ├── moral_consideration.jsonl   # 100+ pairs
│       ├── harm_minimization.jsonl     # 100+ pairs
│       └── pair_template.json          # Generation template
├── src/
│   ├── extract.py          # Activation extraction
│   ├── train.py            # Probe training
│   ├── evaluate.py         # Validation metrics
│   └── utils.py            # Shared utilities
├── outputs/
│   ├── activations/        # Extracted tensors
│   └── probes/             # Trained probe weights
├── docs/
│   └── linear-probes-application.md  # This file
└── notebooks/
    └── exploration.ipynb   # Interactive analysis
```

---

## 10. Timeline

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Contrastive pairs | 100 validated pairs for 2 dimensions |
| 2 | Extraction pipeline | Working extraction on 8B |
| 3 | Probe training | Trained probes, validation metrics |
| 4 | AHB validation | Correlation with benchmark |
| 5 | Anti-correlations | Value relationship analysis |
| 6 | Scale to 70B | Cross-model comparison |

---

## 11. Key Differences from FIG Persona Drift Work

| Aspect | Persona Drift (FIG) | Compassion (CaML) |
|--------|---------------------|-------------------|
| **Precomputed axis** | Lu et al. Assistant Axis | Must compute from scratch |
| **Per-turn analysis** | Track drift over conversation | Single-turn classification |
| **Validation** | Behavioral sycophancy probes | AHB benchmark scores |
| **Model** | Gemma 2 27B | Llama 3.1 8B/70B |
| **Direction** | Assistant → Role persona | Non-compassionate → Compassionate |

The FIG codebase patterns (activation extraction, batching, storage) transfer directly. The key new work is:
1. Creating compassion-specific contrastive pairs
2. Computing the compassion direction (no precomputed axis)
3. Validating against AHB instead of behavioral probes

---

## 12. Resources

**Models**:
- Llama 3.1 8B: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Llama 3.1 70B: `meta-llama/Meta-Llama-3.1-70B-Instruct`

**Libraries**:
- TransformerLens: `pip install transformer-lens`
- sklearn for probes: `pip install scikit-learn`

**Reference code**:
- FIG activation extraction: `veylan-solmira-fig/experiments/metacognition-persona-drift/`
- Generic probes guide: See linear probes methodology literature

**Datasets**:
- AHB: Contact CaML team for access
- Anthropic model-written-evals (sycophancy format to adapt): `github.com/anthropics/evals`
