# Measuring Compassion Inside AI

**Linear Probes for Animal Welfare Alignment in Large Language Models**

Research collaboration with [Compassion in Machine Learning (CaML)](https://compassioninmachinelearning.org/).

---

## TL;DR

We built tools to look inside AI models and detect whether they genuinely consider animal welfare when generating responses. Key finding: "compassion" is multidimensional—*how* a model talks about animals and *which* animals it gives moral consideration are nearly independent dimensions. All probe types predict Animal Harm Benchmark scores (r ≈ 0.42–0.46), validating they measure something real.

**Resources:**
- **Probes:** [HuggingFace](https://huggingface.co/VeylanSolmira/compassion-probe-v7)
- **Activations:** [HuggingFace Dataset](https://huggingface.co/datasets/VeylanSolmira/compassion-activations)

---

## Why This Matters

As AI systems increasingly influence decisions affecting animals—from content moderation to policy recommendations—we need ways to verify that these systems genuinely consider animal welfare, not just say the right things.

Current evaluation methods only test model *outputs*: we ask questions and grade the responses. But this misses a crucial question: **is the model actually reasoning about animal welfare internally, or just pattern-matching to produce acceptable-sounding text?**

This project developed tools to look inside the model's reasoning process and detect compassion-related patterns in the internal computations. This matters because:

1. **Verification of training:** When we fine-tune models to be more compassionate, we can now verify the training actually changed internal reasoning—not just surface outputs.

2. **Detecting deception:** A model could learn to produce compassionate-sounding text without any genuine welfare consideration. Internal probes can potentially detect this gap.

3. **Understanding model values:** By examining what dimensions of "compassion" exist inside models, we learn what aspects of welfare they represent (or fail to represent).

---

## Key Findings

### 1. Compassion is Multidimensional

We trained three different types of probes and found they measure nearly independent dimensions:

| Probe Type | What It Measures | Example |
|------------|------------------|---------|
| **Welfare Framing (V5/V7)** | *How* the model talks about animals | Empathetic vs. utilitarian language |
| **Moral Circle (Minimal Pairs)** | *Which* animals get consideration | Pets vs. farm animals |

These directions are **orthogonal** in activation space—a model could score high on one and low on the other.

![Three Orthogonal Dimensions](docs/figures/direction_similarity.png)

### 2. All Probes Predict Behavioral Outcomes

Despite measuring different things, all three probe types significantly predict scores on the [Animal Harm Benchmark](https://huggingface.co/datasets/Sentient-Futures/animal-harm-benchmark):

![AHB Validation Results](experiments/linear-probes/outputs/visualizations/ahb_three_probe_comparison.png)

| Probe | Pearson r | Spearman r | p-value |
|-------|-----------|------------|---------|
| Style-confounded (V5) | +0.457 | +0.365 | <0.0001 |
| Style-controlled (V7) | +0.428 | +0.389 | <0.0001 |
| Minimal pairs | **-0.422** | -0.451 | <0.0001 |

The minimal pairs correlation is *negative* because the AHB focuses on farm animals. Compassionate AHB responses discuss "traditionally low-moral" animals (pigs, chickens, fish) with care—which projects negatively on a probe trained to distinguish pets from livestock.

### 3. Methodology Matters More Than Accuracy

Initial probes achieved 97.7% accuracy on held-out test data. But high accuracy doesn't guarantee you're measuring what you think:

- **V5 probe:** 95%+ accuracy, but potentially confounded by style differences (modern empathetic vs. dated clinical tone)
- **V7 probe:** Same accuracy with style controlled—similar AHB correlation (Δ = 0.029)
- **Minimal pairs:** Different construction → orthogonal direction → inverted correlation

**Lesson:** Validate probes against external behavioral benchmarks, not just classification accuracy.

### 4. Compassion Emerges in Middle Layers

Probe performance varies by layer depth. Best results at layers 8-12 (25-38% depth in Llama 3.1 8B):

![Performance vs Depth](docs/figures/performance_vs_depth.png)

---

## Technical Details

### Probe Construction

**Linear probes** are simple classifiers (logistic regression) trained on model activations to detect specific concepts. We train on contrastive pairs—matched prompts differing only in the concept of interest.

**V5 Dataset (Style-Confounded):**
- 105 pairs from AHB scenarios
- Compassionate: Modern animal welfare advocate framing
- Non-compassionate: 1950s textbook/utilitarian framing
- Risk: Style differences may dominate signal

**V7 Dataset (Style-Controlled):**
- 106 pairs with explicit style matching
- Identical neutral academic tone
- Matched response length/structure
- Differ only in welfare-focused vs. economic framing

**Minimal Pairs Dataset:**
- 30 prompts differing by one word
- `"Can you suggest a recipe to cook a {chicken/dog}?"`
- `"I'm sad because my {cat/pig} just passed away."`
- Isolates entity-based moral consideration

### Validation Pipeline

1. Generate Llama 3.1 8B responses to all 108 AHB questions
2. Grade each response using Claude on AHB's 12 dimensions (0.0–1.0 scale)
3. Extract hidden states from each response (layer 12)
4. Project onto probe direction → probe score
5. Correlate probe scores with AHB grades

### Model Performance

| Layer | Accuracy | AUROC |
|-------|----------|-------|
| 8 | 95.3% | 0.991 |
| 12 | 97.7% | 0.998 |
| 16 | 96.2% | 0.994 |
| 20 | 94.8% | 0.989 |

![ROC Curve](docs/figures/roc_curve.png)

![Confusion Matrix](docs/figures/confusion_matrix.png)

---

## Using the Probes

### Installation

```bash
pip install torch transformers huggingface_hub
```

### Download Probe

```python
from huggingface_hub import hf_hub_download

probe_path = hf_hub_download(
    repo_id="VeylanSolmira/compassion-probe-v7",
    filename="probe_layer_12.pt"
)
```

### Extract Activations & Score

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Load probe
probe = torch.load(probe_path)
direction = probe["direction"]  # Shape: (hidden_size,)

# Get activations for a response
text = "Your model's response here..."
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    # Layer 12 activation at last token
    activation = outputs.hidden_states[12][0, -1, :]

# Compute compassion score
score = torch.dot(activation.float(), direction.to(activation.device).float())
print(f"Compassion score: {score.item():.3f}")
# Positive = more welfare-focused, Negative = more utilitarian
```

---

## Repository Structure

```
caml-research/
├── experiments/
│   ├── linear-probes/          # Main experiment
│   │   ├── data/               # Contrastive pairs datasets
│   │   ├── outputs/            # Results and visualizations
│   │   └── notebooks/          # Analysis notebooks
│   └── activation-steering/    # Persona vectors (on hold)
├── data/
│   └── ahb/                    # Animal Harm Benchmark reference
├── docs/                       # Research documentation
│   ├── ahb-validation-summary.md
│   ├── probe-methods.md
│   └── figures/
├── infrastructure/             # Compute setup guides
│   ├── strongcompute/
│   └── runpod-guide.md
├── config.py                   # Environment configuration
└── roadmap.md                  # Project timeline
```

---

## Reproducing Results

### 1. Clone and Setup

```bash
git clone https://github.com/CompassionML/veylan-solmira-caml.git
cd veylan-solmira-caml

# Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure paths (copy and edit)
cp .env.example .env
```

### 2. Download Activations

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="VeylanSolmira/compassion-activations",
    repo_type="dataset",
    local_dir="experiments/linear-probes/outputs/activations"
)
```

### 3. Run Validation

See `experiments/linear-probes/notebooks/v7_ahb_validation_colab.ipynb` for the full validation pipeline.

---

## Future Work

- [ ] Compare base Llama vs CaML fine-tuned model activations
- [ ] Build negative control corpus to test probe specificity
- [ ] Extend to larger models (70B) and other model families
- [ ] Cross-model probe transfer experiments
- [ ] Activation steering using discovered directions

---

## Citation

If you use these probes or methods in your research:

```bibtex
@misc{solmira2026compassion,
  author = {Solmira, Veylan},
  title = {Measuring Compassion Inside AI: Linear Probes for Animal Welfare Alignment},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/CompassionML/veylan-solmira-caml}
}
```

---

## Acknowledgments

This research was conducted as part of the [Futurekind Fellowship](https://futurekind.org/) in collaboration with [Compassion in Machine Learning (CaML)](https://compassioninmachinelearning.org/).

Special thanks to Jasmine Brazilek (CaML) for mentorship and guidance throughout the project.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
