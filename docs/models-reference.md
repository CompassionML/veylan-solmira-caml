# Models Reference

Models used in CaML compassion research and available resources.

---

## Target Models

| Model | Size | Purpose | StrongCompute |
|-------|------|---------|---------------|
| **Llama 3.1 8B Instruct** | 8B | Primary prototype | Mounted at `/data/uds-grave-seasoned-brownie-251009/` |
| **Llama 3.1 70B Instruct** | 70B | Scale-up target | Needs 8 GPUs or quantization |
| **Llama 3.3 70B Instruct** | 70B | SAE analysis | Goodfire SAE available |

### Recommended Approach

1. **Prototype on 8B** - faster iteration, fits on single GPU (RTX 3090 Ti)
2. **Validate on 70B** - once methodology is solid
3. **Cross-model comparison** - measure compassion strength differences

---

## SAE Resources

Sparse Autoencoders available for interpretability work:

| Resource | Model | Coverage | Link |
|----------|-------|----------|------|
| **Goodfire SAE** | Llama 3.1 8B | Layer 19 | [HuggingFace](https://huggingface.co/Goodfire) |
| **Goodfire SAE** | Llama 3.3 70B | Layer 50 | [HuggingFace](https://huggingface.co/Goodfire/Llama-3.3-70B-Instruct-SAE-l50) |
| **Llama Scope** | Llama 3.1 8B | All layers (256 SAEs, 32K-128K features) | [Paper](https://arxiv.org/abs/2410.20526) |
| **qresearch SAE** | DeepSeek-R1-Distill-Llama-70B | Layer 48 | [HuggingFace](https://huggingface.co/qresearch/DeepSeek-R1-Distill-Llama-70B-SAE-l48) |

### SAE Strategy

- **8B exploration**: Use Llama Scope (all layers) for comprehensive feature analysis
- **70B validation**: Use Goodfire SAE (Layer 50) for targeted analysis
- **Feature search**: Use [Neuronpedia](https://neuronpedia.org/) for natural language feature descriptions

---

## Model Specifications

### Llama 3.1 8B Instruct

```
Architecture: LlamaForCausalLM
Hidden size: 4096
Layers: 32
Attention heads: 32
KV heads: 8 (GQA)
Vocab size: 128,256
Max context: 131,072 tokens
Dtype: bfloat16
VRAM: ~15 GB (bf16)
```

HuggingFace: `meta-llama/Meta-Llama-3.1-8B-Instruct`

### Llama 3.1 70B Instruct

```
Architecture: LlamaForCausalLM
Hidden size: 8192
Layers: 80
Attention heads: 64
KV heads: 8 (GQA)
Vocab size: 128,256
Max context: 131,072 tokens
Dtype: bfloat16
VRAM: ~140 GB (bf16)
```

HuggingFace: `meta-llama/Meta-Llama-3.1-70B-Instruct`

---

## Layer Guidance

Based on linear probe literature, for Llama 3.1 8B (32 layers):

| Layer Range | Typical Encoding |
|-------------|------------------|
| 0-8 | Token identity, syntax |
| 8-16 | Semantic features |
| 16-24 | High-level concepts, values |
| 24-32 | Output-relevant features |

**Recommended probe layers**: 16, 20, 24, 28 (sparse sampling in value-encoding range)

---

## Related Models (Not CaML)

| Model | Project | Notes |
|-------|---------|-------|
| Gemma 2 27B | FIG persona drift | Different project, different methodology |

---

## GPU Requirements

| Model | VRAM (bf16) | GPUs Needed |
|-------|-------------|-------------|
| Llama 3.1 8B | ~15 GB | 1x RTX 3090 Ti |
| Llama 3.1 70B | ~140 GB | 8x GPU or quantized |

For 70B on limited hardware, consider:
- 4-bit quantization (~35 GB)
- Using Goodfire SAE features instead of full model
