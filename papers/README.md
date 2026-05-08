# Reference Papers

PDFs in this folder are gitignored (copyright). Download links below.

## Activation Steering

| Paper | Link | Notes |
|-------|------|-------|
| Contrastive Grounding: Hallucination Detection via Multi-Layer Mean Difference | [arXiv](https://arxiv.org/abs/2410.09765) | Detection works (0.93 AUROC), steering doesn't |
| Steering Llama 2 via Contrastive Activation Addition | [arXiv](https://arxiv.org/abs/2312.06681) | ACL 2024, CAA method |
| A Sober Look at Steering Vectors for LLMs | [Alignment Forum](https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/a-sober-look-at-steering-vectors-for-llms) | Reliability evaluation |
| SAE-Targeted Steering | [arXiv](https://arxiv.org/abs/2411.02193) | Improved steering via SAE features |

## Linear Probes / Interpretability

| Paper | Link | Notes |
|-------|------|-------|
| Scaling Monosemanticity | [Anthropic](https://transformer-circuits.pub/2024/scaling-monosemanticity/) | SAE features in Claude 3 Sonnet |
| Gemma Scope | [arXiv](https://arxiv.org/abs/2408.05147) | Open SAEs for Gemma 2 |

## Compassion / Ethics

| Paper | Link | Notes |
|-------|------|-------|
| Self-fulfilling Alignment for Animal Compassion | Draft | CaML alignment approach |
| Animal Harm Benchmark | TBD | AHB methodology |

## Constitutional AI

| Paper | Link | Notes |
|-------|------|-------|
| Constitution or Collapse? Exploring Constitutional AI with Llama 3-8B (Zhang 2025) | [arXiv 2504.04918](https://arxiv.org/abs/2504.04918) | Local: `constitution-or-collapse-cai-llama3-8b.pdf` + `.md`. Single-author Stanford preprint (xuezhang68@stanford.edu). Replicates CAI on Llama 3-8B; finds model collapse traced to emoji-repetition in Stage-1 SFT data. **No model/code released.** Direct precedent for CaML constitution-vs-midtraining paper. |

---

## Local Setup

To populate this folder, download PDFs from the links above and save with descriptive names:
```
papers/
├── README.md                                      # This file (committed)
├── contrastive-grounding-hallucination-detection.pdf
├── steering-llama-2-caa.pdf
└── ...
```
