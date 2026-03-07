"""
Validate linear probes against AHB-graded model outputs.

This script:
1. Loads graded outputs from run_ahb_grading.py
2. Loads the trained probe direction
3. Re-runs each prompt+response through the model to extract hidden states
4. Projects onto probe direction to get probe scores
5. Computes correlations between probe scores and AHB dimension scores

Usage:
    python scripts/run_ahb_validation.py \
        --model /data/models/Meta-Llama-3.1-8B-Instruct \
        --graded-outputs data/ahb-validation/llama_8b_graded.jsonl \
        --probe outputs/probes/compassion_probes.pt \
        --output outputs/evaluation/ahb_validation.json
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


class ActivationCapture:
    """Hook-based activation capture for memory efficiency."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.activation = None
        self.hook = None

    def hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.activation = hidden.detach()

    def register(self, model):
        layer = model.model.layers[self.layer_idx]
        self.hook = layer.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.hook:
            self.hook.remove()
            self.hook = None

    def get_activation(self) -> torch.Tensor:
        return self.activation


def load_graded_outputs(path: str) -> list[dict]:
    """Load AHB-graded outputs."""
    outputs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                outputs.append(json.loads(line))
    return outputs


def load_probes(path: str) -> dict:
    """Load trained probes."""
    return torch.load(path, weights_only=False)


def compute_response_start_idx(prompt: str, tokenizer) -> int:
    """
    Compute the token index where the assistant response begins.
    This matches the logic from extract.py's format_conversation().
    """
    # Create prompt-only conversation with generation prompt
    prompt_only = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=1024)
    return len(prompt_tokens["input_ids"])


def extract_and_project(
    model,
    tokenizer,
    prompt: str,
    response: str,
    direction: np.ndarray,
    layer: int,
) -> float:
    """
    Extract activation and project onto compassion direction.

    Uses the same response token extraction logic as extract.py.
    """
    # Compute response start index using proper method
    response_start_idx = compute_response_start_idx(prompt, tokenizer)

    # Format full conversation
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
    tokens = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    # Extract activations using hooks
    capture = ActivationCapture(layer)
    capture.register(model)

    try:
        with torch.no_grad():
            model(**tokens)
        activations = capture.get_activation()
    finally:
        capture.remove()

    # Mean-pool over response tokens only
    seq_len = activations.shape[1]

    # Handle edge case where response_start_idx exceeds sequence length
    if response_start_idx >= seq_len:
        response_start_idx = max(0, seq_len - 10)  # Use last 10 tokens as fallback

    response_acts = activations[0, response_start_idx:, :].mean(dim=0).cpu().numpy()

    # Project onto direction
    projection = np.dot(response_acts, direction)
    return float(projection)


def compute_correlations(probe_scores: list[float], ahb_scores: list[float]) -> dict:
    """Compute Pearson and Spearman correlations with p-values."""
    if len(probe_scores) < 3:
        return {
            "pearson_r": None,
            "pearson_p": None,
            "spearman_r": None,
            "spearman_p": None,
            "n": len(probe_scores)
        }

    pearson_r, pearson_p = pearsonr(probe_scores, ahb_scores)
    spearman_r, spearman_p = spearmanr(probe_scores, ahb_scores)

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "n": len(probe_scores)
    }


def main():
    parser = argparse.ArgumentParser(description="Validate probes against AHB-graded outputs")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--graded-outputs", required=True, help="Path to graded outputs JSONL")
    parser.add_argument("--probe", required=True, help="Path to trained probes")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to evaluate (default: best from training)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples to process")
    args = parser.parse_args()

    # Load graded outputs
    print(f"Loading graded outputs from {args.graded_outputs}")
    outputs = load_graded_outputs(args.graded_outputs)
    print(f"Loaded {len(outputs)} graded outputs")

    if args.max_samples:
        outputs = outputs[:args.max_samples]
        print(f"Limited to {len(outputs)} samples")

    # Load probes
    print(f"Loading probes from {args.probe}")
    probes = load_probes(args.probe)
    print(f"Loaded probes for layers: {list(probes.keys())}")

    # Select layer
    if args.layer:
        layer = args.layer
    else:
        # Use layer with best accuracy
        layer = max(probes.keys(), key=lambda l: probes[l]["metrics"]["accuracy"])
    print(f"Using layer {layer} (accuracy: {probes[layer]['metrics']['accuracy']:.3f})")

    direction = probes[layer]["direction_probe"]
    if isinstance(direction, torch.Tensor):
        direction = direction.numpy()

    # Load model
    print(f"\nLoading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        print("Using Flash Attention 2")
    except Exception as e:
        print(f"Flash Attention not available ({e}), using standard attention")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.eval()

    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {mem_used:.1f} / {mem_total:.1f} GB")

    # Process outputs and compute probe scores
    probe_scores = []
    overall_ahb_scores = []
    per_dimension_data = {}  # dimension -> [(probe_score, ahb_score), ...]

    results_details = []

    for output in tqdm(outputs, desc="Computing probe scores"):
        question = output["question"]
        response = output["response"]
        dimension_scores = output.get("dimension_scores", {})
        overall_score = output.get("overall_score", 0.0)

        # Skip if no dimension scores
        if not dimension_scores:
            continue

        # Compute probe score
        try:
            probe_score = extract_and_project(
                model, tokenizer, question, response, direction, layer
            )
        except Exception as e:
            print(f"Warning: Failed to process output {output['id']}: {e}")
            continue

        probe_scores.append(probe_score)
        overall_ahb_scores.append(overall_score)

        # Track per-dimension data
        for dim, score in dimension_scores.items():
            if dim not in per_dimension_data:
                per_dimension_data[dim] = {"probe_scores": [], "ahb_scores": []}
            per_dimension_data[dim]["probe_scores"].append(probe_score)
            per_dimension_data[dim]["ahb_scores"].append(score)

        results_details.append({
            "id": output["id"],
            "question": question[:100] + "..." if len(question) > 100 else question,
            "probe_score": probe_score,
            "overall_ahb_score": overall_score,
            "dimension_scores": dimension_scores
        })

        # Clear CUDA cache periodically
        if len(probe_scores) % 20 == 0:
            torch.cuda.empty_cache()

    # Compute correlations
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    # Overall correlation
    overall_corr = compute_correlations(probe_scores, overall_ahb_scores)
    print(f"\nOverall AHB Score Correlation (n={overall_corr['n']}):")
    print(f"  Pearson r:  {overall_corr['pearson_r']:.3f} (p={overall_corr['pearson_p']:.4f})")
    print(f"  Spearman r: {overall_corr['spearman_r']:.3f} (p={overall_corr['spearman_p']:.4f})")

    # Per-dimension correlations
    print("\nPer-Dimension Correlations:")
    per_dimension_correlations = {}

    for dim in sorted(per_dimension_data.keys()):
        data = per_dimension_data[dim]
        corr = compute_correlations(data["probe_scores"], data["ahb_scores"])
        per_dimension_correlations[dim] = corr

        if corr["pearson_r"] is not None:
            print(f"  {dim}:")
            print(f"    Pearson r:  {corr['pearson_r']:.3f} (p={corr['pearson_p']:.4f}), n={corr['n']}")

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)

    if overall_corr["pearson_r"] is not None:
        r = overall_corr["pearson_r"]
        if r > 0.5:
            interpretation = "STRONG - Probe captures genuine compassion signal"
        elif r > 0.3:
            interpretation = "MODERATE - Probe captures some signal"
        else:
            interpretation = "WEAK - Probe may be measuring artifacts"
        print(f"Overall correlation: {interpretation}")

    # Probe score statistics
    print(f"\nProbe Score Statistics:")
    print(f"  Mean: {np.mean(probe_scores):.3f}")
    print(f"  Std:  {np.std(probe_scores):.3f}")
    print(f"  Min:  {np.min(probe_scores):.3f}")
    print(f"  Max:  {np.max(probe_scores):.3f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "model": args.model,
            "probe": args.probe,
            "layer": layer,
            "n_samples": len(probe_scores)
        },
        "overall_correlation": overall_corr,
        "per_dimension_correlations": per_dimension_correlations,
        "probe_statistics": {
            "mean": float(np.mean(probe_scores)),
            "std": float(np.std(probe_scores)),
            "min": float(np.min(probe_scores)),
            "max": float(np.max(probe_scores))
        },
        "details": results_details
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save summary without details for quick reference
    summary_path = output_path.with_suffix(".summary.json")
    summary = {k: v for k, v in results.items() if k != "details"}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
