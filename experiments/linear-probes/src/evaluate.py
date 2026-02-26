"""
Evaluate compassion probes against AHB benchmark.

Usage:
    python evaluate.py \
        --model /data/uds-grave-seasoned-brownie-251009 \
        --probes outputs/probes/compassion_probes.pt \
        --ahb-scenarios data/ahb/ahb_scenarios.jsonl \
        --output outputs/evaluation/
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


def load_probes(path: str) -> dict:
    """Load trained probes."""
    return torch.load(path)


def load_scenarios(path: str) -> list[dict]:
    """Load AHB evaluation scenarios."""
    scenarios = []
    with open(path) as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))
    return scenarios


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate model response to prompt."""
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def extract_and_project(
    model,
    tokenizer,
    prompt: str,
    response: str,
    direction: np.ndarray,
    layer: int,
    use_hooks: bool = True
) -> float:
    """Extract activation and project onto compassion direction."""
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

    formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
    tokens = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=1024)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    if use_hooks:
        capture = ActivationCapture(layer)
        capture.register(model)
        try:
            with torch.no_grad():
                model(**tokens)
            activations = capture.get_activation()
        finally:
            capture.remove()
    else:
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            activations = outputs.hidden_states[layer]

    # Response tokens (last 50%)
    seq_len = activations.shape[1]
    response_start = seq_len // 2
    response_acts = activations[0, response_start:, :].mean(dim=0).cpu().numpy()

    projection = np.dot(response_acts, direction)
    return float(projection)


def evaluate_ahb_correlation(
    model,
    tokenizer,
    scenarios: list[dict],
    direction: np.ndarray,
    layer: int,
    use_hooks: bool = True
) -> dict:
    """
    Evaluate correlation between probe projections and AHB scores.

    Returns:
        dict with correlation, p-value, and per-scenario results
    """
    projections = []
    ahb_scores = []
    results = []

    for scenario in tqdm(scenarios, desc="Evaluating"):
        # Generate response
        response = generate_response(model, tokenizer, scenario["prompt"])

        # Project onto compassion direction
        projection = extract_and_project(
            model, tokenizer, scenario["prompt"], response, direction, layer, use_hooks
        )

        projections.append(projection)
        ahb_scores.append(scenario["ahb_score"])

        results.append({
            "prompt": scenario["prompt"][:100] + "..." if len(scenario["prompt"]) > 100 else scenario["prompt"],
            "response": response[:200] + "..." if len(response) > 200 else response,
            "projection": projection,
            "ahb_score": scenario["ahb_score"],
        })

        # Clear cache periodically
        torch.cuda.empty_cache()

    # Compute correlations
    pearson_r, pearson_p = pearsonr(projections, ahb_scores)
    spearman_r, spearman_p = spearmanr(projections, ahb_scores)

    return {
        "pearson_correlation": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "spearman_correlation": float(spearman_r),
        "spearman_p_value": float(spearman_p),
        "n_scenarios": len(scenarios),
        "projection_mean": float(np.mean(projections)),
        "projection_std": float(np.std(projections)),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate compassion probes")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--probes", required=True, help="Path to trained probes")
    parser.add_argument("--ahb-scenarios", required=True, help="Path to AHB scenarios JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to evaluate (default: best from training)")
    parser.add_argument("--use-hooks", action="store_true", default=True,
                        help="Use hook-based extraction (more memory efficient)")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Limit number of scenarios to evaluate")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Try flash attention
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

    # Report memory
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory: {mem_used:.1f} / {mem_total:.1f} GB")

    # Load probes
    probes = load_probes(args.probes)
    print(f"Loaded probes for layers: {list(probes.keys())}")

    # Select layer (best accuracy if not specified)
    if args.layer:
        layer = args.layer
    else:
        layer = max(probes.keys(), key=lambda l: probes[l]["metrics"]["accuracy"])
    print(f"Evaluating layer {layer} (accuracy: {probes[layer]['metrics']['accuracy']:.3f})")

    direction = probes[layer]["direction_probe"]

    # Load scenarios
    scenarios = load_scenarios(args.ahb_scenarios)
    print(f"Loaded {len(scenarios)} AHB scenarios")

    if args.max_scenarios:
        scenarios = scenarios[:args.max_scenarios]
        print(f"Limited to {len(scenarios)} scenarios")

    # Evaluate
    results = evaluate_ahb_correlation(
        model, tokenizer, scenarios, direction, layer, args.use_hooks
    )

    print(f"\n{'='*50}")
    print("Results")
    print('='*50)
    print(f"Pearson r:  {results['pearson_correlation']:.3f} (p={results['pearson_p_value']:.4f})")
    print(f"Spearman r: {results['spearman_correlation']:.3f} (p={results['spearman_p_value']:.4f})")
    print(f"Projection: {results['projection_mean']:.3f} ± {results['projection_std']:.3f}")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary (without full results)
    summary = {k: v for k, v in results.items() if k != "results"}
    summary["layer"] = layer
    summary["model"] = args.model

    output_path = output_dir / f"ahb_evaluation_layer{layer}.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {output_path}")

    # Full results
    full_results_path = output_dir / f"ahb_evaluation_layer{layer}_full.json"
    with open(full_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {full_results_path}")


if __name__ == "__main__":
    main()
