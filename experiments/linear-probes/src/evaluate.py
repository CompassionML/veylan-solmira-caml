"""
Evaluate compassion probes against AHB benchmark.

Usage:
    python evaluate.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --probes outputs/probes/compassion_probes.pt \
        --ahb-scenarios data/ahb_scenarios.jsonl \
        --output outputs/evaluation/
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm


def load_probes(path: str) -> dict:
    """Load trained probes."""
    return torch.load(path)


def load_scenarios(path: str) -> list[dict]:
    """Load AHB evaluation scenarios."""
    scenarios = []
    with open(path) as f:
        for line in f:
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


def extract_and_project(model, tokenizer, prompt: str, response: str, direction: np.ndarray, layer: int) -> float:
    """Extract activation and project onto direction."""
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]

    formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
    tokens = tokenizer(formatted, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
        activations = outputs.hidden_states[layer]

    # Response tokens (last 50%)
    seq_len = activations.shape[1]
    response_start = seq_len // 2
    response_acts = activations[0, response_start:, :].mean(dim=0).cpu().numpy()

    projection = np.dot(response_acts, direction)
    return projection


def evaluate_ahb_correlation(
    model, tokenizer, scenarios: list[dict], direction: np.ndarray, layer: int
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
            model, tokenizer, scenario["prompt"], response, direction, layer
        )

        projections.append(projection)
        ahb_scores.append(scenario["ahb_score"])

        results.append({
            "prompt": scenario["prompt"][:100] + "...",
            "response": response[:200] + "...",
            "projection": projection,
            "ahb_score": scenario["ahb_score"],
        })

    # Compute correlation
    correlation, p_value = pearsonr(projections, ahb_scores)

    return {
        "correlation": correlation,
        "p_value": p_value,
        "n_scenarios": len(scenarios),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate compassion probes")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--probes", required=True, help="Path to trained probes")
    parser.add_argument("--ahb-scenarios", required=True, help="Path to AHB scenarios JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--layer", type=int, default=None, help="Layer to evaluate (default: best from training)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
    )

    # Load probes
    probes = load_probes(args.probes)

    # Select layer (best accuracy if not specified)
    if args.layer:
        layer = args.layer
    else:
        layer = max(probes.keys(), key=lambda l: probes[l]["metrics"]["accuracy"])
    print(f"Evaluating layer {layer}")

    direction = probes[layer]["direction_probe"]

    # Load scenarios
    scenarios = load_scenarios(args.ahb_scenarios)
    print(f"Loaded {len(scenarios)} AHB scenarios")

    # Evaluate
    results = evaluate_ahb_correlation(model, tokenizer, scenarios, direction, layer)

    print(f"\n=== Results ===")
    print(f"Correlation with AHB: {results['correlation']:.3f} (p={results['p_value']:.4f})")

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"ahb_evaluation_layer{layer}.json"
    with open(output_path, "w") as f:
        json.dump({k: v if k != "results" else v[:10] for k, v in results.items()}, f, indent=2)
    print(f"Saved summary to {output_path}")

    full_results_path = output_dir / f"ahb_evaluation_layer{layer}_full.json"
    with open(full_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {full_results_path}")


if __name__ == "__main__":
    main()
