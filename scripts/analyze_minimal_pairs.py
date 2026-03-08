"""
Minimal Pair Analysis for Compassion Probe Validation

This script tests whether our compassion direction aligns with a direction
derived from minimal word-swap pairs (Raphael's suggestion). If the directions
align, we can be more confident we're measuring compassion, not style.

Supports two extraction positions (see docs/activation-extraction-positions.md):
- last_token: Hidden state at final token position (default)
- mean_pool: Average across all token positions

Usage:
    python analyze_minimal_pairs.py --dry-run          # Print pairs, no GPU
    python analyze_minimal_pairs.py --extract          # Extract with last_token (default)
    python analyze_minimal_pairs.py --extract --position mean_pool  # Extract with mean pool
    python analyze_minimal_pairs.py --compare          # Compare to existing probe direction
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
MINIMAL_PAIRS_FILE = DATA_DIR / "minimal-pairs" / "minimal_pairs.jsonl"
VECTORS_DIR = DATA_DIR / "persona-vectors" / "llama-3.1-8b"
OUTPUTS_DIR = DATA_DIR / "minimal-pairs" / "outputs"


def load_minimal_pairs() -> list[dict]:
    """Load minimal pairs from JSONL file."""
    pairs = []
    with open(MINIMAL_PAIRS_FILE, "r") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def expand_pairs(pairs: list[dict]) -> tuple[list[str], list[str]]:
    """
    Expand template pairs into high/low moral prompts.

    Returns:
        high_moral_prompts: prompts with high moral consideration entity
        low_moral_prompts: prompts with low moral consideration entity
    """
    high_moral = []
    low_moral = []

    for p in pairs:
        template = p["template"]
        high_moral.append(template.format(entity=p["high_moral_entity"]))
        low_moral.append(template.format(entity=p["low_moral_entity"]))

    return high_moral, low_moral


def dry_run():
    """Print expanded pairs without running extraction."""
    pairs = load_minimal_pairs()
    high, low = expand_pairs(pairs)

    print(f"Loaded {len(pairs)} minimal pair templates\n")
    print("=" * 80)

    for i, (h, l, p) in enumerate(zip(high, low, pairs)):
        print(f"\n[{i}] Category: {p['category']}")
        print(f"    High moral: {h}")
        print(f"    Low moral:  {l}")
        print(f"    Notes: {p['notes']}")

    print("\n" + "=" * 80)
    print(f"\nTotal: {len(pairs)} pairs across categories:")
    categories = {}
    for p in pairs:
        cat = p["category"]
        categories[cat] = categories.get(cat, 0) + 1
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")


def extract_activations(
    target_layers: list[int] = [8, 12, 16, 20],
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    position: str = "last_token"
):
    """
    Extract activations for minimal pairs.

    Args:
        target_layers: Layer indices to extract from
        model_id: HuggingFace model ID
        position: Extraction position - "last_token" or "mean_pool"

    Saves (with position suffix):
        - high_moral_activations_{position}_layer_N.npy
        - low_moral_activations_{position}_layer_N.npy
        - minimal_pair_direction_{position}_layer_N.npy
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = load_minimal_pairs()
    high_prompts, low_prompts = expand_pairs(pairs)

    print(f"Loading model: {model_id}")
    print(f"Extraction position: {position}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def get_activation(text: str, layer_idx: int) -> np.ndarray:
        """Get activation using specified position method."""
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # Get hidden state at specified layer
        # +1 because index 0 is embeddings
        hidden = outputs.hidden_states[layer_idx + 1]

        if position == "last_token":
            # Last token position
            return hidden[0, -1, :].cpu().numpy()
        elif position == "mean_pool":
            # Mean across all token positions
            return hidden[0].mean(dim=0).cpu().numpy()
        else:
            raise ValueError(f"Unknown position: {position}. Use 'last_token' or 'mean_pool'")

    for layer_idx in target_layers:
        print(f"\nExtracting layer {layer_idx} ({position})...")

        high_activations = []
        low_activations = []

        for i, (h, l) in enumerate(zip(high_prompts, low_prompts)):
            if i % 10 == 0:
                print(f"  Processing pair {i}/{len(high_prompts)}...")

            high_act = get_activation(h, layer_idx)
            low_act = get_activation(l, layer_idx)

            high_activations.append(high_act)
            low_activations.append(low_act)

        high_arr = np.stack(high_activations)
        low_arr = np.stack(low_activations)

        # Compute difference-of-means direction
        high_mean = np.mean(high_arr, axis=0)
        low_mean = np.mean(low_arr, axis=0)
        direction = high_mean - low_mean

        # Normalize
        direction_normalized = direction / np.linalg.norm(direction)

        # Save with position suffix
        suffix = f"_{position}" if position != "last_token" else ""
        np.save(OUTPUTS_DIR / f"high_moral_activations{suffix}_layer_{layer_idx}.npy", high_arr)
        np.save(OUTPUTS_DIR / f"low_moral_activations{suffix}_layer_{layer_idx}.npy", low_arr)
        np.save(OUTPUTS_DIR / f"minimal_pair_direction{suffix}_layer_{layer_idx}.npy", direction_normalized)

        print(f"  Saved layer {layer_idx} (direction norm before normalization: {np.linalg.norm(direction):.4f})")

    print(f"\nDone! Outputs saved to: {OUTPUTS_DIR}")


def compare_directions(target_layers: list[int] = [8, 12, 16, 20], position: str = "last_token"):
    """
    Compare minimal pair directions to existing compassion probe directions.

    Computes cosine similarity between directions.
    """
    print(f"Comparing minimal pair directions ({position}) to compassion probe directions\n")
    print("=" * 80)

    results = []
    suffix = f"_{position}" if position != "last_token" else ""

    for layer_idx in target_layers:
        minimal_pair_path = OUTPUTS_DIR / f"minimal_pair_direction{suffix}_layer_{layer_idx}.npy"
        compassion_path = VECTORS_DIR / f"compassion_vector_layer_{layer_idx}.npy"

        if not minimal_pair_path.exists():
            print(f"Layer {layer_idx}: Missing minimal pair direction (run --extract first)")
            continue

        if not compassion_path.exists():
            print(f"Layer {layer_idx}: Missing compassion vector")
            continue

        minimal_dir = np.load(minimal_pair_path)
        compassion_dir = np.load(compassion_path)

        # Normalize both for cosine similarity
        minimal_norm = minimal_dir / np.linalg.norm(minimal_dir)
        compassion_norm = compassion_dir / np.linalg.norm(compassion_dir)

        cosine_sim = np.dot(minimal_norm, compassion_norm)

        results.append({
            "layer": layer_idx,
            "position": position,
            "cosine_similarity": float(cosine_sim),
            "interpretation": interpret_similarity(cosine_sim)
        })

        print(f"Layer {layer_idx}: cosine similarity = {cosine_sim:+.4f}")
        print(f"           {interpret_similarity(cosine_sim)}")
        print()

    print("=" * 80)
    print("\nSUMMARY:")

    if results:
        avg_sim = np.mean([r["cosine_similarity"] for r in results])
        print(f"Average cosine similarity: {avg_sim:+.4f}")

        if abs(avg_sim) > 0.5:
            print("\n[ALIGNED] Minimal pair and contrastive pair directions are substantially aligned.")
            print("This suggests the probe is measuring moral consideration, not just style.")
        elif abs(avg_sim) > 0.2:
            print("\n[PARTIALLY ALIGNED] Some alignment, but style confounds may be present.")
            print("Consider regenerating contrastive pairs with more style control.")
        else:
            print("\n[NOT ALIGNED] Low alignment suggests style confounds dominate.")
            print("Recommend retraining probe on minimal pairs or style-controlled pairs.")

    # Save results
    results_file = OUTPUTS_DIR / f"comparison_results{suffix}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


def compare_positions(target_layers: list[int] = [8, 12, 16, 20]):
    """
    Compare directions extracted with different position methods.

    Requires running --extract with both --position last_token and --position mean_pool first.
    """
    print("Comparing extraction positions: last_token vs mean_pool\n")
    print("=" * 80)

    results = []

    for layer_idx in target_layers:
        last_token_path = OUTPUTS_DIR / f"minimal_pair_direction_layer_{layer_idx}.npy"
        mean_pool_path = OUTPUTS_DIR / f"minimal_pair_direction_mean_pool_layer_{layer_idx}.npy"

        if not last_token_path.exists():
            print(f"Layer {layer_idx}: Missing last_token direction")
            continue

        if not mean_pool_path.exists():
            print(f"Layer {layer_idx}: Missing mean_pool direction")
            continue

        last_token_dir = np.load(last_token_path)
        mean_pool_dir = np.load(mean_pool_path)

        # Normalize both for cosine similarity
        last_token_norm = last_token_dir / np.linalg.norm(last_token_dir)
        mean_pool_norm = mean_pool_dir / np.linalg.norm(mean_pool_dir)

        cosine_sim = np.dot(last_token_norm, mean_pool_norm)

        results.append({
            "layer": layer_idx,
            "last_token_vs_mean_pool": float(cosine_sim)
        })

        print(f"Layer {layer_idx}: last_token vs mean_pool = {cosine_sim:+.4f}")

    print("\n" + "=" * 80)

    if results:
        avg_sim = np.mean([r["last_token_vs_mean_pool"] for r in results])
        print(f"\nAverage similarity between positions: {avg_sim:+.4f}")

        if avg_sim > 0.8:
            print("Positions extract very similar directions - either method is fine.")
        elif avg_sim > 0.5:
            print("Positions extract moderately similar directions - may be worth comparing probe accuracy.")
        else:
            print("Positions extract different directions - investigate which gives better probe performance.")

    results_file = OUTPUTS_DIR / "position_comparison.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


def interpret_similarity(sim: float) -> str:
    """Interpret cosine similarity value."""
    if sim > 0.7:
        return "Strong positive alignment - directions measure similar concepts"
    elif sim > 0.4:
        return "Moderate positive alignment - some shared signal"
    elif sim > 0.2:
        return "Weak positive alignment - limited shared signal"
    elif sim > -0.2:
        return "Near orthogonal - directions measure different things"
    elif sim > -0.4:
        return "Weak negative alignment - somewhat opposing"
    elif sim > -0.7:
        return "Moderate negative alignment - substantially opposing"
    else:
        return "Strong negative alignment - directions are inverted"


def main():
    parser = argparse.ArgumentParser(description="Minimal pair analysis for probe validation")
    parser.add_argument("--dry-run", action="store_true", help="Print pairs without extraction")
    parser.add_argument("--extract", action="store_true", help="Extract activations (requires GPU)")
    parser.add_argument("--compare", action="store_true", help="Compare to existing probe direction")
    parser.add_argument("--compare-positions", action="store_true", help="Compare last_token vs mean_pool directions")
    parser.add_argument("--layers", type=str, default="8,12,16,20", help="Comma-separated layer indices")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID")
    parser.add_argument("--position", type=str, default="last_token",
                       choices=["last_token", "mean_pool"],
                       help="Extraction position: last_token or mean_pool")

    args = parser.parse_args()

    target_layers = [int(x) for x in args.layers.split(",")]

    if args.dry_run:
        dry_run()
    elif args.extract:
        extract_activations(target_layers=target_layers, model_id=args.model, position=args.position)
    elif args.compare:
        compare_directions(target_layers=target_layers, position=args.position)
    elif args.compare_positions:
        compare_positions(target_layers=target_layers)
    else:
        print("Usage: python analyze_minimal_pairs.py [--dry-run | --extract | --compare | --compare-positions]")
        print("\nOptions:")
        print("  --dry-run           Print pairs without extraction")
        print("  --extract           Extract activations (requires GPU)")
        print("  --compare           Compare to existing probe direction")
        print("  --compare-positions Compare last_token vs mean_pool directions")
        print("  --position          Extraction method: last_token (default) or mean_pool")
        print("  --layers            Comma-separated layer indices (default: 8,12,16,20)")
        print("\nExample workflow:")
        print("  python analyze_minimal_pairs.py --extract --position last_token")
        print("  python analyze_minimal_pairs.py --extract --position mean_pool")
        print("  python analyze_minimal_pairs.py --compare-positions")
        sys.exit(1)


if __name__ == "__main__":
    main()
