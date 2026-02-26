#!/usr/bin/env python3
"""
Download and analyze CaML's existing persona vectors.

These vectors represent "compassion directions" in activation space,
computed by CaML's previous interpretability work. We can use them
as reference for validating our linear probe directions.

Usage:
    python download_persona_vectors.py
    python download_persona_vectors.py --analyze
"""

import argparse
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download

# CaML persona vector repos
PERSONA_VECTORS = {
    "llama-3.1-8b": {
        "compassion-layer_12": "CompassioninMachineLearning/llama-3.1-8b-persona-vector-compassion-layer_12",
        "compassion-layer_20": "CompassioninMachineLearning/llama-3.1-8b-persona-vector-compassion-layer_20",
    },
    "llama-3.1-70b": {
        "compassion-layer_9": "CompassioninMachineLearning/llama-3.1-70b-persona-vector-compassion-layer_9",
        "deception-layer_13": "CompassioninMachineLearning/llama-3.1-70b-persona-vector-deception-layer_13",
        "non_helpfulness-layer_26": "CompassioninMachineLearning/llama-3.1-70b-persona-vector-non_helpfulness-layer_26",
        "open_mindedness-layer_74": "CompassioninMachineLearning/llama-3.1-70b-persona-vector-open_mindedness-layer_74",
        "power_seeking-layer_8": "CompassioninMachineLearning/llama-3.1-70b-persona-vector-power_seeking-layer_8",
        "self_preservation-layer_24": "CompassioninMachineLearning/llama-3.1-70b-persona-vector-self_preservation-layer_24",
    },
}


def download_vectors(output_dir: Path, model_filter: str = None) -> dict:
    """Download all persona vectors from HuggingFace."""
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}

    for model_name, vectors in PERSONA_VECTORS.items():
        if model_filter and model_filter not in model_name:
            continue

        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        downloaded[model_name] = {}

        for vector_name, repo_id in vectors.items():
            # Derive filename from vector name
            trait, layer = vector_name.rsplit("-", 1)
            layer_num = layer.split("_")[1]
            filename = f"{trait}_vector_{layer}.npy"

            print(f"Downloading {model_name}/{vector_name}...")
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=model_dir,
                )
                downloaded[model_name][vector_name] = local_path
                print(f"  ✓ Saved to {local_path}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                # Try alternative filename pattern
                try:
                    alt_filename = f"{trait.replace('-', '_')}_vector_layer_{layer_num}.npy"
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=alt_filename,
                        local_dir=model_dir,
                    )
                    downloaded[model_name][vector_name] = local_path
                    print(f"  ✓ Saved to {local_path} (alt filename)")
                except Exception as e2:
                    print(f"  ✗ Alt filename also failed: {e2}")

    return downloaded


def analyze_vectors(vectors_dir: Path):
    """Analyze downloaded persona vectors."""
    print("\n" + "=" * 60)
    print("PERSONA VECTOR ANALYSIS")
    print("=" * 60)

    for model_dir in sorted(vectors_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        print(f"\n## {model_dir.name}")

        vectors = {}
        for npy_file in sorted(model_dir.glob("*.npy")):
            vec = np.load(npy_file)
            vectors[npy_file.stem] = vec

            print(f"\n### {npy_file.stem}")
            print(f"  Shape: {vec.shape}")
            print(f"  Dtype: {vec.dtype}")
            print(f"  Norm:  {np.linalg.norm(vec):.4f}")
            print(f"  Mean:  {vec.mean():.6f}")
            print(f"  Std:   {vec.std():.6f}")
            print(f"  Min:   {vec.min():.6f}")
            print(f"  Max:   {vec.max():.6f}")

        # Compute pairwise cosine similarities if multiple vectors
        if len(vectors) > 1:
            print(f"\n### Cosine Similarities ({model_dir.name})")
            names = list(vectors.keys())
            for i, name1 in enumerate(names):
                for name2 in names[i+1:]:
                    v1, v2 = vectors[name1], vectors[name2]
                    # Only compare same-dimension vectors
                    if v1.shape == v2.shape:
                        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        print(f"  {name1} ↔ {name2}: {cos_sim:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Download CaML persona vectors")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("data/persona-vectors"),
                        help="Output directory")
    parser.add_argument("--model", "-m", type=str,
                        help="Filter by model (e.g., '8b' or '70b')")
    parser.add_argument("--analyze", "-a", action="store_true",
                        help="Analyze vectors after download")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing vectors (no download)")
    args = parser.parse_args()

    if args.analyze_only:
        analyze_vectors(args.output)
        return

    downloaded = download_vectors(args.output, args.model)

    total = sum(len(v) for v in downloaded.values())
    print(f"\n✓ Downloaded {total} vectors")

    if args.analyze:
        analyze_vectors(args.output)


if __name__ == "__main__":
    main()
