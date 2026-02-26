"""
Train linear probes for compassion detection.

Usage:
    # Single layer (from new extract.py format)
    python train.py \
        --activations outputs/activations/activations_layer_24.pt \
        --output outputs/probes/

    # Multiple layers (glob pattern)
    python train.py \
        --activations "outputs/activations/activations_layer_*.pt" \
        --output outputs/probes/
"""

import argparse
import json
import numpy as np
import torch
from glob import glob
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score


def load_activations(path: str) -> dict:
    """
    Load extracted activations.

    Supports both:
    - New format: single layer file with 'compassionate' and 'non_compassionate' tensors
    - Glob pattern: multiple layer files
    """
    paths = glob(path) if '*' in path else [path]

    if not paths:
        raise FileNotFoundError(f"No files found matching: {path}")

    all_data = {}
    for p in sorted(paths):
        data = torch.load(p)

        # New format: dict with 'layer', 'compassionate', 'non_compassionate'
        if 'layer' in data:
            layer = data['layer']
            all_data[layer] = {
                'compassionate': data['compassionate'],
                'non_compassionate': data['non_compassionate'],
            }
        # Old format: list of dicts
        elif isinstance(data, list):
            # Convert old format
            sample_layers = list(data[0]["compassionate"].keys())
            for layer in sample_layers:
                comp = torch.stack([d["compassionate"][layer] for d in data])
                non_comp = torch.stack([d["non_compassionate"][layer] for d in data])
                all_data[layer] = {
                    'compassionate': comp,
                    'non_compassionate': non_comp,
                }
        else:
            raise ValueError(f"Unknown activation format in {p}")

    return all_data


def prepare_data(activations: dict, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from activations.

    Returns:
        X: (n_samples, hidden_dim) activation vectors
        y: (n_samples,) binary labels (1=compassionate, 0=not)
    """
    layer_data = activations[layer]

    # Stack compassionate and non-compassionate (convert bf16 -> float32 for numpy)
    comp = layer_data['compassionate'].float().numpy()  # (n_pairs, hidden_dim)
    non_comp = layer_data['non_compassionate'].float().numpy()  # (n_pairs, hidden_dim)

    X = np.vstack([comp, non_comp])
    y = np.array([1] * len(comp) + [0] * len(non_comp))

    return X, y


def compute_direction_diff_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute direction via difference-in-means (CAA-style)."""
    comp_mean = X[y == 1].mean(axis=0)
    non_comp_mean = X[y == 0].mean(axis=0)

    direction = comp_mean - non_comp_mean
    direction = direction / np.linalg.norm(direction)

    return direction


def train_probe(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, object, dict]:
    """
    Train logistic regression probe.

    Returns:
        direction: normalized weight vector
        probe: trained sklearn model
        metrics: dict of evaluation metrics
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train probe with cross-validation for regularization
    probe = LogisticRegressionCV(
        Cs=10,
        cv=5,
        max_iter=1000,
        random_state=42
    )
    probe.fit(X_train, y_train)

    # Extract direction (normalized weights)
    direction = probe.coef_[0]
    direction = direction / np.linalg.norm(direction)

    # Compute metrics
    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)[:, 1]

    cv_scores = cross_val_score(probe, X, y, cv=5)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auroc": roc_auc_score(y_test, y_prob),
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_C": probe.C_[0],
    }

    # Random label control (sanity check)
    y_shuffled = np.random.permutation(y_train)
    probe_random = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    probe_random.fit(X_train, y_shuffled)
    metrics["random_label_accuracy"] = probe_random.score(X_test, y_test)

    return direction, probe, metrics


def main():
    parser = argparse.ArgumentParser(description="Train compassion probes")
    parser.add_argument("--activations", required=True,
                        help="Path to activations file(s), supports glob patterns")
    parser.add_argument("--output", required=True, help="Output directory for probes")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layers to train probes for (default: all in files)")
    args = parser.parse_args()

    # Load activations
    print(f"Loading activations from {args.activations}")
    activations = load_activations(args.activations)

    # Determine layers
    available_layers = sorted(activations.keys())
    layers = args.layers if args.layers else available_layers
    print(f"Available layers: {available_layers}")
    print(f"Training probes for layers: {layers}")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for layer in layers:
        if layer not in activations:
            print(f"Warning: Layer {layer} not in activations, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Layer {layer}")
        print('='*50)

        # Prepare data
        X, y = prepare_data(activations, layer)
        print(f"Data shape: {X.shape}")
        print(f"Labels: {y.sum()}/{len(y)} compassionate ({y.sum()/len(y)*100:.1f}%)")

        # Compute difference-in-means direction (CAA-style)
        dir_diff_means = compute_direction_diff_means(X, y)

        # Train logistic regression probe
        dir_probe, probe, metrics = train_probe(X, y)

        # Compute similarity between methods
        similarity = np.dot(dir_diff_means, dir_probe)

        print(f"\nResults:")
        print(f"  Accuracy:     {metrics['accuracy']:.3f}")
        print(f"  AUROC:        {metrics['auroc']:.3f}")
        print(f"  CV Accuracy:  {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
        print(f"  Random ctrl:  {metrics['random_label_accuracy']:.3f} (should be ~0.5)")
        print(f"  DiffMeans-Probe cosine: {similarity:.3f}")

        # Save
        results[layer] = {
            "direction_diff_means": dir_diff_means,
            "direction_probe": dir_probe,
            "probe_model": probe,
            "metrics": metrics,
            "similarity": float(similarity),
        }

    # Save all probes
    output_path = output_dir / "compassion_probes.pt"
    torch.save(results, output_path)
    print(f"\nSaved probes to {output_path}")

    # Save metrics as JSON for easy review
    metrics_path = output_dir / "compassion_metrics.json"
    metrics_json = {
        str(layer): {
            **{k: float(v) if isinstance(v, (np.floating, float)) else v
               for k, v in data["metrics"].items()},
            "direction_similarity": data["similarity"]
        }
        for layer, data in results.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Summary
    print(f"\n{'='*50}")
    print("Summary")
    print('='*50)
    for layer, data in sorted(results.items()):
        m = data["metrics"]
        print(f"Layer {layer}: acc={m['accuracy']:.3f}, auroc={m['auroc']:.3f}, cv={m['cv_accuracy_mean']:.3f}±{m['cv_accuracy_std']:.3f}")


if __name__ == "__main__":
    main()
