"""
Train linear probes for compassion detection.

Usage:
    python train.py \
        --activations outputs/activations/activations_layers16_20_24_28.pt \
        --output outputs/probes/
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score


def load_activations(path: str) -> list[dict]:
    """Load extracted activations."""
    return torch.load(path)


def prepare_data(activations: list[dict], layer: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data from activations.

    Returns:
        X: (n_samples, hidden_dim) activation vectors
        y: (n_samples,) binary labels (1=compassionate, 0=not)
    """
    X = []
    y = []

    for act in activations:
        # Compassionate
        X.append(act["compassionate"][layer].numpy())
        y.append(1)
        # Non-compassionate
        X.append(act["non_compassionate"][layer].numpy())
        y.append(0)

    return np.array(X), np.array(y)


def compute_direction_diff_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute direction via difference-in-means."""
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

    # Train probe
    probe = LogisticRegressionCV(
        Cs=10,
        cv=5,
        max_iter=1000,
        random_state=42
    )
    probe.fit(X_train, y_train)

    # Extract direction
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
    }

    # Random label control
    y_shuffled = np.random.permutation(y_train)
    probe_random = LogisticRegressionCV(cv=5, max_iter=1000, random_state=42)
    probe_random.fit(X_train, y_shuffled)
    metrics["random_label_accuracy"] = probe_random.score(X_test, y_test)

    return direction, probe, metrics


def main():
    parser = argparse.ArgumentParser(description="Train compassion probes")
    parser.add_argument("--activations", required=True, help="Path to activations file")
    parser.add_argument("--output", required=True, help="Output directory for probes")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Layers to train probes for (default: all in file)")
    args = parser.parse_args()

    # Load activations
    print(f"Loading activations from {args.activations}")
    activations = load_activations(args.activations)

    # Determine layers
    sample_layers = list(activations[0]["compassionate"].keys())
    layers = args.layers if args.layers else sample_layers
    print(f"Training probes for layers: {layers}")

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for layer in layers:
        print(f"\n=== Layer {layer} ===")

        # Prepare data
        X, y = prepare_data(activations, layer)
        print(f"Data shape: {X.shape}, labels: {y.sum()}/{len(y)} compassionate")

        # Compute difference-in-means direction
        dir_diff_means = compute_direction_diff_means(X, y)

        # Train logistic regression probe
        dir_probe, probe, metrics = train_probe(X, y)

        # Compute similarity between methods
        similarity = np.dot(dir_diff_means, dir_probe)

        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"AUROC: {metrics['auroc']:.3f}")
        print(f"CV Accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
        print(f"Random label control: {metrics['random_label_accuracy']:.3f}")
        print(f"DiffMeans-Probe similarity: {similarity:.3f}")

        # Save
        results[layer] = {
            "direction_diff_means": dir_diff_means,
            "direction_probe": dir_probe,
            "metrics": metrics,
            "similarity": similarity,
        }

    # Save all probes
    output_path = output_dir / "compassion_probes.pt"
    torch.save(results, output_path)
    print(f"\nSaved probes to {output_path}")

    # Save metrics as JSON for easy review
    metrics_path = output_dir / "compassion_metrics.json"
    metrics_json = {
        layer: {k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in data["metrics"].items()}
        for layer, data in results.items()
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
