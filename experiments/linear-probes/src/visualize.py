"""
Generate visualizations for compassion probe results.

Usage:
    python visualize.py \
        --activations outputs/activations/activations_layer_24.pt \
        --probes outputs/probes/compassion_probes.pt \
        --output outputs/figures/
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def load_data(activations_path: str, probes_path: str, layer: int):
    """Load activations and probes."""
    act_data = torch.load(activations_path, weights_only=False)
    probes = torch.load(probes_path, weights_only=False)

    # Handle new format
    if 'layer' in act_data:
        comp = act_data['compassionate'].float().numpy()
        non_comp = act_data['non_compassionate'].float().numpy()
    else:
        raise ValueError("Unknown activation format")

    X = np.vstack([comp, non_comp])
    y = np.array([1] * len(comp) + [0] * len(non_comp))

    direction = probes[layer]['direction_probe']
    probe_model = probes[layer].get('probe_model')

    return X, y, direction, probe_model


def plot_projection_distribution(X, y, direction, output_path):
    """Plot distribution of projections onto compassion direction."""
    projections = X @ direction

    comp_proj = projections[y == 1]
    non_comp_proj = projections[y == 0]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(comp_proj, bins=30, alpha=0.7, label='Compassionate', color='#2ecc71', edgecolor='white')
    ax.hist(non_comp_proj, bins=30, alpha=0.7, label='Non-compassionate', color='#e74c3c', edgecolor='white')

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Decision boundary')

    ax.set_xlabel('Projection onto Compassion Direction', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Activation Projections: Compassionate vs Non-Compassionate Responses', fontsize=14)
    ax.legend(fontsize=11)

    # Add stats
    separation = comp_proj.mean() - non_comp_proj.mean()
    pooled_std = np.sqrt((comp_proj.std()**2 + non_comp_proj.std()**2) / 2)
    d_prime = separation / pooled_std

    stats_text = f"d' = {d_prime:.2f}\nSeparation = {separation:.2f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return d_prime


def plot_roc_curve(X, y, probe_model, output_path):
    """Plot ROC curve."""
    # Split same as training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_prob = probe_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.3, color='#3498db')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: Compassion Probe (Layer 24)', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return roc_auc


def plot_confusion_matrix(X, y, probe_model, output_path):
    """Plot confusion matrix."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = probe_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Non-compassionate', 'Compassionate']
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    ax.set_title('Confusion Matrix: Compassion Probe (Layer 24)', fontsize=14)

    # Add accuracy annotation
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.text(0.5, -0.15, f'Accuracy: {accuracy:.1%}', transform=ax.transAxes,
            ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_direction_comparison(probes, output_path):
    """Plot comparison between diff-means and probe directions."""
    layers = sorted(probes.keys())

    if len(layers) == 1:
        # Single layer: show cosine similarity as a simple bar
        layer = layers[0]
        similarity = probes[layer]['similarity']

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Layer 24'], [similarity], color='#9b59b6', width=0.5)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Diff-Means vs Probe Direction Similarity', fontsize=14)
        ax.set_ylim([0, 1])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

        for i, v in enumerate([similarity]):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11)

    else:
        # Multiple layers: show comparison
        similarities = [probes[l]['similarity'] for l in layers]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([f'Layer {l}' for l in layers], similarities, color='#9b59b6')
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title('Diff-Means vs Probe Direction Similarity by Layer', fontsize=14)
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate probe visualizations")
    parser.add_argument("--activations", required=True, help="Path to activations file")
    parser.add_argument("--probes", required=True, help="Path to probes file")
    parser.add_argument("--output", required=True, help="Output directory for figures")
    parser.add_argument("--layer", type=int, default=24, help="Layer to visualize")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data...")
    X, y, direction, probe_model = load_data(args.activations, args.probes, args.layer)
    print(f"Data shape: {X.shape}, Labels: {y.sum()}/{len(y)} compassionate")

    # Generate visualizations
    print("\nGenerating visualizations...")

    d_prime = plot_projection_distribution(
        X, y, direction,
        output_dir / "projection_distribution.png"
    )
    print(f"  d' (discriminability): {d_prime:.2f}")

    if probe_model is not None:
        roc_auc = plot_roc_curve(
            X, y, probe_model,
            output_dir / "roc_curve.png"
        )

        plot_confusion_matrix(
            X, y, probe_model,
            output_dir / "confusion_matrix.png"
        )

    # Load full probes for direction comparison
    probes = torch.load(args.probes, weights_only=False)
    plot_direction_comparison(
        probes,
        output_dir / "direction_similarity.png"
    )

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
