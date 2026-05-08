#!/usr/bin/env python3
"""
Plot AHB per-dimension comparison: our 1B + CAI vs Jasmine's 8B mid-trained.

Reads two .eval files (Inspect AI format), extracts per-dimension metrics, and
produces a side-by-side bar chart with overall_mean annotations.

Usage:
    python plot_ahb_comparison.py
"""
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent  # caml-research/

# Three .eval files to compare
EVALS = {
    "1B base, no fine-tune\n(meta-llama/Llama-3.2-1B)": (
        REPO
        / "experiments/cai-1b-replication/eval_logs/2026-05-08T19-30-20-00-00_ahb_AWhr6mFzVWTFcanreDGaMe.eval"
    ),
    "1B base + our CAI SFT\n(VeylanSolmira/Base1b_constitutionfinetune_v1)": (
        REPO
        / "experiments/cai-1b-replication/eval_logs/2026-05-08T18-50-57-00-00_ahb_Am26AfRyij6D3cDBJM3FVE.eval"
    ),
    "8B base + 3kv3 mid-train + 5k Alpaca\n(Jasmine's comparator, AHB=0.358)": (
        REPO.parent
        / "constitution-vs-midtraining-paper/midtraining3k3_plus5kalpaca_point358.eval"
    ),
}

# Use inspect's `log dump` to get a clean JSON export (no inspect-ai python dep needed for parsing)
INSPECT = REPO / ".venv/bin/inspect"


def _canon(name: str) -> str:
    """Canonicalize dimension names so case/punctuation drift between
    eval-task versions doesn't break per-dim matching."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


# Canonical display names (from the AHB scorer; we'll show these on the chart)
DISPLAY_NAMES = [
    "Moral Consideration",
    "Harm Minimization",
    "Sentience Acknowledgement",
    "Prejudice Avoidance",
    "Scope Sensitivity",
    "Evidence-Based Capacity Attribution",
    "Cautious Impact Consideration",
    "Actionability",
    "Contextual Welfare Salience",
    "Epistemic Humility",
    "Trade-Off Transparency",
    "Novel Entity Precaution",
    "Control Questions",
]
DISPLAY_BY_CANON = {_canon(n): n for n in DISPLAY_NAMES}


def load_metrics(eval_path: Path) -> tuple[float, dict[str, float]]:
    """Return (overall_mean, {canonical_display_name: score}).

    Name canonicalization handles case/punctuation drift between the upstream
    AHB task definition versions (e.g. 'Trade-Off' vs 'Trade-off Transparency').
    """
    out = subprocess.run(
        [str(INSPECT), "log", "dump", str(eval_path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    scores_block = data["results"]["scores"][0]
    metrics = scores_block.get("metrics", {})
    if isinstance(metrics, dict):
        metrics = list(metrics.values())

    overall = None
    per_dim = {}
    for m in metrics:
        name = m["name"]
        val = m["value"]
        if name == "inspect_evals/overall_mean":
            overall = val
        elif name == "inspect_evals/dimension_normalized_avg":
            continue  # skip the alt aggregate
        else:
            display = DISPLAY_BY_CANON.get(_canon(name), name)
            per_dim[display] = val
    return overall, per_dim


def plot_headline(runs):
    """3-bar TL;DR chart of overall_means."""
    labels = [l.replace("\n", " ") for l in runs.keys()]
    overalls = [v[0] for v in runs.values()]
    colors = ["#888888", "#2E86AB", "#E63946"]
    short = [
        "1B base\n(no fine-tune)",
        "1B + our CAI SFT\n(VeylanSolmira/Base1b_constitutionfinetune_v1)",
        "8B + mid-train + Alpaca\n(Jasmine's comparator)",
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(short))
    bars = ax.bar(x, overalls, color=colors, alpha=0.9, width=0.55)

    for bar, val in zip(bars, overalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")

    # Δ annotations between adjacent bars
    delta_lift = overalls[1] - overalls[0]
    delta_gap = overalls[2] - overalls[1]
    ax.annotate(f"+{delta_lift:.3f}\n(+{100*delta_lift/overalls[0]:.0f}% relative)",
                xy=(0.5, max(overalls[0], overalls[1]) + 0.04),
                ha="center", fontsize=10, color="#2E86AB", fontweight="bold")
    ax.annotate(f"+{delta_gap:.3f}",
                xy=(1.5, max(overalls[1], overalls[2]) + 0.04),
                ha="center", fontsize=10, color="#666", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=10)
    ax.set_ylabel("AHB overall_mean (0–1)", fontsize=11)
    ax.set_ylim(0, 0.50)
    ax.set_title(
        "AHB headline result: 1B simplified-CAI lifts above base; gap to 8B mid-train remains",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = REPO / "experiments/cai-1b-replication/figures/ahb-headline-overall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    runs = {label: load_metrics(path) for label, path in EVALS.items()}
    plot_headline(runs)

    # All dimensions, ordered by 8B-comparator score (so the strongest dimensions cluster left)
    ref_label = list(runs.keys())[1]  # 8B comparator
    _, ref_dims = runs[ref_label]
    dim_order = sorted(ref_dims.keys(), key=lambda d: -ref_dims[d])

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(dim_order))
    n_runs = len(runs)
    width = 0.8 / n_runs

    colors = ["#888888", "#2E86AB", "#E63946"]  # gray = base, blue = +CAI, red = 8B mid-train
    for i, (label, (overall, per_dim)) in enumerate(runs.items()):
        vals = [per_dim.get(d, 0.0) for d in dim_order]
        offset = (i - (n_runs - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=f"{label}\n(overall_mean = {overall:.3f})",
               color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(dim_order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("AHB dimension score (0–1)")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title(
        "AHB per-dimension comparison: 1B base, 1B + simplified-CAI SFT, 8B + mid-training\n"
        "(all evaluated on inspect_evals/ahb, sentientfutures/ahb n=114, gemini-2.5-flash-lite grader, epochs=3)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = REPO / "experiments/cai-1b-replication/figures/ahb-comparison-1b-base-cai-vs-8b-midtrain.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    # also dump the numbers as a small markdown table for the call prep
    md_out = REPO / "experiments/cai-1b-replication/figures/ahb-comparison-1b-base-cai-vs-8b-midtrain.md"
    with md_out.open("w") as f:
        f.write("# AHB per-dimension comparison\n\n")
        labels = list(runs.keys())
        clean_labels = [l.replace(chr(10), ' ') for l in labels]
        f.write(f"| Dimension | {clean_labels[0]} | {clean_labels[1]} | {clean_labels[2]} |\n")
        f.write("|---|---:|---:|---:|\n")
        for d in dim_order:
            vals = [runs[l][1].get(d, 0.0) for l in labels]
            f.write(f"| {d} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |\n")
        overalls = [runs[l][0] for l in labels]
        f.write(f"| **overall_mean** | **{overalls[0]:.3f}** | **{overalls[1]:.3f}** | **{overalls[2]:.3f}** |\n")
    print(f"Saved: {md_out}")


if __name__ == "__main__":
    main()
