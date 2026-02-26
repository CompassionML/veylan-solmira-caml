#!/usr/bin/env python3
"""
Inventory CaML HuggingFace models and identify base→fine-tuned pairs.

Usage:
    python inventory-models.py
    python inventory-models.py --output docs/model-inventory.md
"""

import argparse
import re
from collections import defaultdict
from huggingface_hub import HfApi

SOURCE_ORG = "CompassioninMachineLearning"

def parse_model_name(name: str) -> dict:
    """Parse model name to extract base model and training stages."""
    info = {
        "name": name,
        "base": None,
        "stages": [],
        "variants": [],
    }

    # Common base model patterns
    base_patterns = [
        (r"(llama|Llama)", "llama"),
        (r"(qwen|Qwen)", "qwen"),
        (r"(mistral|Mistral)", "mistral"),
        (r"(gemma|Gemma)", "gemma"),
    ]

    for pattern, base in base_patterns:
        if re.search(pattern, name, re.IGNORECASE):
            info["base"] = base
            break

    # Training stage patterns (order matters)
    stage_patterns = [
        (r"pretraining", "pretraining"),
        (r"Instruct", "instruct"),
        (r"Base", "base"),
        (r"_plus(\d+k?v?\d*)", "sdf"),  # synthetic document finetuning
        (r"alpaca", "alpaca"),
        (r"medai", "medai"),
    ]

    for pattern, stage in stage_patterns:
        if re.search(pattern, name):
            info["stages"].append(stage)

    # Variant patterns
    variant_patterns = [
        (r"UrbanDensity", "urban_density"),
        (r"16bit", "16bit"),
        (r"reworded", "reworded"),
        (r"shot", "shot"),
        (r"dilution", "dilution"),
    ]

    for pattern, variant in variant_patterns:
        if re.search(pattern, name):
            info["variants"].append(variant)

    return info


def main():
    parser = argparse.ArgumentParser(description="Inventory CaML models")
    parser.add_argument("--output", "-o", help="Output markdown file")
    args = parser.parse_args()

    api = HfApi()

    print(f"Fetching models from {SOURCE_ORG}...")
    models = list(api.list_models(author=SOURCE_ORG))
    print(f"Found {len(models)} models\n")

    # Parse all models
    parsed = []
    for m in models:
        name = m.id.split("/")[-1]
        info = parse_model_name(name)
        info["full_id"] = m.id
        info["downloads"] = getattr(m, "downloads", 0)
        info["likes"] = getattr(m, "likes", 0)
        parsed.append(info)

    # Group by base model
    by_base = defaultdict(list)
    for p in parsed:
        by_base[p["base"] or "unknown"].append(p)

    # Generate report
    lines = []
    lines.append("# CaML Model Inventory\n")
    lines.append(f"**Organization:** {SOURCE_ORG}")
    lines.append(f"**Total models:** {len(models)}\n")

    lines.append("## Summary by Base Model\n")
    lines.append("| Base | Count |")
    lines.append("|------|-------|")
    for base in sorted(by_base.keys()):
        lines.append(f"| {base} | {len(by_base[base])} |")
    lines.append("")

    lines.append("## Models by Base\n")

    for base in sorted(by_base.keys()):
        lines.append(f"### {base.title()}\n")

        # Sort by training stages (simpler first)
        models_sorted = sorted(by_base[base], key=lambda x: (len(x["stages"]), x["name"]))

        for m in models_sorted:
            stages = " → ".join(m["stages"]) if m["stages"] else "unknown"
            variants = ", ".join(m["variants"]) if m["variants"] else ""

            line = f"- **{m['name']}**"
            if stages != "unknown":
                line += f" ({stages})"
            if variants:
                line += f" [{variants}]"
            lines.append(line)

        lines.append("")

    # Identify likely base→fine-tuned pairs
    lines.append("## Potential Base → Fine-tuned Pairs\n")
    lines.append("Based on naming patterns, these appear to be progression pairs:\n")

    # Look for patterns like: BaseX → BaseX_plus...
    pairs_found = []
    for base in by_base:
        models_in_base = by_base[base]
        names = [m["name"] for m in models_in_base]

        for m in models_in_base:
            # If this is a "plus" variant, find its base
            if "sdf" in m["stages"]:
                # Try to find the base version
                base_pattern = m["name"].split("_plus")[0]
                for other in models_in_base:
                    if other["name"] == base_pattern or other["name"] == f"pretraining{base_pattern}":
                        pairs_found.append((other["name"], m["name"]))

    if pairs_found:
        lines.append("| Base Model | Fine-tuned Variant |")
        lines.append("|------------|-------------------|")
        for base_m, ft_m in sorted(set(pairs_found)):
            lines.append(f"| {base_m} | {ft_m} |")
    else:
        lines.append("(No clear pairs identified from naming patterns)")

    lines.append("")

    # Full list
    lines.append("## Full Model List\n")
    lines.append("| Model | Base | Stages | Variants |")
    lines.append("|-------|------|--------|----------|")
    for p in sorted(parsed, key=lambda x: x["name"]):
        stages = ", ".join(p["stages"]) if p["stages"] else "-"
        variants = ", ".join(p["variants"]) if p["variants"] else "-"
        lines.append(f"| {p['name']} | {p['base'] or '-'} | {stages} | {variants} |")

    report = "\n".join(lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
