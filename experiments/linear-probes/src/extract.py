"""
Activation extraction for compassion probes.

Usage:
    python extract.py \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --pairs data/contrastive_pairs/moral_consideration.jsonl \
        --layers 16 20 24 28 \
        --output outputs/activations/
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm


def load_pairs(path: str) -> list[dict]:
    """Load contrastive pairs from JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def extract_response_activation(model, tokenizer, conversation: list[dict], layer: int) -> torch.Tensor:
    """
    Extract mean-pooled activation for the assistant response.

    Returns:
        torch.Tensor: (hidden_dim,) activation vector
    """
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model(tokens, output_hidden_states=True)
        activations = outputs.hidden_states[layer]  # (1, seq, hidden)

    # Use last 50% of tokens as response approximation
    seq_len = activations.shape[1]
    response_start = seq_len // 2
    response_acts = activations[0, response_start:, :]

    return response_acts.mean(dim=0).cpu()


def extract_pair_activations(model, tokenizer, pair: dict, layers: list[int]) -> dict:
    """Extract activations for both responses in a contrastive pair."""
    results = {"compassionate": {}, "non_compassionate": {}}

    conv_comp = [
        {"role": "user", "content": pair["question"]},
        {"role": "assistant", "content": pair["compassionate_response"]}
    ]
    conv_non = [
        {"role": "user", "content": pair["question"]},
        {"role": "assistant", "content": pair["non_compassionate_response"]}
    ]

    for layer in layers:
        results["compassionate"][layer] = extract_response_activation(
            model, tokenizer, conv_comp, layer
        )
        results["non_compassionate"][layer] = extract_response_activation(
            model, tokenizer, conv_non, layer
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract activations for compassion probes")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--pairs", required=True, help="Path to contrastive pairs JSONL")
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 20, 24, 28])
    parser.add_argument("--output", required=True, help="Output directory")
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

    # Load pairs
    pairs = load_pairs(args.pairs)
    print(f"Loaded {len(pairs)} contrastive pairs")

    # Extract activations
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_activations = []
    for i, pair in enumerate(tqdm(pairs, desc="Extracting")):
        acts = extract_pair_activations(model, tokenizer, pair, args.layers)
        acts["pair_idx"] = i
        acts["scenario"] = pair.get("scenario", "")
        all_activations.append(acts)

    # Save
    output_path = output_dir / f"activations_layers{'_'.join(map(str, args.layers))}.pt"
    torch.save(all_activations, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
