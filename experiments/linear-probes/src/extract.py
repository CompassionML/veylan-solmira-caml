"""
Activation extraction for compassion probes.

Extracts hidden state activations and mean-pools over the exact response tokens
(not the user prompt). This provides a single vector per response for probing.

Usage:
    # Standard mode (multiple layers)
    python extract.py \
        --model /data/uds-grave-seasoned-brownie-251009 \
        --pairs data/contrastive-pairs/usable_pairs_deduped.jsonl \
        --layers 16 20 24 28 \
        --output outputs/activations/

    # Memory-efficient mode (one layer at a time, batched)
    python extract.py \
        --model /data/uds-grave-seasoned-brownie-251009 \
        --pairs data/contrastive-pairs/usable_pairs_deduped.jsonl \
        --layers 24 \
        --output outputs/activations/ \
        --batch-size 8 \
        --memory-efficient
"""

import argparse
import json
import time
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Optional


def load_pairs(path: str) -> list[dict]:
    """Load contrastive pairs from JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


class ActivationCapture:
    """Hook-based activation capture for memory efficiency."""

    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.activation = None
        self.hook = None

    def hook_fn(self, module, input, output):
        # output is tuple, first element is hidden states
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


def extract_activation_with_hook(
    model,
    tokenizer,
    text: str,
    layer: int,
    response_start_idx: int = 0
) -> torch.Tensor:
    """
    Extract mean-pooled activation using hooks (memory efficient).
    Only captures the specific layer requested.

    Args:
        response_start_idx: Token index where the response begins.
                           Activations are mean-pooled from this index onward.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    capture = ActivationCapture(layer)
    capture.register(model)

    try:
        with torch.no_grad():
            model(**tokens)

        activation = capture.get_activation()  # (1, seq, hidden)
        response_acts = activation[0, response_start_idx:, :]

        return response_acts.mean(dim=0).cpu()
    finally:
        capture.remove()


def extract_activation_standard(
    model,
    tokenizer,
    text: str,
    layer: int,
    response_start_idx: int = 0
) -> torch.Tensor:
    """
    Extract activation using output_hidden_states (simpler but uses more memory).

    Args:
        response_start_idx: Token index where the response begins.
                           Activations are mean-pooled from this index onward.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        activation = outputs.hidden_states[layer]  # (1, seq, hidden)

    response_acts = activation[0, response_start_idx:, :]

    return response_acts.mean(dim=0).cpu()


def format_conversation(pair: dict, response_type: str, tokenizer) -> tuple[str, int]:
    """
    Format a conversation for the model and compute response start index.

    Returns:
        tuple: (formatted_text, response_start_idx)
               response_start_idx is the token index where the assistant response begins
    """
    # Support both 'prompt' and 'question' keys
    user_content = pair.get("prompt") or pair.get("question")

    # Get the prompt-only portion to find where response starts
    prompt_only = [
        {"role": "user", "content": user_content},
    ]
    prompt_text = tokenizer.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer(prompt_text, truncation=True, max_length=1024)
    response_start_idx = len(prompt_tokens["input_ids"])

    # Get the full conversation
    conversation = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": pair[response_type]}
    ]
    full_text = tokenizer.apply_chat_template(conversation, tokenize=False)

    return full_text, response_start_idx


def extract_single_layer(
    model,
    tokenizer,
    pairs: list[dict],
    layer: int,
    memory_efficient: bool = True,
    batch_size: int = 1,
    show_progress: bool = True
) -> dict:
    """
    Extract activations for a single layer across all pairs.

    Returns:
        dict with 'compassionate' and 'non_compassionate' tensors of shape (n_pairs, hidden_dim)
    """
    extract_fn = extract_activation_with_hook if memory_efficient else extract_activation_standard

    compassionate_acts = []
    non_compassionate_acts = []

    iterator = tqdm(pairs, desc=f"Layer {layer}", disable=not show_progress)

    for i, pair in enumerate(iterator):
        # Extract compassionate response activation
        text_comp, response_start_comp = format_conversation(pair, "compassionate_response", tokenizer)
        act_comp = extract_fn(model, tokenizer, text_comp, layer, response_start_idx=response_start_comp)
        compassionate_acts.append(act_comp)

        # Extract non-compassionate response activation
        text_non, response_start_non = format_conversation(pair, "non_compassionate_response", tokenizer)
        act_non = extract_fn(model, tokenizer, text_non, layer, response_start_idx=response_start_non)
        non_compassionate_acts.append(act_non)

        # Clear cache periodically
        if memory_efficient and (i + 1) % batch_size == 0:
            torch.cuda.empty_cache()

    return {
        "compassionate": torch.stack(compassionate_acts),
        "non_compassionate": torch.stack(non_compassionate_acts)
    }


def estimate_time(n_pairs: int, n_layers: int, time_per_forward: float = 0.3) -> str:
    """Estimate total extraction time."""
    # 2 forward passes per pair (compassionate + non-compassionate)
    total_forwards = n_pairs * 2 * n_layers
    total_seconds = total_forwards * time_per_forward

    if total_seconds < 60:
        return f"{total_seconds:.0f} seconds"
    elif total_seconds < 3600:
        return f"{total_seconds / 60:.1f} minutes"
    else:
        return f"{total_seconds / 3600:.1f} hours"


def main():
    parser = argparse.ArgumentParser(description="Extract activations for compassion probes")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--pairs", required=True, help="Path to contrastive pairs JSONL")
    parser.add_argument("--layers", nargs="+", type=int, default=[24],
                        help="Layers to extract (default: 24, which is 75%% of 32)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Pairs to process before clearing CUDA cache")
    parser.add_argument("--memory-efficient", action="store_true",
                        help="Use hook-based extraction (less memory, same speed)")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Only estimate time, don't run extraction")
    args = parser.parse_args()

    # Load pairs first (to get count for estimation)
    pairs = load_pairs(args.pairs)
    print(f"Loaded {len(pairs)} contrastive pairs")
    print(f"Layers to extract: {args.layers}")
    print(f"Estimated time: {estimate_time(len(pairs), len(args.layers))}")

    if args.estimate_only:
        return

    # Load model
    print(f"\nLoading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Check for flash attention
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

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract activations layer by layer
    start_time = time.time()

    for layer in args.layers:
        print(f"\nExtracting layer {layer}...")
        layer_start = time.time()

        activations = extract_single_layer(
            model=model,
            tokenizer=tokenizer,
            pairs=pairs,
            layer=layer,
            memory_efficient=args.memory_efficient,
            batch_size=args.batch_size,
            show_progress=True
        )

        # Save this layer's activations
        output_path = output_dir / f"activations_layer_{layer}.pt"
        torch.save({
            "layer": layer,
            "compassionate": activations["compassionate"],
            "non_compassionate": activations["non_compassionate"],
            "n_pairs": len(pairs),
            "model": args.model,
            "pooling": "mean_over_response",  # exact response tokens only
        }, output_path)

        layer_time = time.time() - layer_start
        print(f"Layer {layer}: {layer_time:.1f}s, saved to {output_path}")

        # Clear memory between layers
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"\nTotal extraction time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
