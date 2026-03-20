"""Multi-layer activation extraction for probe training.

Extracts hidden states from text pairs at multiple layers with multiple
pooling strategies, and saves to disk for efficient probe training.
"""

import logging
import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from model_handler import ModelHandler
from config import ProbeConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Container for extracted activations from a dataset."""
    # activations[layer][pooling] = np.ndarray of shape (n_samples, hidden_dim)
    activations: Dict[int, Dict[str, np.ndarray]]
    labels: np.ndarray  # 1 = compassionate, 0 = non-compassionate
    pair_ids: List[str]
    texts: List[str]
    metadata: Dict


class ActivationExtractor:
    """Extract multi-layer activations from text pairs for probe training."""

    def __init__(self, config: ProbeConfig, model_handler: Optional[ModelHandler] = None):
        self.config = config

        if model_handler is not None:
            self.model_handler = model_handler
        else:
            logger.info("Loading model: %s", config.model_name)
            self.model_handler = ModelHandler(
                config.model_name,
                hf_token=config.hf_token,
                quantization=config.quantization,
            )

    def extract_from_pairs(
        self,
        pairs: List[Dict],
        layers: Optional[List[int]] = None,
        pooling_strategies: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """Extract activations from a list of text pairs.

        Args:
            pairs: List of dicts with 'compassionate_text' and 'non_compassionate_text' keys.
            layers: Layer indices to extract. Defaults to config.layers.
            pooling_strategies: Pooling methods. Defaults to config.pooling_strategies.

        Returns:
            ExtractionResult with activations for all layer/pooling combinations.
        """
        if layers is None:
            layers = self.config.layers
        if pooling_strategies is None:
            pooling_strategies = self.config.pooling_strategies

        # Build flat text list: all compassionate texts, then all non-compassionate
        compassionate_texts = [p["compassionate_text"] for p in pairs]
        non_compassionate_texts = [p["non_compassionate_text"] for p in pairs]
        all_texts = compassionate_texts + non_compassionate_texts

        # Labels: 1 for compassionate, 0 for non-compassionate
        labels = np.array([1] * len(compassionate_texts) + [0] * len(non_compassionate_texts))

        pair_ids = (
            [p.get("pair_id", f"pair_{i}") + "_pos" for i, p in enumerate(pairs)]
            + [p.get("pair_id", f"pair_{i}") + "_neg" for i, p in enumerate(pairs)]
        )

        logger.info("Extracting activations for %d texts (%d pairs) across %d layers x %d pooling strategies",
                     len(all_texts), len(pairs), len(layers), len(pooling_strategies))

        activations = {}
        for pooling in pooling_strategies:
            logger.info("  Pooling: %s", pooling)
            layer_acts = self.model_handler.encode_text(
                all_texts,
                layers=layers,
                batch_size=self.config.batch_size,
                pooling=pooling,
            )

            for layer_idx, tensor in layer_acts.items():
                if layer_idx not in activations:
                    activations[layer_idx] = {}
                activations[layer_idx][pooling] = tensor.numpy()

        metadata = {
            "model_name": self.config.model_name,
            "n_pairs": len(pairs),
            "n_texts": len(all_texts),
            "layers": layers,
            "pooling_strategies": pooling_strategies,
            "hidden_dim": activations[layers[0]][pooling_strategies[0]].shape[1],
        }

        logger.info("Extraction complete. Hidden dim: %d", metadata["hidden_dim"])

        return ExtractionResult(
            activations=activations,
            labels=labels,
            pair_ids=pair_ids,
            texts=all_texts,
            metadata=metadata,
        )

    def save_extractions(self, result: ExtractionResult, output_path: str):
        """Save extraction results to disk."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        save_data = {
            "activations": {
                str(layer): {
                    pooling: acts.tolist()
                    for pooling, acts in pooling_acts.items()
                }
                for layer, pooling_acts in result.activations.items()
            },
            "labels": result.labels.tolist(),
            "pair_ids": result.pair_ids,
            "metadata": result.metadata,
        }

        # Use numpy format for efficiency
        np_path = output_path.replace(".json", ".npz")

        arrays = {"labels": result.labels}
        for layer, pooling_acts in result.activations.items():
            for pooling, acts in pooling_acts.items():
                arrays[f"layer{layer}_{pooling}"] = acts

        np.savez_compressed(np_path, **arrays)

        # Save metadata separately as JSON
        meta_path = output_path.replace(".json", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "pair_ids": result.pair_ids,
                "texts": result.texts,
                "metadata": result.metadata,
            }, f, indent=2)

        logger.info("Saved activations to %s and metadata to %s", np_path, meta_path)

    @staticmethod
    def load_extractions(npz_path: str) -> ExtractionResult:
        """Load extraction results from disk."""
        data = np.load(npz_path)
        meta_path = npz_path.replace(".npz", "_meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        activations = {}
        for key in data.files:
            if key == "labels":
                continue
            # Parse "layer16_mean" -> layer=16, pooling="mean"
            parts = key.split("_", 1)
            layer = int(parts[0].replace("layer", ""))
            pooling = parts[1]
            if layer not in activations:
                activations[layer] = {}
            activations[layer][pooling] = data[key]

        return ExtractionResult(
            activations=activations,
            labels=data["labels"],
            pair_ids=meta["pair_ids"],
            texts=meta.get("texts", []),
            metadata=meta["metadata"],
        )
