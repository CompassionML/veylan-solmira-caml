"""Train per-dimension probes and analyze their relationships.

Trains separate probes for each of the 5 compassion dimensions from Phase 2,
then analyzes how the directions relate to each other and to the composite.
"""

import logging
import numpy as np
from typing import Dict, List, Optional

from v9.probe_training.activation_extractor import ActivationExtractor, ExtractionResult
from v9.probe_training.probe_trainer import CompassionProbeTrainer, TrainingReport, ProbeResult
from v9.dataset_generation.phase2_sentence_level import DIMENSION_MAP

logger = logging.getLogger(__name__)


class DimensionProbeAnalyzer:
    """Train and analyze per-dimension compassion probes."""

    def __init__(self, trainer: CompassionProbeTrainer, extractor: ActivationExtractor):
        self.trainer = trainer
        self.extractor = extractor

    def train_dimension_probes(
        self,
        target_layer: Optional[int] = None,
        target_pooling: Optional[str] = None,
    ) -> Dict[str, TrainingReport]:
        """Train a probe for each compassion dimension.

        Args:
            target_layer: If set, only train at this layer.
            target_pooling: If set, only use this pooling.

        Returns:
            Dict mapping dimension name -> TrainingReport.
        """
        dimension_reports = {}

        for dim_name, pairs in DIMENSION_MAP.items():
            logger.info("Training probe for dimension: %s (%d pairs)", dim_name, len(pairs))

            # Convert TextPair objects to dicts
            pair_dicts = [
                {
                    "pair_id": p.pair_id,
                    "compassionate_text": p.compassionate_text,
                    "non_compassionate_text": p.non_compassionate_text,
                }
                for p in pairs
            ]

            extraction = self.extractor.extract_from_pairs(pair_dicts)

            if target_layer and target_pooling:
                # Filter to just the target layer/pooling
                filtered = ExtractionResult(
                    activations={target_layer: {target_pooling: extraction.activations[target_layer][target_pooling]}},
                    labels=extraction.labels,
                    pair_ids=extraction.pair_ids,
                    texts=extraction.texts,
                    metadata=extraction.metadata,
                )
                report = self.trainer.train_all_probes(filtered)
            else:
                report = self.trainer.train_all_probes(extraction)

            dimension_reports[dim_name] = report
            logger.info("  %s: best layer=%d pooling=%s auroc=%.3f",
                        dim_name, report.best_layer, report.best_pooling, report.best_auroc)

        return dimension_reports

    @staticmethod
    def analyze_direction_relationships(
        dimension_reports: Dict[str, TrainingReport],
        composite_report: Optional[TrainingReport] = None,
    ) -> Dict:
        """Analyze cosine similarities between dimension directions.

        Returns a dict with:
        - pairwise_similarities: Dict of (dim_a, dim_b) -> cosine_sim
        - composite_similarities: Dict of dim -> cosine_sim with composite (if provided)
        """
        dims = sorted(dimension_reports.keys())
        directions = {d: dimension_reports[d].compassion_direction for d in dims}

        # Pairwise similarities
        pairwise = {}
        logger.info("Direction cosine similarities (dimension vs dimension):")
        logger.info("%-30s %10s", "Pair", "Cosine Sim")
        logger.info("-" * 42)

        for i, dim_a in enumerate(dims):
            for dim_b in dims[i+1:]:
                d_a = directions[dim_a]
                d_b = directions[dim_b]
                cos_sim = float(np.dot(d_a, d_b) / (np.linalg.norm(d_a) * np.linalg.norm(d_b)))
                pairwise[(dim_a, dim_b)] = cos_sim
                logger.info("%-30s %10.3f", f"{dim_a} vs {dim_b}", cos_sim)

        results = {"pairwise_similarities": {f"{a}_vs_{b}": v for (a, b), v in pairwise.items()}}

        # Composite similarities
        if composite_report is not None:
            comp_dir = composite_report.compassion_direction
            composite_sims = {}
            logger.info("\nDirection cosine similarities (dimension vs composite):")
            for dim in dims:
                d = directions[dim]
                cos_sim = float(np.dot(d, comp_dir) / (np.linalg.norm(d) * np.linalg.norm(comp_dir)))
                composite_sims[dim] = cos_sim
                logger.info("  %s: %.3f", dim, cos_sim)
            results["composite_similarities"] = composite_sims

        return results
