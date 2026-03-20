"""Logistic regression probe training with cross-validation.

v9 changes from v8:
1. Sample-weight balancing so each phase contributes equally
2. Percentile-based scoring instead of sigmoid (fixes dynamic range)
"""

import logging
import json
import os
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field, asdict

import warnings
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")

from v9.probe_training.activation_extractor import ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Results from training a single probe at one layer/pooling."""
    layer: int
    pooling: str
    accuracy: float
    auroc: float
    best_C: float
    shuffle_accuracy: float
    direction: np.ndarray = field(repr=False)
    scaler_mean: np.ndarray = field(repr=False)
    scaler_scale: np.ndarray = field(repr=False)


@dataclass
class TrainingReport:
    """Complete training report across all layer/pooling combos."""
    results: List[ProbeResult]
    best_layer: int
    best_pooling: str
    best_auroc: float
    best_accuracy: float
    compassion_direction: np.ndarray = field(repr=False)


class CompassionProbeTrainer:
    """Train logistic regression probes to find the compassion direction."""

    def __init__(
        self,
        cv_folds: int = 5,
        random_seed: int = 42,
        C_values: Optional[np.ndarray] = None,
    ):
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.C_values = C_values if C_values is not None else np.logspace(-4, 4, 20)

    def train_all_probes(
        self,
        extraction: ExtractionResult,
        sample_weights: Optional[np.ndarray] = None,
    ) -> TrainingReport:
        """Train probes at every layer/pooling combination."""
        results = []

        layers = sorted(extraction.activations.keys())
        poolings = sorted(next(iter(extraction.activations.values())).keys())

        logger.info("Training probes: %d layers x %d poolings = %d combos",
                     len(layers), len(poolings), len(layers) * len(poolings))

        for layer in layers:
            for pooling in poolings:
                X = extraction.activations[layer][pooling]
                y = extraction.labels

                result = self._train_single_probe(X, y, layer, pooling, sample_weights)
                results.append(result)

                logger.info("  Layer %2d / %-5s: acc=%.3f auroc=%.3f C=%.4f shuffle_acc=%.3f",
                            layer, pooling, result.accuracy, result.auroc,
                            result.best_C, result.shuffle_accuracy)

        # Select best by AUROC, tie-break by preferring middle layers
        results.sort(key=lambda r: (r.auroc, -abs(r.layer - 16)), reverse=True)
        best = results[0]

        logger.info("Best probe: layer=%d pooling=%s auroc=%.3f acc=%.3f",
                     best.layer, best.pooling, best.auroc, best.accuracy)

        return TrainingReport(
            results=results,
            best_layer=best.layer,
            best_pooling=best.pooling,
            best_auroc=best.auroc,
            best_accuracy=best.accuracy,
            compassion_direction=best.direction,
        )

    def _train_single_probe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        layer: int,
        pooling: str,
        sample_weights: Optional[np.ndarray] = None,
    ) -> ProbeResult:
        """Train and evaluate a single probe."""
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegressionCV(
                Cs=self.C_values,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed),
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                scoring="roc_auc",
                random_state=self.random_seed,
            ))
        ])

        # Fit with optional sample weights
        fit_params = {}
        if sample_weights is not None:
            fit_params["clf__sample_weight"] = sample_weights
        pipeline.fit(X, y, **fit_params)

        clf = pipeline.named_steps["clf"]
        scaler = pipeline.named_steps["scaler"]

        y_pred = pipeline.predict(X)
        y_prob = pipeline.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, y_pred)
        auroc = roc_auc_score(y, y_prob)

        # Extract direction vector (in original feature space)
        direction_scaled = clf.coef_[0]
        direction = direction_scaled / scaler.scale_
        direction = direction / np.linalg.norm(direction)

        # Shuffle test
        rng = np.random.RandomState(self.random_seed + 999)
        y_shuffled = rng.permutation(y)
        shuffle_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegressionCV(
                Cs=self.C_values,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed),
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                random_state=self.random_seed,
            ))
        ])
        shuffle_pipeline.fit(X, y_shuffled)
        shuffle_accuracy = accuracy_score(y_shuffled, shuffle_pipeline.predict(X))

        return ProbeResult(
            layer=layer,
            pooling=pooling,
            accuracy=accuracy,
            auroc=auroc,
            best_C=float(clf.C_[0]),
            shuffle_accuracy=shuffle_accuracy,
            direction=direction,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
        )

    def train_on_phases(
        self,
        extractions: Dict[str, ExtractionResult],
        target_layer: Optional[int] = None,
        target_pooling: Optional[str] = None,
    ) -> TrainingReport:
        """Train a composite probe on multiple phases with balanced weighting.

        v9: Each phase contributes equally to the loss via sample weights.
        """
        first = next(iter(extractions.values()))
        layers = sorted(first.activations.keys())
        poolings = sorted(next(iter(first.activations.values())).keys())

        if target_layer:
            layers = [target_layer]
        if target_pooling:
            poolings = [target_pooling]

        # Stack activations and labels
        merged_activations = {}
        merged_labels = []

        for layer in layers:
            merged_activations[layer] = {}
            for pooling in poolings:
                arrays = []
                for phase_name, ext in extractions.items():
                    arrays.append(ext.activations[layer][pooling])
                merged_activations[layer][pooling] = np.concatenate(arrays, axis=0)

        for phase_name, ext in extractions.items():
            merged_labels.append(ext.labels)
        merged_labels = np.concatenate(merged_labels)

        # Compute per-phase sample weights so each phase contributes equally
        phase_sizes = {name: len(ext.labels) for name, ext in extractions.items()}
        total_samples = sum(phase_sizes.values())
        n_phases = len(phase_sizes)
        target_per_phase = total_samples / n_phases

        sample_weights = np.ones(total_samples)
        offset = 0
        for name, ext in extractions.items():
            n = len(ext.labels)
            phase_weight = target_per_phase / n
            sample_weights[offset:offset + n] = phase_weight
            offset += n

        logger.info("Training composite probe on %d total samples from %d phases (balanced weights)",
                     len(merged_labels), len(extractions))
        for name, ext in extractions.items():
            w = target_per_phase / len(ext.labels)
            logger.info("  %s: %d samples, weight=%.3f", name, len(ext.labels), w)

        merged_extraction = ExtractionResult(
            activations=merged_activations,
            labels=merged_labels,
            pair_ids=[],
            texts=[],
            metadata={"phases": list(extractions.keys())},
        )

        return self.train_all_probes(merged_extraction, sample_weights=sample_weights)

    def save_report(self, report: TrainingReport, output_path: str):
        """Save training report to disk."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        direction_path = output_path.replace(".json", "_direction.npy")
        np.save(direction_path, report.compassion_direction)

        report_data = {
            "best_layer": report.best_layer,
            "best_pooling": report.best_pooling,
            "best_auroc": report.best_auroc,
            "best_accuracy": report.best_accuracy,
            "all_results": [
                {
                    "layer": r.layer,
                    "pooling": r.pooling,
                    "accuracy": r.accuracy,
                    "auroc": r.auroc,
                    "best_C": r.best_C,
                    "shuffle_accuracy": r.shuffle_accuracy,
                }
                for r in report.results
            ]
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info("Saved report to %s and direction to %s", output_path, direction_path)


class CompassionScorer:
    """Score text using percentile-based calibration against training distribution.

    v9 FIX: Replaces sigmoid scoring which compressed everything to 58-59.
    Uses linear interpolation between training distribution percentiles.
    """

    def __init__(
        self,
        direction: np.ndarray,
        p5_negative: float,
        p95_positive: float,
    ):
        self.direction = direction
        self.p5 = p5_negative      # 5th percentile of negative-class projections
        self.p95 = p95_positive     # 95th percentile of positive-class projections

    def score(self, activations: np.ndarray) -> np.ndarray:
        """Score activations on 0-100 compassion scale.

        Linear interpolation: p5 → 0, p95 → 100, clipped to [0, 100].
        """
        projections = activations @ self.direction
        scores = (projections - self.p5) / (self.p95 - self.p5) * 100
        return np.clip(scores, 0, 100)

    @classmethod
    def from_training_data(
        cls,
        direction: np.ndarray,
        train_activations: np.ndarray,
        train_labels: np.ndarray,
    ) -> 'CompassionScorer':
        """Create scorer calibrated against training data distribution."""
        projections = train_activations @ direction

        pos_projections = projections[train_labels == 1]
        neg_projections = projections[train_labels == 0]

        # Use median of each class as anchors — robust to outliers
        # but guarantees meaningful spread between the two anchor points
        p5_neg = float(np.median(neg_projections))
        p95_pos = float(np.median(pos_projections))

        # Fallback: if medians are too close, use full min/max
        if abs(p95_pos - p5_neg) < 0.01:
            p5_neg = float(np.min(neg_projections))
            p95_pos = float(np.max(pos_projections))

        logger.info("Calibration: p5_neg=%.4f, p95_pos=%.4f, range=%.4f",
                     p5_neg, p95_pos, p95_pos - p5_neg)

        return cls(direction=direction, p5_negative=p5_neg, p95_positive=p95_pos)

    @classmethod
    def from_probe_result(cls, result: ProbeResult, pipeline=None) -> 'CompassionScorer':
        """Create scorer from a ProbeResult (requires calibration data)."""
        raise NotImplementedError(
            "v9 CompassionScorer requires training data for calibration. "
            "Use CompassionScorer.from_training_data() instead."
        )
