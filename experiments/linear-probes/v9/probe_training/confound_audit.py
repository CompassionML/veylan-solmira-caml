"""Confound detection for compassion probes.

v9 changes from v8:
1. Added vocabulary confound test (word-identity detector)
2. Lowered similarity threshold from 0.5 to 0.3
3. With 2x2 crossing, word-identity confound should be ~0.0
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from v9.probe_training.probe_trainer import CompassionProbeTrainer, TrainingReport

logger = logging.getLogger(__name__)


@dataclass
class ConfoundResult:
    """Result of a single confound audit."""
    confound_name: str
    confound_accuracy: float
    confound_auroc: float
    cosine_similarity_with_compassion: float
    is_confounded: bool


class ConfoundAuditor:
    """Audit compassion probes for confounding features."""

    def __init__(
        self,
        trainer: CompassionProbeTrainer,
        similarity_threshold: float = 0.3,  # v9: stricter than v8's 0.5
    ):
        self.trainer = trainer
        self.threshold = similarity_threshold

    def audit_word_identity(
        self,
        activations: np.ndarray,
        texts: List[str],
        compassion_direction: np.ndarray,
        layer: int,
        pooling: str,
    ) -> ConfoundResult:
        """v9 NEW: Check if probe is detecting animal-name vs meat-term presence.

        With 2x2 crossing, both animal names and meat terms appear in both
        label classes. This confound test should show cosine_sim ≈ 0.0.
        """
        animal_names = [
            "pig", "cow", "calf", "chicken", "sheep", "deer",
            "lamb", "hen", "sow", "piglet", "heifer", "chick",
            "bull", "ewe", "rabbit", "fish",
        ]

        word_identity_labels = np.zeros(len(texts), dtype=int)
        for i, text in enumerate(texts):
            text_lower = text.lower()
            if any(f" {name} " in f" {text_lower} " or
                   text_lower.startswith(f"{name} ") or
                   text_lower.endswith(f" {name}") or
                   text_lower.endswith(f" {name}.")
                   for name in animal_names):
                word_identity_labels[i] = 1

        return self._run_confound_test(
            "word_identity",
            activations, word_identity_labels,
            compassion_direction, layer, pooling,
        )

    def audit_euphemism_presence(
        self,
        activations: np.ndarray,
        texts: List[str],
        compassion_direction: np.ndarray,
        layer: int,
        pooling: str,
    ) -> ConfoundResult:
        """Check if probe detects euphemism/commodity terms."""
        commodity_terms = [
            "pork", "beef", "veal", "poultry", "mutton", "venison",
            "game", "seafood", "layer", "breeding stock", "weaner",
            "replacement stock", "day-old", "sire unit", "breeding ewe",
        ]

        euphemism_labels = np.zeros(len(texts), dtype=int)
        for i, text in enumerate(texts):
            text_lower = text.lower()
            if any(term in text_lower for term in commodity_terms):
                euphemism_labels[i] = 1

        return self._run_confound_test(
            "euphemism_presence",
            activations, euphemism_labels,
            compassion_direction, layer, pooling,
        )

    def audit_text_length(
        self,
        activations: np.ndarray,
        texts: List[str],
        compassion_direction: np.ndarray,
        layer: int,
        pooling: str,
    ) -> ConfoundResult:
        """Check if probe is detecting text length."""
        lengths = np.array([len(t.split()) for t in texts])
        median_length = np.median(lengths)
        length_labels = (lengths > median_length).astype(int)

        return self._run_confound_test(
            "text_length",
            activations, length_labels,
            compassion_direction, layer, pooling,
        )

    def audit_sentiment(
        self,
        activations: np.ndarray,
        texts: List[str],
        compassion_direction: np.ndarray,
        layer: int,
        pooling: str,
    ) -> ConfoundResult:
        """Check if probe is detecting emotional/sentiment words."""
        emotion_words = {
            "suffering", "pain", "fear", "distress", "terror", "grief",
            "cruel", "harm", "trauma", "agony", "anguish", "misery",
            "torture", "abuse", "neglect", "heartbreaking", "devastating",
            "joy", "happy", "love", "care", "gentle", "tender",
        }

        sentiment_labels = np.zeros(len(texts), dtype=int)
        for i, text in enumerate(texts):
            words = set(text.lower().split())
            if len(words & emotion_words) >= 2:
                sentiment_labels[i] = 1

        return self._run_confound_test(
            "sentiment_loading",
            activations, sentiment_labels,
            compassion_direction, layer, pooling,
        )

    def _run_confound_test(
        self,
        confound_name: str,
        activations: np.ndarray,
        confound_labels: np.ndarray,
        compassion_direction: np.ndarray,
        layer: int,
        pooling: str,
    ) -> ConfoundResult:
        """Run a confound detection test."""
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.pipeline import Pipeline

        unique = np.unique(confound_labels)
        if len(unique) < 2:
            logger.warning("Confound %s has only one class — skipping", confound_name)
            return ConfoundResult(
                confound_name=confound_name,
                confound_accuracy=0.5,
                confound_auroc=0.5,
                cosine_similarity_with_compassion=0.0,
                is_confounded=False,
            )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegressionCV(
                Cs=np.logspace(-4, 4, 10),
                cv=StratifiedKFold(n_splits=max(2, min(5, min(np.bincount(confound_labels)))),
                                   shuffle=True, random_state=42),
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            ))
        ])

        pipeline.fit(activations, confound_labels)

        y_pred = pipeline.predict(activations)
        y_prob = pipeline.predict_proba(activations)[:, 1]

        accuracy = accuracy_score(confound_labels, y_pred)
        auroc = roc_auc_score(confound_labels, y_prob)

        clf = pipeline.named_steps["clf"]
        scaler = pipeline.named_steps["scaler"]
        confound_direction = clf.coef_[0] / scaler.scale_
        confound_direction = confound_direction / np.linalg.norm(confound_direction)

        cos_sim = float(np.dot(confound_direction, compassion_direction) /
                       (np.linalg.norm(confound_direction) * np.linalg.norm(compassion_direction)))

        is_confounded = abs(cos_sim) > self.threshold

        logger.info("Confound audit [%s]: acc=%.3f auroc=%.3f cos_sim=%.3f %s",
                     confound_name, accuracy, auroc, cos_sim,
                     "CONFOUNDED!" if is_confounded else "ok")

        return ConfoundResult(
            confound_name=confound_name,
            confound_accuracy=accuracy,
            confound_auroc=auroc,
            cosine_similarity_with_compassion=cos_sim,
            is_confounded=is_confounded,
        )

    def run_full_audit(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        texts: List[str],
        compassion_direction: np.ndarray,
        layer: int,
        pooling: str,
        phase: str = "all",
    ) -> List[ConfoundResult]:
        """Run all applicable confound tests."""
        results = []

        # v9 NEW: Word identity confound (most important for 2x2 validation)
        results.append(self.audit_word_identity(
            activations, texts, compassion_direction, layer, pooling))

        if phase in ("phase1", "all"):
            results.append(self.audit_euphemism_presence(
                activations, texts, compassion_direction, layer, pooling))

        results.append(self.audit_sentiment(
            activations, texts, compassion_direction, layer, pooling))

        results.append(self.audit_text_length(
            activations, texts, compassion_direction, layer, pooling))

        confounded = [r for r in results if r.is_confounded]
        if confounded:
            logger.warning("CONFOUND WARNING: %d confounds detected: %s (threshold=%.2f)",
                          len(confounded), [r.confound_name for r in confounded], self.threshold)
        else:
            logger.info("No confounds detected (threshold=%.2f)", self.threshold)

        return results
