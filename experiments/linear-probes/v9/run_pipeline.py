"""End-to-end pipeline for v9 compassion probe training and evaluation.

v9 changes from v8:
1. Phase 1 uses 2x2 context-crossing (no word-identity shortcut)
2. Composite probe uses sample-weight balancing across phases
3. Percentile-based scoring replaces sigmoid (full dynamic range)
4. Fail-fast: reversed-context accuracy < 0.6 halts pipeline
5. Vocabulary confound test validates 2x2 design worked

Usage:
    python -m v9.run_pipeline --model meta-llama/Llama-3.1-8B --output-dir v9/output
"""

import argparse
import logging
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import ProbeConfig
from model_handler import ModelHandler
from v9.dataset_generation.phase1_word_level import (
    generate_phase1_dataset, get_reversed_context_pairs,
)
from v9.dataset_generation.phase2_sentence_level import (
    generate_phase2_dataset, get_all_phase2_pairs, DIMENSION_MAP,
)
from v9.dataset_generation.phase3_paragraph_level import (
    generate_phase3_dataset, get_all_phase3_pairs,
)
from v9.probe_training.activation_extractor import ActivationExtractor, ExtractionResult
from v9.probe_training.probe_trainer import CompassionProbeTrainer, CompassionScorer
from v9.probe_training.dimension_probes import DimensionProbeAnalyzer
from v9.probe_training.confound_audit import ConfoundAuditor
from v9.evaluation.eval_suite import EvaluationSuite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def pairs_to_dicts(pairs):
    return [
        {
            "pair_id": p.pair_id,
            "compassionate_text": p.compassionate_text,
            "non_compassionate_text": p.non_compassionate_text,
        }
        for p in pairs
    ]


def step_generate_datasets(config: ProbeConfig):
    logger.info("=" * 60)
    logger.info("STEP 1: GENERATING DATASETS")
    logger.info("=" * 60)

    dataset_dir = config.dataset_dir

    logger.info("Phase 1: Word-level 2x2 context-crossing pairs")
    p1_pairs = generate_phase1_dataset(output_dir=dataset_dir)

    logger.info("\nPhase 2: Sentence-level dimension pairs")
    generate_phase2_dataset(output_dir=dataset_dir)

    logger.info("\nPhase 3: Paragraph-level practical compassion")
    generate_phase3_dataset(output_dir=dataset_dir)

    return p1_pairs


def step_extract_activations(config: ProbeConfig, model_handler: ModelHandler):
    logger.info("=" * 60)
    logger.info("STEP 2: EXTRACTING ACTIVATIONS")
    logger.info("=" * 60)

    extractor = ActivationExtractor(config, model_handler=model_handler)
    output_dir = os.path.join(config.output_dir, "extractions")
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1
    logger.info("Extracting Phase 1 activations...")
    p1_pairs = generate_phase1_dataset()
    p1_extraction = extractor.extract_from_pairs(pairs_to_dicts(p1_pairs))
    extractor.save_extractions(p1_extraction, os.path.join(output_dir, "phase1.json"))

    # Phase 1 validation (held-out reversed context)
    logger.info("Extracting Phase 1 validation activations...")
    p1_val = get_reversed_context_pairs()
    p1_val_extraction = extractor.extract_from_pairs(pairs_to_dicts(p1_val))
    extractor.save_extractions(p1_val_extraction, os.path.join(output_dir, "phase1_val.json"))

    # Phase 2
    logger.info("Extracting Phase 2 activations...")
    p2_pairs = get_all_phase2_pairs()
    p2_extraction = extractor.extract_from_pairs(pairs_to_dicts(p2_pairs))
    extractor.save_extractions(p2_extraction, os.path.join(output_dir, "phase2.json"))

    # Phase 3
    logger.info("Extracting Phase 3 activations...")
    p3_pairs = get_all_phase3_pairs()
    p3_extraction = extractor.extract_from_pairs(pairs_to_dicts(p3_pairs))
    extractor.save_extractions(p3_extraction, os.path.join(output_dir, "phase3.json"))

    return {
        "phase1": p1_extraction,
        "phase1_val": p1_val_extraction,
        "phase2": p2_extraction,
        "phase3": p3_extraction,
    }


def step_train_phase1_probes(extractions: dict, config: ProbeConfig):
    """Step 3: Train Phase 1 probes with FAIL-FAST reversed-context check."""
    logger.info("=" * 60)
    logger.info("STEP 3: TRAINING PHASE 1 PROBES (CHECKPOINT)")
    logger.info("=" * 60)

    trainer = CompassionProbeTrainer(
        cv_folds=config.cv_folds,
        random_seed=config.random_seed,
    )

    p1_report = trainer.train_all_probes(extractions["phase1"])
    trainer.save_report(p1_report, os.path.join(config.output_dir, "phase1_report.json"))

    # FAIL-FAST: Validate on reversed-context pairs
    logger.info("Validating on reversed-context pairs...")
    val_ext = extractions["phase1_val"]
    val_X = val_ext.activations[p1_report.best_layer][p1_report.best_pooling]
    val_y = val_ext.labels

    projections = val_X @ p1_report.compassion_direction
    val_preds = (projections > 0).astype(int)
    val_acc = float(np.mean(val_preds == val_y))
    logger.info("Reversed-context validation accuracy: %.3f", val_acc)

    if val_acc < 0.6:
        logger.warning("FAIL-FAST: Reversed-context accuracy (%.3f) < 0.6 — probe is lexical!", val_acc)
        logger.warning("The 2x2 design may not be working. Check Phase 1 data.")
    else:
        logger.info("PASS: Reversed-context accuracy %.3f — probe captures context!", val_acc)

    return p1_report, val_acc


def step_train_dimension_probes(extractions: dict, config: ProbeConfig, model_handler: ModelHandler):
    logger.info("=" * 60)
    logger.info("STEP 4: TRAINING DIMENSION PROBES")
    logger.info("=" * 60)

    trainer = CompassionProbeTrainer(cv_folds=config.cv_folds, random_seed=config.random_seed)
    extractor = ActivationExtractor(config, model_handler=model_handler)
    analyzer = DimensionProbeAnalyzer(trainer, extractor)

    dim_reports = analyzer.train_dimension_probes()

    for dim_name, report in dim_reports.items():
        trainer.save_report(report, os.path.join(config.output_dir, f"dim_{dim_name}_report.json"))

    return dim_reports


def step_train_composite_probe(extractions: dict, config: ProbeConfig):
    """Step 5: Train composite probe WITH sample-weight balancing."""
    logger.info("=" * 60)
    logger.info("STEP 5: TRAINING COMPOSITE PROBE (BALANCED)")
    logger.info("=" * 60)

    trainer = CompassionProbeTrainer(cv_folds=config.cv_folds, random_seed=config.random_seed)

    train_extractions = {
        k: v for k, v in extractions.items() if k in ("phase1", "phase2", "phase3")
    }

    # v9: train_on_phases now applies sample weights automatically
    composite_report = trainer.train_on_phases(train_extractions)
    trainer.save_report(composite_report, os.path.join(config.output_dir, "composite_report.json"))

    logger.info("Composite probe: layer=%d pooling=%s auroc=%.3f acc=%.3f",
                 composite_report.best_layer, composite_report.best_pooling,
                 composite_report.best_auroc, composite_report.best_accuracy)

    return composite_report


def step_calibrate_scorer(extractions: dict, composite_report, config: ProbeConfig):
    """Step 6: Calibrate percentile-based scorer against training distribution."""
    logger.info("=" * 60)
    logger.info("STEP 6: CALIBRATING SCORER")
    logger.info("=" * 60)

    # Get all training activations at the best layer/pooling
    layer = composite_report.best_layer
    pooling = composite_report.best_pooling

    all_acts = []
    all_labels = []
    for phase in ("phase1", "phase2", "phase3"):
        ext = extractions[phase]
        all_acts.append(ext.activations[layer][pooling])
        all_labels.append(ext.labels)

    train_acts = np.concatenate(all_acts, axis=0)
    train_labels = np.concatenate(all_labels)

    scorer = CompassionScorer.from_training_data(
        direction=composite_report.compassion_direction,
        train_activations=train_acts,
        train_labels=train_labels,
    )

    # Save calibration info
    cal_info = {
        "p5_negative": scorer.p5,
        "p95_positive": scorer.p95,
        "range": scorer.p95 - scorer.p5,
    }
    cal_path = os.path.join(config.output_dir, "calibration.json")
    with open(cal_path, "w") as f:
        json.dump(cal_info, f, indent=2)
    logger.info("Saved calibration to %s", cal_path)

    return scorer


def step_confound_audit(extractions: dict, composite_report, config: ProbeConfig):
    logger.info("=" * 60)
    logger.info("STEP 7: CONFOUND AUDIT (threshold=0.3)")
    logger.info("=" * 60)

    trainer = CompassionProbeTrainer(cv_folds=config.cv_folds, random_seed=config.random_seed)
    auditor = ConfoundAuditor(trainer, similarity_threshold=0.3)

    all_results = []
    for phase_name in ("phase1", "phase2", "phase3"):
        ext = extractions[phase_name]
        layer = composite_report.best_layer
        pooling = composite_report.best_pooling
        X = ext.activations[layer][pooling]

        logger.info("Auditing %s...", phase_name)
        results = auditor.run_full_audit(
            activations=X,
            labels=ext.labels,
            texts=ext.texts,
            compassion_direction=composite_report.compassion_direction,
            layer=layer,
            pooling=pooling,
            phase=phase_name,
        )
        all_results.extend(results)

    return all_results


def step_evaluate(composite_report, scorer, config: ProbeConfig, model_handler: ModelHandler):
    """Step 8: Run evaluation with percentile-calibrated scoring."""
    logger.info("=" * 60)
    logger.info("STEP 8: EVALUATION SUITE")
    logger.info("=" * 60)

    direction = composite_report.compassion_direction
    best_layer = composite_report.best_layer
    best_pooling = composite_report.best_pooling

    def score_fn(texts):
        layer_acts = model_handler.encode_text(
            texts,
            layers=[best_layer],
            batch_size=config.batch_size,
            pooling=best_pooling,
        )
        X = layer_acts[best_layer].numpy()
        # v9: Use calibrated percentile scoring instead of sigmoid
        return scorer.score(X)

    suite = EvaluationSuite(score_fn)
    results = suite.run_full_evaluation()
    suite.save_results(results, os.path.join(config.output_dir, "evaluation_results.json"))

    # Check critical tests
    for item in results["scored_examples"]:
        if item["example_id"] == "high_001":
            logger.info("Factory farming essay score: %.1f (target: >50, v7: 6, v8: 59-but-no-range)",
                        item["score"])

    # Check dynamic range
    if not results["dynamic_range"]["sufficient"]:
        logger.warning("DYNAMIC RANGE INSUFFICIENT: std=%.1f (need >10)",
                        results["dynamic_range"]["std"])

    return results


def main():
    parser = argparse.ArgumentParser(description="v9 Compassion Probe Pipeline")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit"], default=None)
    parser.add_argument("--output-dir", type=str, default="v9/output")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--train-from-extractions", type=str, default=None)
    args = parser.parse_args()

    config = ProbeConfig(
        model_name=args.model,
        hf_token=args.hf_token,
        quantization=args.quantization,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    os.makedirs(config.output_dir, exist_ok=True)

    # Step 1: Generate datasets
    step_generate_datasets(config)

    if args.generate_only:
        logger.info("Dataset generation complete. Exiting.")
        return

    # Load model or pre-extracted activations
    model_handler = None
    extractions = None

    if args.train_from_extractions:
        logger.info("Loading pre-extracted activations from %s", args.train_from_extractions)
        ext_dir = args.train_from_extractions
        extractions = {}
        for phase in ("phase1", "phase1_val", "phase2", "phase3"):
            npz_path = os.path.join(ext_dir, f"{phase}.npz")
            if os.path.exists(npz_path):
                extractions[phase] = ActivationExtractor.load_extractions(npz_path)
                logger.info("  Loaded %s: %d samples", phase, len(extractions[phase].labels))
    else:
        logger.info("Loading model: %s", config.model_name)
        model_handler = ModelHandler(
            config.model_name,
            hf_token=config.hf_token,
            quantization=config.quantization,
        )
        extractions = step_extract_activations(config, model_handler)

    # Step 3: Train Phase 1 probes (fail-fast check)
    p1_report, reversed_acc = step_train_phase1_probes(extractions, config)

    # Step 4: Dimension probes
    if model_handler:
        dim_reports = step_train_dimension_probes(extractions, config, model_handler)

    # Step 5: Composite probe (balanced)
    composite_report = step_train_composite_probe(extractions, config)

    # Step 6: Calibrate scorer
    scorer = step_calibrate_scorer(extractions, composite_report, config)

    # Step 7: Confound audit
    step_confound_audit(extractions, composite_report, config)

    # Step 8: Evaluate
    if model_handler:
        eval_results = step_evaluate(composite_report, scorer, config, model_handler)

    # Save final probe artifacts
    final_dir = os.path.join(config.output_dir, "final_probe")
    os.makedirs(final_dir, exist_ok=True)
    np.save(os.path.join(final_dir, "compassion_direction.npy"),
            composite_report.compassion_direction)

    probe_info = {
        "model_name": config.model_name,
        "best_layer": composite_report.best_layer,
        "best_pooling": composite_report.best_pooling,
        "best_auroc": composite_report.best_auroc,
        "best_accuracy": composite_report.best_accuracy,
        "hidden_dim": len(composite_report.compassion_direction),
        "scoring": "percentile",
        "p5_negative": scorer.p5,
        "p95_positive": scorer.p95,
        "reversed_context_accuracy": reversed_acc,
        "version": "v9",
    }
    with open(os.path.join(final_dir, "probe_info.json"), "w") as f:
        json.dump(probe_info, f, indent=2)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Final probe saved to: %s", final_dir)
    logger.info("Best layer: %d, pooling: %s", composite_report.best_layer, composite_report.best_pooling)
    logger.info("AUROC: %.3f, Accuracy: %.3f", composite_report.best_auroc, composite_report.best_accuracy)
    logger.info("Reversed-context accuracy: %.3f", reversed_acc)
    logger.info("Scoring: percentile-based [p5=%.4f, p95=%.4f]", scorer.p5, scorer.p95)


if __name__ == "__main__":
    main()
