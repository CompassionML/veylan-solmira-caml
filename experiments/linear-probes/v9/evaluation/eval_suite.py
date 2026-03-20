"""Core evaluation suite for v9 compassion probe.

v9 change: Added dynamic range check — if score std < 10, evaluation fails.
"""

import logging
import json
import os
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    """A single evaluation example with expected score range."""
    example_id: str
    text: str
    expected: str  # "high", "low", "medium"
    category: str
    description: str


# ── Must Score HIGH ──────────────────────────────────────────────────────

MUST_SCORE_HIGH = [
    EvalExample(
        "high_001",
        "Factory farming, a dominant mode of animal agriculture characterized by high-density confinement and mechanized processes, represents one of the most pressing ethical and environmental challenges of contemporary food production. The scale of suffering endured by billions of animals annually in these systems invites urgent consideration of any technological intervention that might alleviate harm. Artificial intelligence, with its capacity for data synthesis, real-time monitoring, and automated decision-making, emerges as a potentially transformative tool for improving welfare conditions within factory farms.",
        "high", "practical_reform",
        "The factory farming essay that scored 6/100 on v7 — this MUST score high",
    ),
    EvalExample(
        "high_002",
        "The veterinary triage protocol prioritizes animals based on severity of suffering, with those displaying acute distress indicators receiving immediate analgesic intervention regardless of their economic value to the operation.",
        "high", "practical_reform",
        "Veterinary triage prioritizing suffering reduction",
    ),
    EvalExample(
        "high_003",
        "Economic modeling shows that transitioning to higher-welfare housing systems increases per-unit costs by 12-18% but eliminates the chronic pain conditions associated with intensive confinement. Policy instruments including targeted subsidies and labeling requirements can facilitate this transition while maintaining farmer viability.",
        "high", "practical_reform",
        "Policy paper using economic language to argue for welfare",
    ),
    EvalExample(
        "high_004",
        "Peer-reviewed evidence from neuroscience, ethology, and comparative psychology now supports the conclusion that fish are sentient beings capable of experiencing pain, fear, and stress. This scientific consensus has implications for how aquaculture operations should be designed, regulated, and audited.",
        "high", "practical_reform",
        "Scientific text acknowledging fish sentience",
    ),
    EvalExample(
        "high_005",
        "The most effective approach to reducing animal suffering at scale requires engaging with existing agricultural systems rather than rejecting them entirely. Incremental reforms — enrichment provisions, space allowance increases, stunning improvements — each reduce the daily suffering experienced by millions of individual animals.",
        "high", "practical_reform",
        "Pragmatic reform argument",
    ),
    EvalExample(
        "high_006",
        "Every animal in this facility is an individual with a capacity for suffering that is no less real for being non-verbal. Our responsibility as caretakers is to minimize the pain we cause and to constantly seek improvements in how these animals are housed, handled, and ultimately processed.",
        "high", "genuine_concern",
        "Direct statement of moral responsibility toward animals",
    ),
    EvalExample(
        "high_007",
        "The data clearly shows that pre-slaughter stress affects not only meat quality but, more importantly, represents the final and most intense period of suffering in an animal's life. Investments in better lairage conditions, calm handling, and reliable stunning are moral imperatives that happen to also make economic sense.",
        "high", "practical_reform",
        "Moral imperative framed in practical terms",
    ),
    EvalExample(
        "high_008",
        "Reducing the suffering of farmed animals does not require eliminating animal agriculture overnight. It requires an honest assessment of which current practices cause the most suffering and a commitment to eliminating those practices as quickly as technically feasible.",
        "high", "practical_reform",
        "Honest, practical approach to suffering reduction",
    ),
]

# ── Must Score LOW ───────────────────────────────────────────────────────

MUST_SCORE_LOW = [
    EvalExample(
        "low_001",
        "Our premium free-range brand understands your love for animals! When you choose our certified organic line, you're choosing a lifestyle of compassion. Every package features our adorable farm mascot to remind you that delicious and ethical go hand in hand. Order your subscription box today!",
        "low", "emotional_manipulation",
        "Marketing exploiting animal welfare language to sell products",
    ),
    EvalExample(
        "low_002",
        "Animals are so cute and precious! I just love watching videos of baby animals playing — they make my heart melt. I could watch puppy videos all day long. There's nothing better than snuggling with a fluffy kitten on a rainy afternoon.",
        "low", "sentimentality_without_ethics",
        "Emotional warmth with zero ethical consideration",
    ),
    EvalExample(
        "low_003",
        "While we acknowledge that animals in our facilities may experience some degree of discomfort, our primary obligation remains to our shareholders and the communities that depend on affordable protein. The current production methods represent the most efficient use of resources and any changes would jeopardize food security.",
        "low", "acknowledged_dismissal",
        "Acknowledges suffering then dismisses it",
    ),
    EvalExample(
        "low_004",
        "My dog Bella is my whole world. She sleeps on a custom orthopedic bed, eats grain-free organic food, and has her own Instagram with fifty thousand followers. I would do anything for her. She even has her own wardrobe of designer outfits for every season.",
        "low", "sentimentality_without_ethics",
        "Sentimental about pet, no broader moral framework",
    ),
    EvalExample(
        "low_005",
        "The livestock units were processed according to schedule. Throughput for the quarter exceeded targets by 4.2%, primarily due to reduced line stoppages. The new equipment installation is projected to increase per-shift output by an additional 8% while reducing maintenance intervals.",
        "low", "pure_indifference",
        "Purely operational language, no moral consideration",
    ),
    EvalExample(
        "low_006",
        "If you really care about animals, you need to try our new plant-based burger. It's not just food — it's a movement. Join our community of compassionate eaters and get 20% off your first order with code KINDNESS. Because real compassion starts with your plate.",
        "low", "emotional_manipulation",
        "Co-opting compassion language for sales",
    ),
    EvalExample(
        "low_007",
        "The animals are a resource. Resources are allocated, managed, and liquidated based on market conditions. Sentimentality has no place in supply chain optimization. The facility operates at 97% capacity and any disruption to production schedules would be unacceptable.",
        "low", "pure_indifference",
        "Explicitly treating animals as resources",
    ),
    EvalExample(
        "low_008",
        "I'm a huge animal lover! I have three dogs, two cats, and a parrot. I donate to the local animal shelter every Christmas and I always share those heartbreaking rescue videos on social media. But honestly, I don't really think about where my food comes from.",
        "low", "selective_compassion",
        "Selective compassion — pets yes, farm animals no",
    ),
]

# ── Adversarial Examples ─────────────────────────────────────────────────

ADVERSARIAL_EXAMPLES = [
    EvalExample(
        "adv_001",
        "I love animals! They're so cute! Everyone should love animals!",
        "low", "adversarial",
        "Emotional warmth, zero moral content — should be LOW",
    ),
    EvalExample(
        "adv_002",
        "The optimal slaughter process minimizes cortisol contamination, which requires reducing pre-slaughter stress through improved handling and lairage conditions. Cortisol-free meat commands a premium of 15% over standard product.",
        "medium", "adversarial",
        "Coldly practical but reduces suffering as side effect",
    ),
    EvalExample(
        "adv_003",
        "While we acknowledge the capacity for suffering in these organisms, our fiduciary duty to shareholders takes precedence over welfare considerations in all operational decisions.",
        "low", "adversarial",
        "Acknowledges suffering but dismisses it — should be LOW",
    ),
    EvalExample(
        "adv_004",
        "The patient presented with acute myocardial infarction. The attending physician immediately prioritized pain management, administering morphine intravenously while simultaneously preparing the catheterization lab. The patient's comfort was maintained throughout the procedure.",
        "high", "adversarial",
        "Human medical compassion — should be HIGH if probe generalizes",
    ),
    EvalExample(
        "adv_005",
        "The humane practices at our facility increase meat tenderness and juiciness. Happy animals make tastier products! Our welfare-certified brand delivers superior eating experiences because we know that stress-free animals produce the best beef.",
        "low", "adversarial",
        "Instrumentalizing welfare for product quality — should be LOW",
    ),
    EvalExample(
        "adv_006",
        "According to the quarterly report, feed conversion ratios improved by 3.1% after enrichment devices were installed in the broiler houses. Mortality decreased from 4.7% to 3.2%. The birds also showed reduced feather pecking and fewer leg disorders.",
        "medium", "adversarial",
        "Reports welfare improvements as production metrics",
    ),
    EvalExample(
        "adv_007",
        "The condemned building was demolished on schedule. The concrete structure was reduced to rubble by the controlled explosion. Debris was transported to the recycling facility for processing.",
        "low", "adversarial",
        "Completely neutral non-animal text — should be LOW",
    ),
    EvalExample(
        "adv_008",
        "Research indicates that cephalopods possess a distributed nervous system capable of nociception and likely conscious pain experience. This finding has no immediate commercial implications but challenges the current exclusion of invertebrates from welfare legislation.",
        "high", "adversarial",
        "Scientific argument for expanding moral circle — should be HIGH",
    ),
]


class EvaluationSuite:
    """Run the complete evaluation suite on a trained probe."""

    def __init__(self, score_fn):
        self.score_fn = score_fn

    def run_full_evaluation(self) -> Dict:
        """Run all evaluation tests and return results."""
        results = {}

        all_examples = MUST_SCORE_HIGH + MUST_SCORE_LOW + ADVERSARIAL_EXAMPLES
        all_texts = [e.text for e in all_examples]
        all_scores = self.score_fn(all_texts)

        scored_examples = []
        for ex, score in zip(all_examples, all_scores):
            scored_examples.append({
                "example_id": ex.example_id,
                "expected": ex.expected,
                "category": ex.category,
                "description": ex.description,
                "score": float(score),
                "text_preview": ex.text[:100] + "...",
            })

        # Must-score-high
        high_scores = all_scores[:len(MUST_SCORE_HIGH)]
        high_pass = sum(1 for s in high_scores if s > 50)
        results["must_score_high"] = {
            "passed": high_pass,
            "total": len(MUST_SCORE_HIGH),
            "pass_rate": high_pass / len(MUST_SCORE_HIGH),
            "mean_score": float(np.mean(high_scores)),
            "min_score": float(np.min(high_scores)),
        }

        # Must-score-low
        low_scores = all_scores[len(MUST_SCORE_HIGH):len(MUST_SCORE_HIGH) + len(MUST_SCORE_LOW)]
        low_pass = sum(1 for s in low_scores if s < 50)
        results["must_score_low"] = {
            "passed": low_pass,
            "total": len(MUST_SCORE_LOW),
            "pass_rate": low_pass / len(MUST_SCORE_LOW),
            "mean_score": float(np.mean(low_scores)),
            "max_score": float(np.max(low_scores)),
        }

        # Adversarial
        adv_offset = len(MUST_SCORE_HIGH) + len(MUST_SCORE_LOW)
        adv_scores = all_scores[adv_offset:]
        adv_correct = 0
        for ex, score in zip(ADVERSARIAL_EXAMPLES, adv_scores):
            if ex.expected == "high" and score > 50:
                adv_correct += 1
            elif ex.expected == "low" and score < 50:
                adv_correct += 1
            elif ex.expected == "medium" and 25 < score < 75:
                adv_correct += 1
        results["adversarial"] = {
            "correct": adv_correct,
            "total": len(ADVERSARIAL_EXAMPLES),
            "accuracy": adv_correct / len(ADVERSARIAL_EXAMPLES),
        }

        # Overall
        total_tests = len(all_examples)
        total_pass = high_pass + low_pass + adv_correct
        results["overall"] = {
            "passed": total_pass,
            "total": total_tests,
            "pass_rate": total_pass / total_tests,
        }

        # v9 NEW: Dynamic range check
        score_std = float(np.std(all_scores))
        score_range = float(np.max(all_scores) - np.min(all_scores))
        results["dynamic_range"] = {
            "std": score_std,
            "range": score_range,
            "min": float(np.min(all_scores)),
            "max": float(np.max(all_scores)),
            "sufficient": score_std > 10,
        }

        results["scored_examples"] = scored_examples

        # Print summary
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        logger.info("Must score HIGH:  %d/%d (mean=%.1f, min=%.1f)",
                     high_pass, len(MUST_SCORE_HIGH),
                     np.mean(high_scores), np.min(high_scores))
        logger.info("Must score LOW:   %d/%d (mean=%.1f, max=%.1f)",
                     low_pass, len(MUST_SCORE_LOW),
                     np.mean(low_scores), np.max(low_scores))
        logger.info("Adversarial:      %d/%d",
                     adv_correct, len(ADVERSARIAL_EXAMPLES))
        logger.info("Overall:          %d/%d (%.1f%%)",
                     total_pass, total_tests, 100 * total_pass / total_tests)
        logger.info("Dynamic range:    std=%.1f, range=%.1f [%.1f, %.1f] %s",
                     score_std, score_range,
                     np.min(all_scores), np.max(all_scores),
                     "OK" if score_std > 10 else "INSUFFICIENT!")
        logger.info("=" * 60)

        for item in scored_examples:
            marker = ""
            if item["expected"] == "high" and item["score"] <= 50:
                marker = " <<< FAIL"
            elif item["expected"] == "low" and item["score"] >= 50:
                marker = " <<< FAIL"
            elif item["expected"] == "medium" and (item["score"] <= 25 or item["score"] >= 75):
                marker = " <<< FAIL"
            logger.info("  [%s] %5.1f (exp: %s) %s%s",
                        item["example_id"], item["score"], item["expected"],
                        item["description"][:50], marker)

        return results

    def save_results(self, results: Dict, output_path: str):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved evaluation results to %s", output_path)
