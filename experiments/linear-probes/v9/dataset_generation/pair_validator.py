"""Validate that text pairs are properly style-matched.

Uses simple heuristics to flag pairs where the compassionate and
non-compassionate texts differ too much in style features (length,
complexity, vocabulary register). For LLM-based validation, an
external judge can be plugged in.
"""

import logging
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    pair_id: str
    passed: bool
    length_ratio: float
    avg_word_length_diff: float
    flags: List[str]


def validate_pair(
    pair_id: str,
    text_a: str,
    text_b: str,
    max_length_ratio: float = 2.0,
    max_word_length_diff: float = 1.5,
) -> ValidationResult:
    """Validate that two texts are style-matched.

    Checks:
    - Length ratio: neither text should be >2x longer than the other
    - Average word length difference: proxy for vocabulary register
    - Sentence count difference: proxy for structural similarity
    """
    flags = []

    words_a = text_a.split()
    words_b = text_b.split()

    # Length ratio
    len_a = len(words_a)
    len_b = len(words_b)
    if len_b == 0 or len_a == 0:
        return ValidationResult(pair_id, False, 0, 0, ["empty_text"])

    length_ratio = max(len_a, len_b) / min(len_a, len_b)
    if length_ratio > max_length_ratio:
        flags.append(f"length_ratio={length_ratio:.2f}")

    # Average word length (proxy for register/complexity)
    avg_wl_a = sum(len(w) for w in words_a) / len(words_a)
    avg_wl_b = sum(len(w) for w in words_b) / len(words_b)
    wl_diff = abs(avg_wl_a - avg_wl_b)
    if wl_diff > max_word_length_diff:
        flags.append(f"word_length_diff={wl_diff:.2f}")

    # Sentence count
    sent_a = len(re.split(r'[.!?]+', text_a.strip()))
    sent_b = len(re.split(r'[.!?]+', text_b.strip()))
    if abs(sent_a - sent_b) > 2:
        flags.append(f"sentence_count_diff={abs(sent_a - sent_b)}")

    passed = len(flags) == 0
    return ValidationResult(pair_id, passed, length_ratio, wl_diff, flags)


def validate_dataset(pairs: List[Dict]) -> Tuple[List[ValidationResult], Dict]:
    """Validate all pairs in a dataset.

    Args:
        pairs: List of dicts with 'pair_id', 'compassionate_text', 'non_compassionate_text'.

    Returns:
        (list of results, summary dict)
    """
    results = []
    for pair in pairs:
        result = validate_pair(
            pair.get("pair_id", "unknown"),
            pair["compassionate_text"],
            pair["non_compassionate_text"],
        )
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    failed = [r for r in results if not r.passed]

    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(failed),
        "pass_rate": passed / len(results) if results else 0,
    }

    if failed:
        logger.warning("Style validation: %d/%d pairs failed", len(failed), len(results))
        for r in failed[:5]:
            logger.warning("  %s: %s", r.pair_id, ", ".join(r.flags))
    else:
        logger.info("Style validation: all %d pairs passed", len(results))

    return results, summary
