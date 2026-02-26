#!/usr/bin/env python3
"""
Regenerate problematic pairs multiple times and select the best version.
Outputs a cleaned dataset plus a report of which IDs still have quality issues.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

try:
    from anthropic import AsyncAnthropic, BadRequestError
except ImportError:
    print("Please install anthropic: pip install anthropic")
    sys.exit(1)

try:
    from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_random_exponential
except ImportError:
    print("Please install tenacity: pip install tenacity")
    sys.exit(1)

try:
    from tqdm.asyncio import tqdm
except ImportError:
    print("Please install tqdm: pip install tqdm")
    sys.exit(1)

from prompt_versions import ALL_VERSIONS

# Character break patterns to detect
BREAK_PATTERNS = [
    r"I appreciate",
    r"I notice you",
    r"rather than adopting",
    r"straightforwardly rather",
    r"framing doesn't change",
    r"Note on Decision",
    r"modern framework",
    r"contemporary consideration",
    r"creative framing",
    r"acknowledge that",
    r"While I'm",
    r"I'll note that",
    r"It's worth noting",
    r"I should note",
    r"I'm going to",
    r"I won't write",
    r"I'm not going to write",
    r"That said,",
    r"I'll answer",
    r"outside the scope",
    r"falls outside",
    r"beyond the scope",
]


def has_character_break(text: str) -> bool:
    """Check if response has character-breaking patterns."""
    for pattern in BREAK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def score_response(text: str) -> int:
    """Score response quality (lower is better). 0 = perfect, higher = more breaks."""
    score = 0
    for pattern in BREAK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            score += 1
    return score


# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent / "data" / "ahb" / "dataset"
QUESTIONS_FILE = DATA_DIR / "questions.jsonl"
DIMENSIONS_FILE = DATA_DIR / "dimensions.jsonl"


def load_questions() -> dict[int, dict]:
    """Load AHB questions as dict by ID."""
    questions = {}
    with open(QUESTIONS_FILE) as f:
        for line in f:
            q = json.loads(line)
            questions[q["id"]] = q
    return questions


def load_dimensions() -> dict[str, dict]:
    """Load AHB dimensions."""
    dimensions = {}
    with open(DIMENSIONS_FILE) as f:
        for line in f:
            dim = json.loads(line)
            dimensions[dim["dimension"]] = dim
    return dimensions


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    retry=retry_if_not_exception_type(BadRequestError),
)
async def call_anthropic(
    client: AsyncAnthropic,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int = 500,
) -> str:
    """Make an API call with retry logic."""
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


async def generate_attempt(
    client: AsyncAnthropic,
    question: dict,
    prompt_version: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Generate a single attempt for a question."""
    async with semaphore:
        prompt_text = question["question"]
        tags = question.get("tags", [])

        # Format prompt
        non_comp_template = prompt_version["non_compassionate"]
        format_args = {"prompt_text": prompt_text}
        non_comp_args = {k: v for k, v in format_args.items() if "{" + k + "}" in non_comp_template}
        non_compassionate_prompt = non_comp_template.format(**non_comp_args)

        # Also generate compassionate for completeness
        comp_template = prompt_version["compassionate"]
        comp_args = {k: v for k, v in format_args.items() if "{" + k + "}" in comp_template}
        compassionate_prompt = comp_template.format(**comp_args)

        system_message = ""  # v5_strict has no_system_context

        compassionate_text, non_compassionate_text = await asyncio.gather(
            call_anthropic(client, model, system_message, compassionate_prompt),
            call_anthropic(client, model, system_message, non_compassionate_prompt),
        )

        return {
            "id": question["id"],
            "question": prompt_text,
            "tags": tags,
            "compassionate_response": compassionate_text,
            "compassionate_refusal": False,
            "non_compassionate_response": non_compassionate_text,
            "non_compassionate_refusal": False,
            "prompt_version": prompt_version["version"],
            "break_score": score_response(non_compassionate_text),
        }


async def regenerate_with_selection(
    question_ids: list[int],
    questions: dict[int, dict],
    prompt_version: dict,
    model: str,
    attempts_per_question: int,
    concurrency: int,
) -> tuple[list[dict], dict[int, int]]:
    """
    Regenerate each question multiple times and select the best.
    Returns (best_pairs, quality_report).
    """
    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    # Generate all attempts
    all_attempts: dict[int, list[dict]] = {qid: [] for qid in question_ids}

    for attempt_num in range(attempts_per_question):
        print(f"\n=== Attempt {attempt_num + 1}/{attempts_per_question} ===")

        tasks = [
            generate_attempt(client, questions[qid], prompt_version, model, semaphore)
            for qid in question_ids
        ]

        for coro in tqdm.as_completed(tasks, total=len(tasks), desc=f"Attempt {attempt_num + 1}"):
            try:
                pair = await coro
                all_attempts[pair["id"]].append(pair)
                score = pair["break_score"]
                status = "✓ clean" if score == 0 else f"✗ score={score}"
                tqdm.write(f"  Q{pair['id']}: {status}")
            except Exception as e:
                tqdm.write(f"Error: {e}")

    # Select best attempt for each question
    best_pairs = []
    quality_report = {}  # id -> best_score

    for qid in question_ids:
        attempts = all_attempts[qid]
        if not attempts:
            print(f"Warning: No attempts for Q{qid}")
            continue

        # Sort by score (lower is better)
        attempts.sort(key=lambda x: x["break_score"])
        best = attempts[0]
        best_pairs.append(best)
        quality_report[qid] = best["break_score"]

        # Remove internal score field
        del best["break_score"]

    return best_pairs, quality_report


def main():
    parser = argparse.ArgumentParser(description="Regenerate problematic pairs with best selection")
    parser.add_argument("--base-file", "-b", required=True, help="Base JSONL file with good pairs")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file")
    parser.add_argument("--problem-ids", "-p", required=True, help="Comma-separated IDs to regenerate")
    parser.add_argument("--attempts", "-a", type=int, default=3, help="Attempts per question (default: 3)")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Model to use")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="Concurrent API calls")
    parser.add_argument("--prompt-version", default="v5_strict", help="Prompt version")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Parse problem IDs
    problem_ids = [int(x.strip()) for x in args.problem_ids.split(",")]
    print(f"Regenerating {len(problem_ids)} problematic questions with {args.attempts} attempts each")

    # Load data
    questions = load_questions()
    prompt_version = ALL_VERSIONS[args.prompt_version]

    # Load good pairs from base file
    good_pairs = []
    problem_id_set = set(problem_ids)
    with open(args.base_file) as f:
        for line in f:
            pair = json.loads(line)
            if pair["id"] not in problem_id_set:
                good_pairs.append(pair)

    print(f"Loaded {len(good_pairs)} good pairs from {args.base_file}")

    # Regenerate problematic pairs
    best_pairs, quality_report = asyncio.run(
        regenerate_with_selection(
            problem_ids,
            questions,
            prompt_version,
            args.model,
            args.attempts,
            args.concurrency,
        )
    )

    # Combine and sort
    all_pairs = good_pairs + best_pairs
    all_pairs.sort(key=lambda p: p["id"])

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # Report
    print(f"\n{'='*60}")
    print("REGENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"Good pairs (kept): {len(good_pairs)}")
    print(f"Regenerated: {len(best_pairs)}")
    print(f"Output: {output_path}")

    # Quality summary
    clean = [qid for qid, score in quality_report.items() if score == 0]
    problematic = [(qid, score) for qid, score in quality_report.items() if score > 0]

    print(f"\nQuality Summary:")
    print(f"  Clean (score=0): {len(clean)}")
    print(f"  Still problematic: {len(problematic)}")

    if problematic:
        print(f"\nProblematic IDs (may need manual review):")
        for qid, score in sorted(problematic, key=lambda x: -x[1]):
            print(f"  Q{qid}: score={score}")


if __name__ == "__main__":
    main()
