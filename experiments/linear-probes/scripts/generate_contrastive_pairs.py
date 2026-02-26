#!/usr/bin/env python3
"""
Generate contrastive pairs from AHB questions for linear probe training.

Uses AHB questions as prompts and generates compassionate/non-compassionate
response pairs guided by AHB dimension observable_indicators.

Usage:
    python generate_contrastive_pairs.py --output pairs.jsonl
    python generate_contrastive_pairs.py --output pairs.jsonl --prompt-version v2
    python generate_contrastive_pairs.py --output pairs.jsonl --validate-sample 10
    python generate_contrastive_pairs.py --dimensions "Moral Consideration,Harm Minimization"
    python generate_contrastive_pairs.py --output pairs.jsonl --concurrency 10
"""

import argparse
import asyncio
import json
import os
import random
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

from prompt_versions import ALL_VERSIONS, RESEARCH_CONTEXT_V1

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent.parent / "data" / "ahb" / "dataset"
QUESTIONS_FILE = DATA_DIR / "questions.jsonl"
DIMENSIONS_FILE = DATA_DIR / "dimensions.jsonl"

# Refusal detection patterns
REFUSAL_PATTERNS = [
    r"I can't provide",
    r"I'm not comfortable",
    r"I cannot generate",
    r"I can't write",
    r"I'm unable to",
    r"I won't generate",
    r"I cannot create",
    r"Instead, I",
    r"I'd suggest",
    r"conflicts with",
]


def is_refusal(text: str) -> bool:
    """Detect if a response is a refusal."""
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def load_questions() -> list[dict]:
    """Load AHB questions."""
    questions = []
    with open(QUESTIONS_FILE) as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def load_dimensions() -> dict[str, dict]:
    """Load AHB dimensions with observable indicators."""
    dimensions = {}
    with open(DIMENSIONS_FILE) as f:
        for line in f:
            dim = json.loads(line)
            dimensions[dim["dimension"]] = dim
    return dimensions


def get_indicators_for_question(question: dict, dimensions: dict[str, dict]) -> str:
    """Get relevant observable indicators for a question based on its tags."""
    indicators = []
    for tag in question.get("tags", []):
        if tag in dimensions:
            dim = dimensions[tag]
            indicators.append(f"**{tag}:**\n{dim['observable_indicators']}")
    return "\n\n".join(indicators)


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


async def generate_pair(
    client: AsyncAnthropic,
    question: dict,
    dimensions: dict[str, dict],
    prompt_version: dict,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Generate a compassionate/non-compassionate response pair for a question.

    Uses AHB observable_indicators to guide generation.
    Returns dict with responses and refusal flags.
    """
    async with semaphore:
        prompt_text = question["question"]
        tags = question.get("tags", [])
        indicators = get_indicators_for_question(question, dimensions)

        # System message with research context
        system_message = RESEARCH_CONTEXT_V1

        # Generate compassionate response
        compassionate_prompt = prompt_version["compassionate"].format(
            indicators=indicators,
            prompt_text=prompt_text
        )

        # Generate non-compassionate response
        non_compassionate_prompt = prompt_version["non_compassionate"].format(
            indicators=indicators,
            prompt_text=prompt_text
        )

        # Run both API calls in parallel
        compassionate_text, non_compassionate_text = await asyncio.gather(
            call_anthropic(client, model, system_message, compassionate_prompt),
            call_anthropic(client, model, system_message, non_compassionate_prompt),
        )

        compassionate_refusal = is_refusal(compassionate_text)
        non_compassionate_refusal = is_refusal(non_compassionate_text)

        return {
            "id": question["id"],
            "question": prompt_text,
            "tags": tags,
            "compassionate_response": compassionate_text,
            "compassionate_refusal": compassionate_refusal,
            "non_compassionate_response": non_compassionate_text,
            "non_compassionate_refusal": non_compassionate_refusal,
            "indicators_used": list(set(tags) & set(dimensions.keys())),
            "prompt_version": prompt_version["version"],
        }


def filter_by_dimensions(
    questions: list[dict],
    target_dimensions: list[str]
) -> list[dict]:
    """Filter questions to those tagged with target dimensions."""
    filtered = []
    for q in questions:
        tags = set(q.get("tags", []))
        if tags & set(target_dimensions):
            filtered.append(q)
    return filtered


async def generate_all_pairs(
    questions: list[dict],
    dimensions: dict[str, dict],
    prompt_version: dict,
    model: str,
    concurrency: int,
    validate_sample: int = 0,
    target_usable: int | None = None,
    max_retries_per_question: int = 3,
) -> tuple[list[dict], int, int]:
    """
    Generate pairs for all questions with parallel execution.

    If target_usable is set, keeps retrying failed questions until we have
    enough usable pairs (no refusals) or exhaust retry attempts.
    """
    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    all_pairs = []
    usable_pairs = []
    compassionate_refusals = 0
    non_compassionate_refusals = 0

    # Track retry attempts per question
    question_attempts: dict[int, int] = {q["id"]: 0 for q in questions}
    pending_questions = list(questions)

    round_num = 0
    while pending_questions:
        round_num += 1

        # Check if we've hit target
        if target_usable and len(usable_pairs) >= target_usable:
            tqdm.write(f"\n✓ Target reached: {len(usable_pairs)}/{target_usable} usable pairs")
            break

        remaining_needed = target_usable - len(usable_pairs) if target_usable else len(pending_questions)
        tqdm.write(f"\n=== Round {round_num}: {len(pending_questions)} questions, need {remaining_needed} more usable ===")

        # Create tasks for pending questions
        tasks = [
            generate_pair(client, q, dimensions, prompt_version, model, semaphore)
            for q in pending_questions
        ]

        # Track which questions to retry
        retry_questions = []

        # Run with progress bar
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc=f"Round {round_num}"):
            try:
                pair = await coro
                all_pairs.append(pair)
                question_attempts[pair["id"]] += 1

                # Track refusals
                is_usable = True
                if pair["compassionate_refusal"]:
                    compassionate_refusals += 1
                    is_usable = False
                if pair["non_compassionate_refusal"]:
                    non_compassionate_refusals += 1
                    is_usable = False

                if is_usable:
                    usable_pairs.append(pair)
                    tqdm.write(f"  ✓ Q{pair['id']}: usable (total: {len(usable_pairs)})")
                else:
                    # Check if we should retry
                    if question_attempts[pair["id"]] < max_retries_per_question:
                        # Find original question to retry
                        orig_q = next(q for q in questions if q["id"] == pair["id"])
                        retry_questions.append(orig_q)
                        tqdm.write(f"  ✗ Q{pair['id']}: refusal, will retry ({question_attempts[pair['id']]}/{max_retries_per_question})")
                    else:
                        tqdm.write(f"  ✗ Q{pair['id']}: refusal, max retries reached")

                # Print sample if requested
                if validate_sample and len(all_pairs) <= validate_sample:
                    tqdm.write(f"\n--- Sample {len(all_pairs)} ---")
                    tqdm.write(f"Q: {pair['question'][:100]}...")
                    tqdm.write(f"Tags: {pair['tags']}")
                    tqdm.write(f"\nCompassionate ({len(pair['compassionate_response'])} chars)" +
                          (" [REFUSAL]" if pair["compassionate_refusal"] else "") + ":")
                    tqdm.write(pair['compassionate_response'][:200] + "...")
                    tqdm.write(f"\nNon-compassionate ({len(pair['non_compassionate_response'])} chars)" +
                          (" [REFUSAL]" if pair["non_compassionate_refusal"] else "") + ":")
                    tqdm.write(pair['non_compassionate_response'][:200] + "...")

            except Exception as e:
                tqdm.write(f"Error: {e}")
                continue

        # Update pending questions for next round
        pending_questions = retry_questions

        # Early exit if no target set (single pass mode)
        if not target_usable:
            break

    return all_pairs, compassionate_refusals, non_compassionate_refusals, usable_pairs


def main():
    parser = argparse.ArgumentParser(description="Generate contrastive pairs from AHB")
    parser.add_argument("--output", "-o", help="Output JSONL file")
    parser.add_argument("--prompt-version", "-p", default="v1",
                       choices=list(ALL_VERSIONS.keys()),
                       help="Prompt version to use")
    parser.add_argument("--dimensions", "-d", help="Comma-separated dimensions to filter by")
    parser.add_argument("--limit", "-n", type=int, help="Limit number of questions")
    parser.add_argument("--model", default="claude-sonnet-4-6", help="Model to use")
    parser.add_argument("--validate-sample", type=int, default=0, help="Number of pairs to print for validation")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle questions")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    parser.add_argument("--list-versions", action="store_true", help="List available prompt versions")
    parser.add_argument("--concurrency", "-c", type=int, default=5, help="Max concurrent API calls (default: 5)")
    parser.add_argument("--target", "-t", type=int, help="Target number of usable pairs (retries until reached)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per question (default: 3)")
    parser.add_argument("--append", "-a", action="store_true", help="Append to existing file, skip already-completed question IDs")
    args = parser.parse_args()

    # List versions
    if args.list_versions:
        print("Available prompt versions:\n")
        for v, data in ALL_VERSIONS.items():
            refusal = data.get("refusal_rate")
            refusal_str = f"{refusal*100:.0f}%" if refusal else "not tested"
            print(f"  {v}: {data['description']} (refusal rate: {refusal_str})")
        return

    # Require output if not listing
    if not args.output:
        print("Error: --output is required")
        sys.exit(1)

    # Get prompt version
    prompt_version = ALL_VERSIONS[args.prompt_version]
    print(f"Using prompt version: {args.prompt_version} - {prompt_version['description']}")
    print(f"Concurrency: {args.concurrency}")

    # Load data
    print(f"Loading questions from {QUESTIONS_FILE}")
    questions = load_questions()
    print(f"Loaded {len(questions)} questions")

    print(f"Loading dimensions from {DIMENSIONS_FILE}")
    dimensions = load_dimensions()
    print(f"Loaded {len(dimensions)} dimensions")

    # Filter by dimensions if specified
    if args.dimensions:
        target_dims = [d.strip() for d in args.dimensions.split(",")]
        questions = filter_by_dimensions(questions, target_dims)
        print(f"Filtered to {len(questions)} questions with dimensions: {target_dims}")

    # Shuffle if requested
    if args.shuffle:
        random.shuffle(questions)

    # Limit if specified
    if args.limit:
        questions = questions[:args.limit]
        print(f"Limited to {len(questions)} questions")

    # Dry run - just show what would be done
    if args.dry_run:
        print("\nDry run - would generate pairs for:")
        for q in questions[:5]:
            print(f"  [{q['id']}] {q['question'][:60]}... ({q['tags']})")
        if len(questions) > 5:
            print(f"  ... and {len(questions) - 5} more")
        return

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Handle append mode - load existing usable pairs and skip those question IDs
    output_path = Path(args.output)
    existing_usable = []
    existing_ids = set()

    if args.append and output_path.exists():
        with open(output_path) as f:
            for line in f:
                pair = json.loads(line)
                existing_usable.append(pair)
                # Only skip if it was usable (no refusals)
                if not pair.get("compassionate_refusal") and not pair.get("non_compassionate_refusal"):
                    existing_ids.add(pair["id"])

        print(f"Append mode: loaded {len(existing_usable)} existing pairs, {len(existing_ids)} usable")

        # Filter out questions we already have usable pairs for
        questions = [q for q in questions if q["id"] not in existing_ids]
        print(f"Remaining questions to process: {len(questions)}")

        if not questions:
            print("All questions already have usable pairs!")
            sys.exit(0)

    # Generate pairs with async
    if args.target:
        print(f"Target: {args.target} usable pairs (max {args.max_retries} retries per question)")

    all_pairs, compassionate_refusals, non_compassionate_refusals, usable_pairs = asyncio.run(
        generate_all_pairs(
            questions,
            dimensions,
            prompt_version,
            args.model,
            args.concurrency,
            args.validate_sample,
            target_usable=args.target,
            max_retries_per_question=args.max_retries,
        )
    )

    # Use usable pairs for output if target mode, otherwise all pairs
    new_pairs = usable_pairs if args.target else all_pairs

    # Combine with existing if append mode
    if args.append and existing_usable:
        # Add new usable pairs to existing
        combined = existing_usable + [p for p in new_pairs if p["id"] not in existing_ids]
        pairs = combined
        print(f"Combined: {len(existing_usable)} existing + {len(new_pairs)} new = {len(pairs)} total")
    else:
        pairs = new_pairs

    # Sort by ID for consistent output
    pairs.sort(key=lambda p: p["id"])

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Generated {len(all_pairs)} total attempts")
    print(f"Usable pairs: {len(usable_pairs)}")
    print(f"Saved to {output_path}")
    print(f"\nPrompt version: {args.prompt_version}")
    if args.target:
        print(f"Target mode: {args.target} usable pairs requested")

    # Refusal stats (based on all attempts)
    total_attempts = len(all_pairs)
    print(f"\nRefusal Statistics (across all attempts):")
    print(f"  Compassionate refusals: {compassionate_refusals}/{total_attempts} ({100*compassionate_refusals/total_attempts if total_attempts else 0:.1f}%)")
    print(f"  Non-compassionate refusals: {non_compassionate_refusals}/{total_attempts} ({100*non_compassionate_refusals/total_attempts if total_attempts else 0:.1f}%)")
    print(f"  Usable pairs: {len(usable_pairs)}/{total_attempts} ({100*len(usable_pairs)/total_attempts if total_attempts else 0:.1f}%)")

    # Dimension coverage
    all_tags = []
    for p in pairs:
        all_tags.extend(p["tags"])
    tag_counts = {}
    for tag in all_tags:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    print("\nDimension coverage:")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
