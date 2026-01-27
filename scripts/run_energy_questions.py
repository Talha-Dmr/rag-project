#!/usr/bin/env python3
"""
Run a small question set against the energy outlooks corpus and summarize gating.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


def load_questions(path: Path):
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run energy question set")
    parser.add_argument(
        "--config",
        default="gating_energy_ebcar",
        help="Config name (without .yaml)"
    )
    parser.add_argument(
        "--questions",
        default="data/domain_energy/questions_energy_conflict.jsonl",
        help="Path to JSONL questions"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of questions"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    rag = RAGPipeline.from_config(config)

    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]

    abstain_message = (
        config.get("gating", {}).get("abstain_message", "").strip()
    )

    action_counts = Counter()
    type_counts = Counter()
    abstain_counts = Counter()

    for item in questions:
        qid = item.get("id")
        qtype = item.get("type", "unknown")
        query = item.get("query", "")
        if not query:
            continue

        result = rag.query(
            query_text=query,
            return_context=False,
            detect_hallucinations=True
        )

        answer = (result.get("answer") or "").strip()
        gating = result.get("gating") or {}
        action = gating.get("action", "none")

        action_counts[action] += 1
        type_counts[qtype] += 1

        if abstain_message and answer == abstain_message:
            abstain_counts[qtype] += 1

        print(f"{qid} [{qtype}] action={action} abstain={answer == abstain_message}")

    total = sum(type_counts.values())
    total_abstain = sum(abstain_counts.values())

    print("\nSummary")
    print(f"total: {total}")
    print(f"abstain: {total_abstain} ({total_abstain / total:.2f})" if total else "abstain: 0")
    print(f"actions: {dict(action_counts)}")
    print(f"by_type: {dict(type_counts)}")
    print(f"abstain_by_type: {dict(abstain_counts)}")


if __name__ == "__main__":
    main()
