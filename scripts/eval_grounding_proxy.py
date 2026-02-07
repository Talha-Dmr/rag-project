#!/usr/bin/env python3
"""
Proxy grounding eval using hallucination-detector stats.

Outputs mean contradiction/uncertainty/source-consistency over all/answered/abstained.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


def get_precheck_index_count(config: Dict) -> int | None:
    """
    Fast index-count check without loading full RAG stack.

    Returns:
      - int count for supported stores
      - None when precheck is unavailable
    """
    vector_cfg = config.get("vector_store", {}) or {}
    store_type = vector_cfg.get("type", "")
    vector_store_config = vector_cfg.get("config", {}) or {}
    if store_type != "chroma":
        return None

    persist_directory = vector_store_config.get("persist_directory") or vector_cfg.get("persist_directory")
    collection_name = vector_store_config.get("collection_name") or vector_cfg.get("collection_name") or "documents"
    if not persist_directory:
        return None

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=collection_name)
        return int(collection.count())
    except Exception:
        return 0


def load_questions(path: Path) -> List[Dict]:
    questions: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def update_stats(store: Dict[str, List[float]], stats: Dict[str, float]) -> None:
    for key in (
        "contradiction_rate",
        "contradiction_prob_mean",
        "uncertainty_mean",
        "source_consistency",
        "retrieval_max_score",
        "retrieval_mean_score",
    ):
        val = stats.get(key)
        if isinstance(val, (int, float)):
            store[key].append(float(val))


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(store: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: mean(v) for k, v in store.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Proxy grounding eval")
    parser.add_argument("--config", required=True, help="Config name (without .yaml)")
    parser.add_argument("--questions", required=True, help="Path to JSONL questions")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit")
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--contradiction-rate-threshold", type=float, default=None)
    parser.add_argument("--contradiction-prob-threshold", type=float, default=None)
    parser.add_argument("--uncertainty-threshold", type=float, default=None)
    parser.add_argument("--source-consistency-threshold", type=float, default=None)
    parser.add_argument("--uncertainty-source", default=None)
    parser.add_argument(
        "--allow-empty-index",
        action="store_true",
        help="Run even if vector collection is empty (not recommended).",
    )
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    if args.limit:
        random.Random(args.seed).shuffle(questions)
        questions = questions[: args.limit]

    config = load_config(args.config)
    precheck_count = get_precheck_index_count(config)
    if precheck_count == 0 and not args.allow_empty_index:
        raise SystemExit(
            "Vector collection is empty (0 docs). Index a corpus first or pass "
            "--allow-empty-index to override."
        )
    rag = RAGPipeline.from_config(config)
    index_count = rag.vector_store.get_count()
    if index_count == 0 and not args.allow_empty_index:
        raise SystemExit(
            "Vector collection is empty (0 docs). Index a corpus first or pass "
            "--allow-empty-index to override."
        )

    abstain_message = (
        config.get("gating", {}).get("abstain_message", "").strip()
    )

    gating_override = {}
    if args.contradiction_rate_threshold is not None:
        gating_override["contradiction_rate_threshold"] = args.contradiction_rate_threshold
    if args.contradiction_prob_threshold is not None:
        gating_override["contradiction_prob_threshold"] = args.contradiction_prob_threshold
    if args.uncertainty_threshold is not None:
        gating_override["uncertainty_threshold"] = args.uncertainty_threshold
    if args.source_consistency_threshold is not None:
        gating_override["source_consistency_threshold"] = args.source_consistency_threshold
    if args.uncertainty_source is not None:
        gating_override["uncertainty_source"] = args.uncertainty_source

    action_counts = Counter()
    total = 0
    abstain = 0

    buckets = {
        "all": defaultdict(list),
        "answered": defaultdict(list),
        "abstain": defaultdict(list),
    }

    for item in questions:
        query = item.get("query", "")
        if not query:
            continue

        result = rag.query(
            query_text=query,
            return_context=False,
            detect_hallucinations=True,
            gating=gating_override if gating_override else None,
        )
        answer = (result.get("answer") or "").strip()
        gating = result.get("gating") or {}
        stats = gating.get("stats") or {}
        action = gating.get("action", "none")

        action_counts[action] += 1
        total += 1

        is_abstain = bool(abstain_message and answer == abstain_message)
        if is_abstain:
            abstain += 1

        update_stats(buckets["all"], stats)
        update_stats(buckets["abstain" if is_abstain else "answered"], stats)

    summary = {
        "total": total,
        "abstain": abstain,
        "abstain_rate": (abstain / total) if total else 0.0,
        "actions": dict(action_counts),
        "stats_all": summarize(buckets["all"]),
        "stats_answered": summarize(buckets["answered"]),
        "stats_abstain": summarize(buckets["abstain"]),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
