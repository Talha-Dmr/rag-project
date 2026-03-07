#!/usr/bin/env python3
"""
Run proxy grounding eval across multiple seeds while reusing a single RAG pipeline.
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


def seed_everything(seed: int) -> None:
    # Keep multi-seed runs reproducible even when the LLM uses sampling.
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort determinism; some kernels may still be nondeterministic.
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            from transformers import set_seed  # type: ignore

            set_seed(seed)
        except Exception:
            pass
    except Exception:
        pass


def get_precheck_index_count(config: Dict) -> int | None:
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


def build_seed_questions(all_questions: List[Dict], seed: int, limit: int) -> List[Dict]:
    questions = list(all_questions)
    if limit:
        random.Random(seed).shuffle(questions)
        questions = questions[:limit]
    return questions


def run_eval(
    rag: RAGPipeline,
    questions: List[Dict],
    abstain_message: str,
    gating_override: Dict[str, float] | None,
) -> Dict:
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

    return {
        "total": total,
        "abstain": abstain,
        "abstain_rate": (abstain / total) if total else 0.0,
        "actions": dict(action_counts),
        "stats_all": summarize(buckets["all"]),
        "stats_answered": summarize(buckets["answered"]),
        "stats_abstain": summarize(buckets["abstain"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Proxy grounding eval (multi-seed, single pipeline)")
    parser.add_argument("--config", required=True, help="Config name (without .yaml)")
    parser.add_argument("--questions", required=True, help="Path to JSONL questions")
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds, e.g. 7,11,19")
    parser.add_argument("--limit", type=int, default=0, help="Optional per-seed limit")
    parser.add_argument(
        "--output-pattern",
        required=True,
        help="Output pattern containing {seed}, e.g. out/domain_seed{seed}.json",
    )
    parser.add_argument("--skip-existing", action="store_true")
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

    if "{seed}" not in args.output_pattern:
        raise SystemExit("--output-pattern must contain '{seed}' placeholder")

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

    all_questions = load_questions(Path(args.questions))
    abstain_message = (config.get("gating", {}).get("abstain_message", "").strip())

    gating_override: Dict[str, float] = {}
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

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise SystemExit("No valid seeds found in --seeds")

    for seed in seeds:
        seed_everything(seed)
        out_path = Path(args.output_pattern.format(seed=seed))
        if args.skip_existing and out_path.exists():
            print(f"Skipping existing: {out_path}")
            continue

        seed_questions = build_seed_questions(all_questions, seed, args.limit)
        summary = run_eval(
            rag=rag,
            questions=seed_questions,
            abstain_message=abstain_message,
            gating_override=gating_override if gating_override else None,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[seed={seed}] wrote {out_path}")


if __name__ == "__main__":
    main()
