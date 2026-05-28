#!/usr/bin/env python3
"""Shadow-evaluate evidence subset sampling for FinReg RAG gating.

This script does not change production gate behavior. It runs the normal RAG
pipeline, keeps the pre-gating candidate answer, then rechecks that answer
against multiple sampled evidence subsets. The goal is to measure whether the
gate decision is stable under plausible retrieval/evidence variation.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.core.config_loader import load_config
from src.rag.rag_pipeline import MODEL_ABSTAIN_PHRASES, RAGPipeline


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def seed_everything(seed: int) -> None:
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
    except Exception:
        pass


def is_model_abstain(answer: str) -> bool:
    lower = (answer or "").strip().lower()
    return any(phrase in lower for phrase in MODEL_ABSTAIN_PHRASES)


def doc_key(doc: dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    source = (
        metadata.get("doc_id")
        or metadata.get("title")
        or metadata.get("source")
        or metadata.get("path")
        or ""
    )
    content = str(doc.get("content") or "")
    return f"{source}|{content[:200]}"


def dedupe_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for doc in docs:
        key = doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def build_evidence_subsets(
    rag: RAGPipeline,
    query: str,
    docs: list[dict[str, Any]],
    subset_size: int,
    num_subsets: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Build deterministic plus sampled evidence subsets from ranked docs."""
    docs = dedupe_docs(docs)
    if not docs:
        return []

    subset_size = max(1, min(subset_size, len(docs)))
    subsets: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def add(name: str, selected: list[dict[str, Any]]) -> None:
        selected = dedupe_docs(selected)[:subset_size]
        if not selected:
            return
        key = tuple(doc_key(doc) for doc in selected)
        if key in seen:
            return
        seen.add(key)
        subsets.append({"name": name, "docs": selected})

    add("top_ranked", docs[:subset_size])

    stride_docs = docs[::2] + docs[1::2]
    add("rank_stride", stride_docs[:subset_size])

    tail_mix = docs[: max(1, subset_size // 2)] + docs[-max(1, subset_size - subset_size // 2) :]
    add("top_tail_mix", tail_mix)

    named_families = rag._extract_named_source_families(query)  # analysis helper
    if named_families:
        family_docs: list[dict[str, Any]] = []
        used_ids: set[int] = set()
        for family in named_families:
            candidates = [doc for doc in docs if rag._doc_source_family(doc) == family]
            if candidates:
                chosen = candidates[0]
                family_docs.append(chosen)
                used_ids.add(id(chosen))
        for doc in docs:
            if len(family_docs) >= subset_size:
                break
            if id(doc) in used_ids:
                continue
            family_docs.append(doc)
        add("named_family_balanced", family_docs)

    attempts = 0
    while len(subsets) < num_subsets and attempts < num_subsets * 10:
        attempts += 1
        if len(docs) <= subset_size:
            add(f"sample_{attempts}", docs)
            continue
        # Bias toward high-ranked evidence while still allowing perturbation.
        head_count = max(1, subset_size // 2)
        head = docs[: min(len(docs), max(subset_size, head_count + 1))]
        tail = docs[min(len(docs), head_count) :]
        selected = rng.sample(head, min(head_count, len(head)))
        remaining = subset_size - len(selected)
        pool = [doc for doc in tail if id(doc) not in {id(item) for item in selected}]
        if len(pool) >= remaining:
            selected.extend(rng.sample(pool, remaining))
        else:
            selected.extend(pool)
            selected.extend(
                doc for doc in docs if id(doc) not in {id(item) for item in selected}
            )
        add(f"sample_{attempts}", selected)

    return subsets[:num_subsets]


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def subset_source_families(rag: RAGPipeline, docs: list[dict[str, Any]]) -> list[str]:
    families: list[str] = []
    for doc in docs:
        family = rag._doc_source_family(doc)
        if family and family not in families:
            families.append(family)
    return families


def evaluate_subset(
    rag: RAGPipeline,
    query: str,
    answer: str,
    docs: list[dict[str, Any]],
    aggregation: str,
    gating_config: dict[str, Any],
) -> dict[str, Any]:
    contexts = [str(doc.get("content") or "") for doc in docs if doc.get("content")]
    detection = rag._run_cuda_stage(
        "evidence_subset_detector",
        rag.hallucination_detector.verify_answer_with_contexts,
        answer=answer,
        contexts=contexts,
        aggregation=aggregation,
        question=query,
    )
    retrieval_stats = rag._compute_retrieval_stats(docs)
    source_consistency = rag._compute_source_consistency(
        docs,
        gating_config.get("source_consistency_top_k"),
    )
    stats = {
        "retrieval_max_score": retrieval_stats.get("max_score", 0.0),
        "retrieval_mean_score": retrieval_stats.get("mean_score", 0.0),
        "source_consistency": source_consistency,
        "source_inconsistency": (
            rag._clamp01(1.0 - float(source_consistency))
            if isinstance(source_consistency, (int, float))
            else 0.0
        ),
    }
    stats.update(
        rag._compute_uncertainty_stats(
            detection,
            gating_config.get("uncertainty_source"),
        )
    )
    stats.update(rag._quality_stats(None))
    stats["combined_conflict"] = rag._clamp01(
        0.45 * float(
            stats.get(
                "detector_conflict_consensus",
                stats.get("detector_conflict", 0.0),
            )
            or 0.0
        )
        + 0.35 * float(stats.get("source_inconsistency", 0.0) or 0.0)
        + 0.20
        * max(
            0.0,
            float(stats.get("retrieval_max_score", 0.0) or 0.0)
            - float(stats.get("retrieval_mean_score", 0.0) or 0.0),
        )
    )
    action = rag._decide_gating_action(stats, gating_config)
    return {
        "action": action,
        "stats": stats,
        "hallucination_detected": detection.get("is_hallucination"),
        "answer_include_detected": detection.get("answer_include_detected"),
        "num_contexts": len(contexts),
    }


def summarize_question(
    subset_results: list[dict[str, Any]],
    baseline_action: str,
) -> dict[str, Any]:
    actions = Counter(str(item.get("action") or "none") for item in subset_results)
    total = len(subset_results)
    max_action_count = max(actions.values()) if actions else 0
    action_instability = 1.0 - (max_action_count / total) if total else 0.0
    non_answer_count = actions["retrieve_more"] + actions["abstain"]

    contradiction_probs = [
        float(item["stats"].get("contradiction_prob_mean", 0.0) or 0.0)
        for item in subset_results
    ]
    contradiction_rates = [
        float(item["stats"].get("contradiction_rate", 0.0) or 0.0)
        for item in subset_results
    ]
    unsupported_risks = [
        float(item["stats"].get("answer_include_risk", item["stats"].get("unsupported_risk", 0.0)) or 0.0)
        for item in subset_results
    ]

    return {
        "subset_count": total,
        "subset_actions": dict(actions),
        "subset_answer_rate": actions["none"] / total if total else 0.0,
        "subset_retrieve_more_rate": actions["retrieve_more"] / total if total else 0.0,
        "subset_abstain_rate": actions["abstain"] / total if total else 0.0,
        "subset_non_answer_rate": non_answer_count / total if total else 0.0,
        "subset_action_instability": action_instability,
        "baseline_action": baseline_action,
        "baseline_action_stability": (
            actions[baseline_action] / total if total and baseline_action in actions else 0.0
        ),
        "contradiction_prob_mean_across_subsets": mean(contradiction_probs),
        "contradiction_rate_mean_across_subsets": mean(contradiction_rates),
        "answer_include_risk_mean_across_subsets": mean(unsupported_risks),
        "answer_include_risk_max_across_subsets": max(unsupported_risks) if unsupported_risks else None,
    }


def build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate_actions: Counter[str] = Counter()
    aggregate_subset_actions: Counter[str] = Counter()
    instabilities: list[float] = []
    baseline_stabilities: list[float] = []

    for row in rows:
        baseline_action = str(row.get("baseline_action") or "none")
        aggregate_actions[baseline_action] += 1
        for action, count in (row.get("subset_actions") or {}).items():
            aggregate_subset_actions[str(action)] += int(count)
        if row.get("subset_action_instability") is not None:
            instabilities.append(float(row["subset_action_instability"]))
        if row.get("baseline_action_stability") is not None:
            baseline_stabilities.append(float(row["baseline_action_stability"]))

    return {
        "total": len(rows),
        "config": args.config,
        "questions": args.questions,
        "seed": args.seed,
        "subset_size": args.subset_size,
        "num_subsets": args.num_subsets,
        "baseline_actions": dict(aggregate_actions),
        "subset_actions": dict(aggregate_subset_actions),
        "mean_subset_action_instability": mean(instabilities),
        "mean_baseline_action_stability": mean(baseline_stabilities),
        "rows": rows,
    }


def write_summary(output: Path, summary: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    tmp_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_output.replace(output)


def row_key(item: dict[str, Any]) -> str:
    return str(item.get("id") or item.get("query") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow evidence subset sampling eval for FinReg.")
    parser.add_argument("--config", required=True, help="Config name without .yaml.")
    parser.add_argument("--questions", required=True, help="Question JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--subset-size", type=int, default=5)
    parser.add_argument("--num-subsets", type=int, default=6)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--include-subset-details", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output JSON.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write partial output every N rows.")
    args = parser.parse_args()

    output = Path(args.output)
    rows: list[dict[str, Any]] = []
    processed_keys: set[str] = set()
    if args.resume and output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        rows = list(existing.get("rows") or [])
        processed_keys = {
            row_key({"id": row.get("id"), "query": row.get("query")})
            for row in rows
        }

    seed_everything(args.seed)
    config = load_config(args.config)
    rag = RAGPipeline.from_config(config)
    if not rag.hallucination_detector:
        raise SystemExit("Hallucination detector is required for evidence subset sampling.")

    questions = read_jsonl(Path(args.questions))
    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    gating_config = rag._merge_gating_config(None)
    aggregation = (
        config.get("hallucination_aggregation")
        or (config.get("hallucination_detector", {}) or {}).get("aggregation")
        or rag.hallucination_aggregation
        or "any"
    )
    rng = random.Random(args.seed)

    for item_index, item in enumerate(questions, start=1):
        key = row_key(item)
        if key in processed_keys:
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        result = rag.query(
            query_text=query,
            k=args.k or None,
            return_context=True,
            detect_hallucinations=True,
            hallucination_aggregation=aggregation,
        )
        baseline_action = str((result.get("gating") or {}).get("action") or "none")

        candidate_answer = str(result.get("pre_gating_answer") or result.get("answer") or "").strip()
        docs = list(result.get("context") or [])
        if not candidate_answer or is_model_abstain(candidate_answer) or not docs:
            row = {
                "id": item.get("id"),
                "type": item.get("type"),
                "query": query,
                "baseline_action": baseline_action,
                "candidate_unavailable": True,
                "num_docs": len(docs),
            }
            rows.append(row)
            processed_keys.add(key)
            if args.checkpoint_every > 0 and len(rows) % args.checkpoint_every == 0:
                write_summary(output, build_summary(args, rows))
            continue

        subsets = build_evidence_subsets(
            rag=rag,
            query=query,
            docs=docs,
            subset_size=args.subset_size,
            num_subsets=args.num_subsets,
            rng=rng,
        )
        subset_rows: list[dict[str, Any]] = []
        for subset in subsets:
            evaluated = evaluate_subset(
                rag=rag,
                query=query,
                answer=candidate_answer,
                docs=subset["docs"],
                aggregation=aggregation,
                gating_config=gating_config,
            )
            subset_row = {
                "name": subset["name"],
                "action": evaluated["action"],
                "families": subset_source_families(rag, subset["docs"]),
                "num_contexts": evaluated["num_contexts"],
                "stats": {
                    key: evaluated["stats"].get(key)
                    for key in (
                        "contradiction_rate",
                        "contradiction_prob_mean",
                        "uncertainty_mean",
                        "source_consistency",
                        "retrieval_max_score",
                        "retrieval_mean_score",
                        "answer_include_risk",
                        "support_score",
                        "label_disagreement",
                        "neutral_prob_mean",
                    )
                },
            }
            subset_rows.append(subset_row)

        q_summary = summarize_question(subset_rows, baseline_action)
        row = {
            "id": item.get("id"),
            "type": item.get("type"),
            "query": query,
            "baseline_action": baseline_action,
            "baseline_attempts": (result.get("gating") or {}).get("attempts"),
            "baseline_k_used": (result.get("gating") or {}).get("k_used"),
            "num_docs": len(docs),
            **q_summary,
        }
        if args.include_subset_details:
            row["subsets"] = subset_rows
        rows.append(row)
        processed_keys.add(key)
        if args.checkpoint_every > 0 and len(rows) % args.checkpoint_every == 0:
            write_summary(output, build_summary(args, rows))

    summary = build_summary(args, rows)
    write_summary(output, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
