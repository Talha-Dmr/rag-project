#!/usr/bin/env python3
"""
Compare two detector/gate configs on the same question slice.

Goal:
- measure whether detector-side changes propagate into gate actions
- identify when stat deltas are suppressed by the current gate policy
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


DETECTOR_STAT_KEYS = (
    "contradiction_rate",
    "contradiction_prob_mean",
    "uncertainty_mean",
    "entailment_prob_mean",
    "neutral_prob_mean",
    "contradiction_prob_std",
    "conflict_mass_mean",
    "label_disagreement",
    "confidence_mean",
    "top2_margin_mean",
    "contradiction_neutral_gap_mean",
    "entailment_neutral_gap_mean",
)


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


def load_questions(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_pipeline(config_name: str, llm_temperature: float | None, max_retries: int | None) -> Tuple[Dict[str, Any], RAGPipeline]:
    config = load_config(config_name)
    if llm_temperature is not None:
        llm_cfg = config.get("llm") or {}
        if not isinstance(llm_cfg, dict):
            llm_cfg = {}
        llm_cfg["temperature"] = float(llm_temperature)
        config["llm"] = llm_cfg

    if max_retries is not None:
        gating_cfg = config.get("gating") or {}
        if not isinstance(gating_cfg, dict):
            gating_cfg = {}
        gating_cfg["max_retries"] = int(max_retries)
        config["gating"] = gating_cfg

    rag = RAGPipeline.from_config(config)
    count = rag.vector_store.get_count()
    if count == 0:
        raise SystemExit(f"Vector collection is empty for config '{config_name}'")
    return config, rag


def build_run_view(result: Dict[str, Any], abstain_message: str) -> Dict[str, Any]:
    gating = result.get("gating") or {}
    stats = dict((gating.get("stats") or {}))
    answer = (result.get("answer") or "").strip()
    return {
        "action": gating.get("action", "none"),
        "attempts": int(gating.get("attempts", 0) or 0),
        "k_used": gating.get("k_used"),
        "answer_preview": answer.replace("\n", " ").strip()[:240],
        "is_abstain": bool(abstain_message and answer == abstain_message),
        "stats": stats,
        "thresholds": dict(gating.get("thresholds") or {}),
        "hallucination_detected": result.get("hallucination_detected"),
        "hallucination_error": result.get("hallucination_error"),
    }


def evaluate_config(
    config_name: str,
    questions: List[Dict[str, Any]],
    llm_temperature: float | None,
    max_retries: int | None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    config, rag = load_pipeline(config_name, llm_temperature, max_retries)
    abstain_message = str((config.get("gating") or {}).get("abstain_message", "")).strip()
    rows: List[Dict[str, Any]] = []
    for item in questions:
        query = item.get("query", "")
        if not query:
            continue
        result = rag.query(query_text=query, return_context=False, detect_hallucinations=True)
        rows.append({
            "query": query,
            "run": build_run_view(result, abstain_message),
        })
    # Best-effort cleanup so the second config can reuse GPU memory.
    try:
        import gc
        import torch  # type: ignore

        del rag
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return config, rows


def compute_trigger_state(stats: Dict[str, Any], thresholds: Dict[str, Any]) -> Dict[str, bool]:
    contradiction_rate = _to_float(stats.get("contradiction_rate"))
    contradiction_prob = _to_float(stats.get("contradiction_prob_mean"))
    uncertainty = _to_float(stats.get("uncertainty_mean"))
    source_consistency = stats.get("source_consistency")
    retrieval_max = _to_float(stats.get("retrieval_max_score"), 1.0)
    retrieval_mean = _to_float(stats.get("retrieval_mean_score"), 1.0)

    cr_thr = thresholds.get("contradiction_rate")
    cp_thr = thresholds.get("contradiction_prob")
    u_thr = thresholds.get("uncertainty")
    sc_thr = thresholds.get("source_consistency")

    cr = isinstance(cr_thr, (int, float)) and contradiction_rate >= float(cr_thr)
    cp = isinstance(cp_thr, (int, float)) and contradiction_prob >= float(cp_thr)
    unc = isinstance(u_thr, (int, float)) and uncertainty >= float(u_thr)
    sc = isinstance(sc_thr, (int, float)) and source_consistency is not None and float(source_consistency) < float(sc_thr)

    rm_thr = thresholds.get("min_retrieval_score")
    rmean_thr = thresholds.get("min_mean_retrieval_score")
    retrieval_low = False
    if isinstance(rm_thr, (int, float)) and retrieval_max < float(rm_thr):
        retrieval_low = True
    if isinstance(rmean_thr, (int, float)) and retrieval_mean < float(rmean_thr):
        retrieval_low = True
    retrieval_low = retrieval_low or sc

    detector_trigger = bool(cr or cp or unc)
    overall_trigger = bool(detector_trigger or retrieval_low)
    return {
        "contradiction_rate": bool(cr),
        "contradiction_prob": bool(cp),
        "uncertainty": bool(unc),
        "source_consistency": bool(sc),
        "retrieval_low": bool(retrieval_low),
        "detector_trigger": detector_trigger,
        "overall_trigger": overall_trigger,
    }


def near_threshold_metrics(
    stats: Dict[str, Any],
    thresholds: Dict[str, Any],
    rel_margin: float,
    abs_margin: float,
) -> List[str]:
    metric_to_threshold = {
        "contradiction_rate": thresholds.get("contradiction_rate"),
        "contradiction_prob_mean": thresholds.get("contradiction_prob"),
        "uncertainty_mean": thresholds.get("uncertainty"),
    }
    near: List[str] = []
    for metric, thr in metric_to_threshold.items():
        if not isinstance(thr, (int, float)):
            continue
        value = _to_float(stats.get(metric))
        tol = max(abs_margin, abs(float(thr)) * rel_margin)
        if abs(value - float(thr)) <= tol:
            near.append(metric)
    return near


def stats_delta(
    stats_a: Dict[str, Any],
    stats_b: Dict[str, Any],
    epsilon: float,
) -> Tuple[Dict[str, float], List[str]]:
    deltas: Dict[str, float] = {}
    changed: List[str] = []
    for key in DETECTOR_STAT_KEYS:
        delta = abs(_to_float(stats_a.get(key)) - _to_float(stats_b.get(key)))
        deltas[key] = delta
        if delta > epsilon:
            changed.append(key)
    return deltas, changed


def summarize_examples(rows: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        out.append({
            "query": row["query"],
            "config_a_action": row["config_a"]["action"],
            "config_b_action": row["config_b"]["action"],
            "changed_stats": row["comparison"]["changed_stats"],
            "near_threshold_metrics": row["comparison"]["near_threshold_metrics"],
            "suppression_reason": row["comparison"].get("suppression_reason"),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare detector->gate sensitivity for two configs")
    parser.add_argument("--config-a", required=True, help="Baseline config name (without .yaml)")
    parser.add_argument("--config-b", required=True, help="Candidate config name (without .yaml)")
    parser.add_argument("--questions", required=True, help="Path to JSONL questions")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of questions")
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="Force deterministic-ish decoding")
    parser.add_argument("--max-retries-override", type=int, default=None, help="Optional gating max_retries override")
    parser.add_argument("--delta-epsilon", type=float, default=1e-4, help="Minimum stat delta to count as changed")
    parser.add_argument("--near-threshold-rel-margin", type=float, default=0.15)
    parser.add_argument("--near-threshold-abs-margin", type=float, default=0.01)
    parser.add_argument("--output", default="", help="Optional JSON summary path")
    parser.add_argument("--details-output", default="", help="Optional JSONL per-question output path")
    args = parser.parse_args()

    seed_everything(int(args.seed))
    questions = load_questions(Path(args.questions))
    if args.limit:
        random.Random(args.seed).shuffle(questions)
        questions = questions[: args.limit]

    config_a, runs_a = evaluate_config(
        args.config_a,
        questions,
        args.llm_temperature,
        args.max_retries_override,
    )
    config_b, runs_b = evaluate_config(
        args.config_b,
        questions,
        args.llm_temperature,
        args.max_retries_override,
    )

    details_path = Path(args.details_output) if args.details_output else None
    details_handle = None
    if details_path is not None:
        details_path.parent.mkdir(parents=True, exist_ok=True)
        details_handle = details_path.open("w", encoding="utf-8")

    same_stats_same_action = 0
    diff_stats_same_action = 0
    diff_action = 0
    threshold_near_examples = 0
    detector_trigger_changed = 0
    suppression_reasons = Counter()
    action_counts_a = Counter()
    action_counts_b = Counter()
    delta_sums = defaultdict(float)
    rows_by_bucket = defaultdict(list)

    for row_a, row_b in zip(runs_a, runs_b):
        query = row_a["query"]
        if row_b["query"] != query:
            raise RuntimeError("Question ordering diverged between config evaluations.")

        run_a = row_a["run"]
        run_b = row_b["run"]
        action_counts_a[run_a["action"]] += 1
        action_counts_b[run_b["action"]] += 1

        deltas, changed_stats = stats_delta(run_a["stats"], run_b["stats"], args.delta_epsilon)
        for key, delta in deltas.items():
            delta_sums[key] += delta

        trigger_a = compute_trigger_state(run_a["stats"], run_a["thresholds"])
        trigger_b = compute_trigger_state(run_b["stats"], run_b["thresholds"])
        if trigger_a["detector_trigger"] != trigger_b["detector_trigger"]:
            detector_trigger_changed += 1

        near_a = near_threshold_metrics(
            run_a["stats"],
            run_a["thresholds"],
            rel_margin=args.near_threshold_rel_margin,
            abs_margin=args.near_threshold_abs_margin,
        )
        near_b = near_threshold_metrics(
            run_b["stats"],
            run_b["thresholds"],
            rel_margin=args.near_threshold_rel_margin,
            abs_margin=args.near_threshold_abs_margin,
        )
        near_metrics = sorted(set(near_a + near_b))
        if near_metrics:
            threshold_near_examples += 1

        same_action = (
            run_a["action"] == run_b["action"]
            and run_a["is_abstain"] == run_b["is_abstain"]
        )
        suppression_reason = None
        if not changed_stats and same_action:
            bucket = "same_stats_same_action"
            same_stats_same_action += 1
        elif same_action:
            bucket = "diff_stats_same_action"
            diff_stats_same_action += 1
            if trigger_a["overall_trigger"] == trigger_b["overall_trigger"]:
                if trigger_a["retrieval_low"] or trigger_b["retrieval_low"]:
                    suppression_reason = "retrieval_or_source_trigger_unchanged"
                elif trigger_a["detector_trigger"] == trigger_b["detector_trigger"]:
                    suppression_reason = "same_detector_trigger_region"
                else:
                    suppression_reason = "same_final_action_after_threshold_crossing"
            else:
                suppression_reason = "action_collapse_after_trigger_change"
        else:
            bucket = "diff_action"
            diff_action += 1

        if suppression_reason:
            suppression_reasons[suppression_reason] += 1

        row = {
            "query": query,
            "config_a": {
                **run_a,
                "trigger_state": trigger_a,
            },
            "config_b": {
                **run_b,
                "trigger_state": trigger_b,
            },
            "comparison": {
                "bucket": bucket,
                "changed_stats": changed_stats,
                "stat_deltas": deltas,
                "near_threshold_metrics": near_metrics,
                "detector_trigger_changed": bool(trigger_a["detector_trigger"] != trigger_b["detector_trigger"]),
                "suppression_reason": suppression_reason,
            },
        }
        rows_by_bucket[bucket].append(row)
        if details_handle is not None:
            details_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = same_stats_same_action + diff_stats_same_action + diff_action
    summary = {
        "config_a": args.config_a,
        "config_b": args.config_b,
        "seed": args.seed,
        "limit": args.limit,
        "total": total,
        "action_counts_a": dict(action_counts_a),
        "action_counts_b": dict(action_counts_b),
        "same_stats_same_action": same_stats_same_action,
        "same_stats_same_action_rate": (same_stats_same_action / total) if total else 0.0,
        "diff_stats_same_action": diff_stats_same_action,
        "diff_stats_same_action_rate": (diff_stats_same_action / total) if total else 0.0,
        "diff_action": diff_action,
        "diff_action_rate": (diff_action / total) if total else 0.0,
        "detector_trigger_changed": detector_trigger_changed,
        "detector_trigger_changed_rate": (detector_trigger_changed / total) if total else 0.0,
        "threshold_near_examples": threshold_near_examples,
        "threshold_near_examples_rate": (threshold_near_examples / total) if total else 0.0,
        "mean_abs_detector_stat_delta": {
            key: (delta_sums[key] / total) if total else 0.0
            for key in DETECTOR_STAT_KEYS
        },
        "suppression_reasons": dict(suppression_reasons),
        "examples": {
            "diff_action": summarize_examples(rows_by_bucket["diff_action"]),
            "diff_stats_same_action": summarize_examples(rows_by_bucket["diff_stats_same_action"]),
            "threshold_near": summarize_examples(
                [row for bucket_rows in rows_by_bucket.values() for row in bucket_rows if row["comparison"]["near_threshold_metrics"]]
            ),
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote summary to {out_path}")
    if details_handle is not None:
        details_handle.close()
        print(f"Wrote per-question details to {details_path}")


if __name__ == "__main__":
    main()
