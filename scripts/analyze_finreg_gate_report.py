#!/usr/bin/env python3
"""Analyze saved FinReg full-RAG gate reports without rerunning generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def risk_bucket(value: float) -> str:
    if value >= 0.97:
        return ">=0.97"
    if value >= 0.90:
        return "0.90-0.97"
    if value >= 0.75:
        return "0.75-0.90"
    if value >= 0.50:
        return "0.50-0.75"
    return "<0.50"


def summarize(rows: list[dict[str, Any]], simulated_thresholds: list[float]) -> dict[str, Any]:
    total = len(rows)
    answered = [row for row in rows if not bool(row.get("abstained"))]
    abstained = [row for row in rows if bool(row.get("abstained"))]
    mismatches = [row for row in rows if row.get("expected_behavior_match") is False]
    forbidden_hits = [
        row for row in rows
        if int(row.get("forbidden_claim_hit_count") or 0) > 0
    ]

    by_type = Counter(str(row.get("question_type") or "unknown") for row in rows)
    action_counts = Counter(str(row.get("gating_action") or "none") for row in rows)
    bucket_counts = Counter(
        risk_bucket(as_float((row.get("gating_stats") or {}).get("answer_include_risk")))
        for row in rows
    )

    high_risk_answered = []
    for row in answered:
        stats = row.get("gating_stats") or {}
        risk = as_float(stats.get("answer_include_risk"))
        if risk >= 0.75 or row.get("expected_behavior_match") is False:
            high_risk_answered.append({
                "id": row.get("id"),
                "question_type": row.get("question_type"),
                "expected_behavior": row.get("expected_behavior"),
                "expected_behavior_match": row.get("expected_behavior_match"),
                "answer_include_risk": risk,
                "answer_include_score": as_float(stats.get("answer_include_score")),
                "uncertainty_mean": as_float(stats.get("uncertainty_mean")),
                "combined_conflict": as_float(stats.get("combined_conflict")),
                "retrieval_max_score": as_float(stats.get("retrieval_max_score")),
                "answer_preview": " ".join(str(row.get("answer") or "").split())[:220],
            })

    threshold_views = {}
    for threshold in simulated_thresholds:
        would_abstain = [
            row for row in rows
            if as_float((row.get("gating_stats") or {}).get("answer_include_risk")) >= threshold
        ]
        newly_abstained_correct = [
            row for row in would_abstain
            if not bool(row.get("abstained")) and row.get("expected_behavior_match") is True
        ]
        newly_abstained_bad = [
            row for row in would_abstain
            if not bool(row.get("abstained")) and row.get("expected_behavior_match") is False
        ]
        threshold_views[str(threshold)] = {
            "would_abstain_count": len(would_abstain),
            "would_abstain_rate": pct(len(would_abstain), total),
            "newly_abstained_answered_correct": [row.get("id") for row in newly_abstained_correct],
            "newly_abstained_answered_bad": [row.get("id") for row in newly_abstained_bad],
        }

    return {
        "total": total,
        "answered_count": len(answered),
        "answered_rate": pct(len(answered), total),
        "abstain_count": len(abstained),
        "abstain_rate": pct(len(abstained), total),
        "mismatch_count": len(mismatches),
        "mismatch_ids": [row.get("id") for row in mismatches],
        "forbidden_claim_hit_count": len(forbidden_hits),
        "forbidden_claim_hit_ids": [row.get("id") for row in forbidden_hits],
        "action_counts": dict(action_counts),
        "question_type_counts": dict(by_type),
        "answer_include_risk_buckets": dict(bucket_counts),
        "high_risk_answered": high_risk_answered,
        "threshold_simulation": threshold_views,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize gate behavior from a saved FinReg per_question.jsonl report."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to run_finreg_real_life_benchmark.py per_question.jsonl output.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.75,0.85,0.90,0.95,0.97",
        help="Comma-separated answer_include_risk thresholds to simulate.",
    )
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    thresholds = [
        float(item.strip())
        for item in args.thresholds.split(",")
        if item.strip()
    ]
    summary = summarize(rows, thresholds)

    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
