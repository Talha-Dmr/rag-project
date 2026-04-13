#!/usr/bin/env python3
"""
Mine a hard-risk watchlist from full detector outputs.

Unlike the general expansion miner, this script is intentionally aggressive:
- prioritizes high contradiction probability / conflict / hallucination scores
- keeps abstention-like rows because they are useful contradiction/unsupported review targets
- outputs a compact CSV for manual review
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard-risk watchlist for Phase 2")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant input in the form alias=path/to/per_question.jsonl",
    )
    parser.add_argument(
        "--existing-reviewed-csv",
        default="evaluation_results/finreg_detector_phase2_priority_review_smoke/reviewer_notes_draft.csv",
        help="Existing reviewed CSV used to exclude already-covered ids",
    )
    parser.add_argument(
        "--output-csv",
        default="evaluation_results/finreg_detector_phase2_hard_risk_watchlist.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=24,
        help="Maximum number of rows to emit",
    )
    return parser.parse_args()


def resolve_variant_specs(raw_specs: List[str]) -> List[Tuple[str, Path]]:
    resolved: List[Tuple[str, Path]] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise SystemExit(f"Invalid --variant value: {raw}")
        alias, path_str = raw.split("=", 1)
        resolved.append((alias.strip(), Path(path_str.strip())))
    if not resolved:
        raise SystemExit("At least one --variant alias=path is required.")
    return resolved


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def classify_watch(row: Dict[str, Any]) -> Tuple[str, str]:
    cp = safe_float(row.get("contradiction_prob_mean"))
    hp = safe_float(row.get("hallucination_prob_topk"))
    dc = safe_float(row.get("detector_conflict"))
    answer = (row.get("generated_answer") or "").lower()

    if "human resources department" in answer or "project manager" in answer:
        return "likely_unsupported", "generation_contamination"
    if cp >= 0.14 or hp >= 0.14:
        return "contradiction_watch", "high_contradiction_probability"
    if cp >= 0.10 or dc >= 0.08:
        return "unsupported_or_ambiguous_watch", "elevated_conflict_signal"
    return "risk_watch", "moderate_risk_signal"


def main() -> None:
    args = parse_args()
    variant_specs = resolve_variant_specs(args.variant)
    reviewed_rows = load_csv(Path(args.existing_reviewed_csv))
    exclude_ids = {row.get("id", "") for row in reviewed_rows if row.get("id")}

    watch_rows: List[Dict[str, Any]] = []
    for alias, path in variant_specs:
        for row in load_jsonl(path):
            record_id = row.get("id", "")
            if record_id in exclude_ids:
                continue
            cp = safe_float(row.get("contradiction_prob_mean"))
            hp = safe_float(row.get("hallucination_prob_topk"))
            dc = safe_float(row.get("detector_conflict"))
            if max(cp, hp, dc) < 0.04:
                continue
            watch_type, risk_reason = classify_watch(row)
            watch_rows.append({
                "id": record_id,
                "detector_variant": alias,
                "question": row.get("question", ""),
                "predicted_detector_label": row.get("predicted_detector_label", ""),
                "contradiction_prob_mean": cp,
                "hallucination_prob_topk": hp,
                "detector_conflict": dc,
                "watch_type": watch_type,
                "risk_reason": risk_reason,
                "review_target": "prioritize_for_contradicted_or_unsupported_review",
                "gold_label": "",
                "gold_error_type": "",
                "review_notes": "",
            })

    watch_rows.sort(
        key=lambda row: (
            -float(row["contradiction_prob_mean"]),
            -float(row["hallucination_prob_topk"]),
            -float(row["detector_conflict"]),
        )
    )
    selected = watch_rows[: args.limit]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()) if selected else [
            "id", "watch_type", "risk_reason"
        ])
        writer.writeheader()
        writer.writerows(selected)

    print(f"Wrote {len(selected)} hard-risk watchlist rows to {output_path}")


if __name__ == "__main__":
    main()
