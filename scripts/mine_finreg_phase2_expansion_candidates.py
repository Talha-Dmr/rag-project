#!/usr/bin/env python3
"""
Mine additional supported/contradicted review candidates from full detector outputs.

Purpose:
- expand the reviewed gold seed beyond the initial p0/p1 subset
- prioritize low-risk supported controls and high-risk contradiction candidates
- produce a review CSV, not final gold labels
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine Phase 2 expansion candidates")
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
        default="evaluation_results/finreg_detector_phase2_expansion_candidates.csv",
        help="Output candidate CSV",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=48,
        help="Maximum number of additional candidates to emit",
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


def classify_candidate(row: Dict[str, Any]) -> Tuple[str, str, str]:
    cp = safe_float(row.get("contradiction_prob_mean"))
    hp = safe_float(row.get("hallucination_prob_topk"))
    dc = safe_float(row.get("detector_conflict"))
    pred = (row.get("predicted_detector_label") or "").strip().lower()
    answer = (row.get("generated_answer") or "").lower()

    contamination = any(marker in answer for marker in (
        "human resources department",
        "project manager",
        "job title:",
        "essential duties:",
    ))
    if contamination:
        return "unsupported", "fabricated_fact", "p1"

    if cp >= 0.12 or hp >= 0.12 or pred == "contradiction":
        return "contradicted", "misinterpretation", "p1"

    if dc >= 0.10:
        return "partial", "incomplete_reasoning", "p2"

    if cp <= 0.03 and hp <= 0.03 and dc <= 0.02 and pred in {"entailment", "neutral"}:
        return "supported", "none", "p2"

    return "ambiguous", "cross_document_conflict", "p3"


def build_candidate(alias: str, row: Dict[str, Any]) -> Dict[str, Any]:
    suggested_label, suggested_error_type, review_priority = classify_candidate(row)
    cp = safe_float(row.get("contradiction_prob_mean"))
    hp = safe_float(row.get("hallucination_prob_topk"))
    dc = safe_float(row.get("detector_conflict"))
    review_reason = (
        f"mined_from_full_outputs; detector_pred={row.get('predicted_detector_label','')}; "
        f"cp={cp:.3f}; hp={hp:.3f}; dc={dc:.3f}"
    )
    return {
        "id": row.get("id", ""),
        "question_id": row.get("question_id", ""),
        "detector_variant": alias,
        "bucket": "expansion_mined",
        "question": row.get("question", ""),
        "generated_answer": row.get("generated_answer", ""),
        "suggested_label": suggested_label,
        "suggested_error_type": suggested_error_type,
        "review_priority": review_priority,
        "review_reason": review_reason,
        "proposed_slice": "",
        "predicted_detector_label": row.get("predicted_detector_label", ""),
        "contradiction_prob_mean": cp,
        "hallucination_prob_topk": hp,
        "detector_conflict": dc,
        "retrieval_max_score": safe_float(row.get("retrieval_max_score")),
        "retrieval_mean_score": safe_float(row.get("retrieval_mean_score")),
        "gold_label": "",
        "gold_error_type": "",
        "review_notes": "",
    }


def main() -> None:
    args = parse_args()
    variant_specs = resolve_variant_specs(args.variant)
    reviewed_rows = load_csv(Path(args.existing_reviewed_csv))
    exclude_ids = {row.get("id", "") for row in reviewed_rows if row.get("id")}

    candidates: List[Dict[str, Any]] = []
    for alias, path in variant_specs:
        for row in load_jsonl(path):
            record_id = row.get("id", "")
            if record_id in exclude_ids:
                continue
            candidates.append(build_candidate(alias, row))

    priority_order = {"p1": 0, "p2": 1, "p3": 2}
    label_order = {"contradicted": 0, "supported": 1, "partial": 2, "ambiguous": 3, "unsupported": 4}
    candidates.sort(
        key=lambda row: (
            priority_order.get(row["review_priority"], 9),
            label_order.get(row["suggested_label"], 9),
            -float(row["contradiction_prob_mean"]),
            -float(row["detector_conflict"]),
        )
    )

    selected = candidates[: args.target_total]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0].keys()) if selected else [
            "id", "suggested_label", "review_priority"
        ])
        writer.writeheader()
        writer.writerows(selected)

    print(f"Wrote {len(selected)} expansion candidates to {output_path}")


if __name__ == "__main__":
    main()
