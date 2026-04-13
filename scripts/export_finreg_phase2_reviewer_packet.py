#!/usr/bin/env python3
"""
Create a reviewer-friendly packet from the Phase 2 priority review CSV.

This keeps only the highest-signal fields for manual review and adds shortened text columns
so reviewers can work faster without scrolling through the full raw export first.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export compact reviewer packet for Phase 2")
    parser.add_argument(
        "--input-csv",
        default="evaluation_results/finreg_detector_phase2_priority_review_smoke/priority_review.csv",
        help="Priority review CSV path",
    )
    parser.add_argument(
        "--output-csv",
        default="evaluation_results/finreg_detector_phase2_priority_review_smoke/reviewer_packet.csv",
        help="Reviewer packet CSV path",
    )
    parser.add_argument(
        "--answer-char-limit",
        type=int,
        default=700,
        help="Maximum characters for the answer preview",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def shorten(text: str, limit: int) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)] + "..."


def main() -> None:
    args = parse_args()
    rows = load_csv(Path(args.input_csv))

    compact_rows: List[Dict[str, Any]] = []
    for row in rows:
        compact_rows.append({
            "id": row.get("id", ""),
            "review_priority": row.get("review_priority", ""),
            "review_reason": row.get("review_reason", ""),
            "question": row.get("question", ""),
            "answer_preview": shorten(row.get("generated_answer", ""), args.answer_char_limit),
            "suggested_label": row.get("suggested_label", ""),
            "suggested_error_type": row.get("suggested_error_type", ""),
            "proposed_slice": row.get("proposed_slice", ""),
            "detector_variant": row.get("detector_variant", ""),
            "bucket": row.get("bucket", ""),
            "predicted_detector_label": row.get("predicted_detector_label", ""),
            "contradiction_prob_mean": row.get("contradiction_prob_mean", ""),
            "hallucination_prob_topk": row.get("hallucination_prob_topk", ""),
            "detector_conflict": row.get("detector_conflict", ""),
            "retrieval_max_score": row.get("retrieval_max_score", ""),
            "retrieval_mean_score": row.get("retrieval_mean_score", ""),
            "gold_label": row.get("gold_label", ""),
            "gold_error_type": row.get("gold_error_type", ""),
            "review_notes": row.get("review_notes", ""),
        })

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact_rows[0].keys()) if compact_rows else [
            "id", "review_priority", "question", "answer_preview", "gold_label", "review_notes"
        ])
        writer.writeheader()
        writer.writerows(compact_rows)

    print(f"Wrote reviewer packet to {output_path}")


if __name__ == "__main__":
    main()
