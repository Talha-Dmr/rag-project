#!/usr/bin/env python3
"""
Generate draft review notes for the high-priority Phase 2 reviewer packet.

This script does not fill final gold labels. It provides:
- likely_label
- likely_error_type
- review_note_draft
- reviewer_confidence
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft review notes for Phase 2 high-priority rows")
    parser.add_argument(
        "--input-csv",
        default="evaluation_results/finreg_detector_phase2_priority_review_smoke/reviewer_packet.csv",
        help="Reviewer packet CSV path",
    )
    parser.add_argument(
        "--output-csv",
        default="evaluation_results/finreg_detector_phase2_priority_review_smoke/reviewer_notes_draft.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def infer_note(row: Dict[str, Any]) -> Dict[str, str]:
    reason = (row.get("review_reason") or "").lower()
    question = row.get("question") or ""
    answer = row.get("answer_preview") or ""
    suggested_label = row.get("suggested_label") or ""
    suggested_error = row.get("suggested_error_type") or ""

    if "generation_contamination" in reason:
        return {
            "likely_label": "unsupported",
            "likely_error_type": "fabricated_fact",
            "reviewer_confidence": "high",
            "review_note_draft": (
                "Likely generation contamination. Answer includes unrelated non-finreg content "
                "appended to an otherwise domain-relevant response; review whether this should be "
                "labeled unsupported rather than contradicted."
            ),
        }

    if suggested_label == "contradicted":
        return {
            "likely_label": "contradicted",
            "likely_error_type": suggested_error or "wrong_number_or_threshold",
            "reviewer_confidence": "medium",
            "review_note_draft": (
                "Detector marked this as a high-contradiction case. Check whether the answer makes "
                "a claim about alignment or thresholds that conflicts with retrieved evidence, rather "
                "than merely abstaining."
            ),
        }

    if suggested_label == "ambiguous":
        return {
            "likely_label": "ambiguous",
            "likely_error_type": suggested_error or "misinterpretation",
            "reviewer_confidence": "medium",
            "review_note_draft": (
                "Likely insufficient evidence or unresolved cross-source comparison. Verify whether the "
                "answer correctly abstains from a claim the context cannot cleanly resolve."
            ),
        }

    if suggested_label == "partial":
        return {
            "likely_label": "partial",
            "likely_error_type": suggested_error or "incomplete_reasoning",
            "reviewer_confidence": "medium",
            "review_note_draft": (
                "Likely partially supported. Check whether the main answer is grounded but overgeneralizes, "
                "drops caveats, or blends source-specific claims."
            ),
        }

    return {
        "likely_label": suggested_label or "",
        "likely_error_type": suggested_error or "",
        "reviewer_confidence": "low",
        "review_note_draft": (
            "Review manually. Suggested label exists, but evidence should be checked directly before "
            "promoting any final gold annotation."
        ),
    }


def main() -> None:
    args = parse_args()
    rows = load_csv(Path(args.input_csv))

    output_rows: List[Dict[str, Any]] = []
    for row in rows:
        note = infer_note(row)
        output_rows.append({
            "id": row.get("id", ""),
            "review_priority": row.get("review_priority", ""),
            "question": row.get("question", ""),
            "suggested_label": row.get("suggested_label", ""),
            "suggested_error_type": row.get("suggested_error_type", ""),
            "likely_label": note["likely_label"],
            "likely_error_type": note["likely_error_type"],
            "reviewer_confidence": note["reviewer_confidence"],
            "review_note_draft": note["review_note_draft"],
            "gold_label": row.get("gold_label", ""),
            "gold_error_type": row.get("gold_error_type", ""),
            "review_notes": row.get("review_notes", ""),
        })

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()) if output_rows else [
            "id", "likely_label", "review_note_draft"
        ])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote draft review notes to {output_path}")


if __name__ == "__main__":
    main()
