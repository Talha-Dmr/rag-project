#!/usr/bin/env python3
"""
Build a draft FinReg eval-set v2 (80-100 questions) from annotated findings.

This script is intentionally conservative:
- reuses labeled questions when they fit the target bucket
- emits authoring slots when a target bucket is under-filled
- keeps provenance so manual follow-up is easy
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


TARGET_DISTRIBUTION = {
    "supported_easy": 20,
    "unsupported": 20,
    "contradicted": 20,
    "partial_or_ambiguous": 20,
    "hard_edge_case": 10,
}


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_candidate_row(
    row: Dict[str, Any],
    target_bucket: str,
    difficulty: str,
) -> Dict[str, Any]:
    return {
        "id": f"v2::{row['id']}",
        "source_annotation_id": row["id"],
        "question": row["question"],
        "target_bucket": target_bucket,
        "difficulty": difficulty,
        "detector_variant_source": row.get("detector_variant", ""),
        "notes": (
            f"Derived from manual label={row.get('label', '')}; "
            f"error_type={row.get('error_type', '')}; "
            f"bucket={row.get('bucket', '')}"
        ).strip(),
    }


def build_placeholder(target_bucket: str, index: int, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    example_ids = [row["id"] for row in examples[:3]]
    return {
        "id": f"draft::{target_bucket}::{index:02d}",
        "question": "",
        "target_bucket": target_bucket,
        "difficulty": "hard" if target_bucket == "hard_edge_case" else "medium",
        "needs_authoring": True,
        "pattern_hint": target_bucket,
        "source_examples": example_ids,
        "notes": "Author a new question from this pattern before promoting to the final eval set.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build draft finreg eval-set v2")
    parser.add_argument(
        "--annotations",
        required=True,
        help="Filled manual_annotation_sheet.csv path",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results/finreg_detector_diagnostic/finreg_eval_v2_80_100.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = load_annotations(Path(args.annotations))
    labeled = [
        row for row in annotations
        if (row.get("label") or "").strip().lower() in {
            "supported", "unsupported", "contradicted", "partial", "ambiguous"
        }
    ]
    if not labeled:
        raise SystemExit("No labeled annotation rows found.")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in labeled:
        label = (row.get("label") or "").strip().lower()
        notes = (row.get("notes") or "").strip().lower()
        error_type = (row.get("error_type") or "").strip().lower()
        if label == "supported":
            groups["supported_easy"].append(row)
        elif label == "unsupported":
            groups["unsupported"].append(row)
        elif label == "contradicted":
            groups["contradicted"].append(row)
        elif label in {"partial", "ambiguous"}:
            groups["partial_or_ambiguous"].append(row)

        if error_type in {"cross_document_conflict", "outdated_regulation"} or "multi-hop" in notes:
            groups["hard_edge_case"].append(row)

    draft_rows: List[Dict[str, Any]] = []
    for bucket, target_count in TARGET_DISTRIBUTION.items():
        source_rows = groups.get(bucket, [])
        for row in source_rows[:target_count]:
            difficulty = "hard" if bucket == "hard_edge_case" else "medium"
            if bucket == "supported_easy":
                difficulty = "easy"
            draft_rows.append(build_candidate_row(row, bucket, difficulty))
        if len(source_rows) < target_count:
            missing = target_count - len(source_rows)
            for idx in range(1, missing + 1):
                draft_rows.append(build_placeholder(bucket, idx, source_rows))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in draft_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote draft eval set to {output_path}")
    print(f"Rows: {len(draft_rows)}")


if __name__ == "__main__":
    main()
