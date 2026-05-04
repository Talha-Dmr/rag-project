#!/usr/bin/env python3
"""
Sync reviewed gold labels from reviewer_notes_draft.csv back into the other review artifacts.

Targets:
- priority_review.csv
- reviewer_packet.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List


SYNC_FIELDS = ("gold_label", "gold_error_type", "review_notes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync Phase 2 review labels into review artifacts")
    parser.add_argument(
        "--source-csv",
        default="evaluation_results/finreg_detector_phase2_priority_review_smoke/reviewer_notes_draft.csv",
        help="Source CSV containing reviewed gold fields",
    )
    parser.add_argument(
        "--target-csv",
        action="append",
        default=[],
        help="Target CSV to update. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else list(SYNC_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    targets = args.target_csv or [
        "evaluation_results/finreg_detector_phase2_priority_review_smoke/priority_review.csv",
        "evaluation_results/finreg_detector_phase2_priority_review_smoke/reviewer_packet.csv",
    ]

    source_rows = load_csv(Path(args.source_csv))
    source_by_id = {row.get("id", ""): row for row in source_rows if row.get("id")}

    for target in targets:
        path = Path(target)
        target_rows = load_csv(path)
        updated = 0
        for row in target_rows:
            source = source_by_id.get(row.get("id", ""))
            if not source:
                continue
            for field in SYNC_FIELDS:
                value = source.get(field, "")
                if value:
                    row[field] = value
            updated += 1
        write_csv(path, target_rows)
        print(f"Updated {updated} rows in {path}")


if __name__ == "__main__":
    main()
