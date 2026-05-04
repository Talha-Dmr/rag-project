#!/usr/bin/env python3
"""Prefill FinRegBench error review decisions from expected labels.

This is a productivity helper, not human validation.  It marks rows as
approved_prefill_needs_spotcheck so downstream scripts can include them only
when explicitly configured to trust prefilled decisions.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


VALID_STATUSES = {"supported", "unsupported", "contradicted"}
PREFILL_DECISION = "approved_prefill_needs_spotcheck"


def risk_flags(row: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    artifact_flags = str(row.get("artifact_flags") or "")
    error_type = str(row.get("error_type") or "")
    source_id = str(row.get("source_id") or "")

    if not artifact_flags:
        flags.append("no_artifact_flag")
    if source_id == "ccpa":
        flags.append("minority_source_ccpa")
    if error_type in {"supported_to_contradicted", "unsupported_to_supported"}:
        flags.append("high_impact_error_type")

    try:
        gap = abs(float(row.get("expected_score_gap") or 0.0))
        margin = abs(float(row.get("prediction_margin") or 0.0))
    except ValueError:
        gap = 0.0
        margin = 0.0
    if gap < 0.03 or margin < 0.03:
        flags.append("low_confidence_margin")

    return flags


def read_csv(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("FinRegBench/data/phase2_error_mining/error_review_packet.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FinRegBench/data/phase2_error_mining/error_review_packet_prefilled.csv"),
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite non-pending review decisions.",
    )
    args = parser.parse_args()

    fieldnames, rows = read_csv(args.input)
    for required in ("review_decision", "approved_support_status", "expected_status", "review_notes"):
        if required not in fieldnames:
            raise SystemExit(f"Missing required CSV field: {required}")

    if "prefill_risk_flags" not in fieldnames:
        fieldnames.append("prefill_risk_flags")

    decision_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()

    for row in rows:
        current_decision = str(row.get("review_decision") or "").strip().lower()
        if current_decision not in {"", "pending"} and not args.overwrite_existing:
            decision_counts[current_decision] += 1
            continue

        expected = str(row.get("expected_status") or "").strip().lower()
        if expected not in VALID_STATUSES:
            row["review_decision"] = "skip"
            row["approved_support_status"] = ""
            row["prefill_risk_flags"] = "invalid_expected_status"
            risk_counts["invalid_expected_status"] += 1
            decision_counts["skip"] += 1
            continue

        risks = risk_flags(row)
        row["review_decision"] = PREFILL_DECISION
        row["approved_support_status"] = expected
        row["prefill_risk_flags"] = "|".join(risks)

        note = str(row.get("review_notes") or "").strip()
        prefill_note = "prefilled from expected_status; needs spot-check before gold use"
        if risks:
            prefill_note += f"; risk_flags={row['prefill_risk_flags']}"
        row["review_notes"] = f"{note} | {prefill_note}".strip(" |")

        decision_counts[PREFILL_DECISION] += 1
        status_counts[expected] += 1
        for risk in risks:
            risk_counts[risk] += 1

    write_csv(args.output, fieldnames, rows)

    print(
        {
            "rows": len(rows),
            "output": str(args.output),
            "decision_counts": dict(sorted(decision_counts.items())),
            "approved_status_counts": dict(sorted(status_counts.items())),
            "risk_counts": dict(sorted(risk_counts.items())),
        }
    )


if __name__ == "__main__":
    main()
