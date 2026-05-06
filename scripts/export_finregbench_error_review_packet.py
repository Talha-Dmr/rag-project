#!/usr/bin/env python3
"""Export FinRegBench mined errors to reviewer-friendly CSV and JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "id",
    "priority",
    "error_type",
    "expected_status",
    "predicted_status",
    "source_id",
    "generation_method",
    "artifact_level",
    "artifact_flags",
    "supported_score",
    "unsupported_score",
    "contradicted_score",
    "expected_score_gap",
    "prediction_margin",
    "query",
    "candidate_answer",
    "evidence_span",
    "review_decision",
    "approved_support_status",
    "review_notes",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def csv_row(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("scores") or {}
    return {
        "id": row.get("id"),
        "priority": row.get("priority"),
        "error_type": row.get("error_type"),
        "expected_status": row.get("expected_status"),
        "predicted_status": row.get("predicted_status"),
        "source_id": row.get("source_id"),
        "generation_method": row.get("generation_method"),
        "artifact_level": row.get("artifact_level"),
        "artifact_flags": "|".join(str(item) for item in row.get("artifact_flags") or []),
        "supported_score": scores.get("supported"),
        "unsupported_score": scores.get("unsupported"),
        "contradicted_score": scores.get("contradicted"),
        "expected_score_gap": row.get("expected_score_gap"),
        "prediction_margin": row.get("prediction_margin"),
        "query": row.get("query"),
        "candidate_answer": row.get("candidate_answer"),
        "evidence_span": row.get("evidence_span"),
        "review_decision": row.get("review_decision") or "pending",
        "approved_support_status": row.get("approved_support_status") or "",
        "review_notes": row.get("review_notes") or "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_row(row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("FinRegBench/data/phase2_error_mining/error_review_queue.jsonl"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("FinRegBench/data/phase2_error_mining/error_review_packet.csv"),
    )
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=Path("FinRegBench/data/phase2_error_mining/error_review_packet.jsonl"),
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    rows = sorted(
        rows,
        key=lambda row: (
            -int(row.get("priority") or 0),
            str(row.get("error_type") or ""),
            str(row.get("id") or ""),
        ),
    )
    write_csv(args.csv_output, rows)
    write_jsonl(args.jsonl_output, rows)


if __name__ == "__main__":
    main()
