#!/usr/bin/env python3
"""Build reviewed NLI training seed from FinRegBench error review decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


SUPPORT_TO_NLI = {
    "supported": "entailment",
    "unsupported": "neutral",
    "contradicted": "contradiction",
}
STATUS_ALIASES = {
    "supported": "supported",
    "support": "supported",
    "entailment": "supported",
    "entailed": "supported",
    "unsupported": "unsupported",
    "not_supported": "unsupported",
    "not supported": "unsupported",
    "neutral": "unsupported",
    "missing_evidence": "unsupported",
    "missing evidence": "unsupported",
    "contradicted": "contradicted",
    "contradiction": "contradicted",
    "conflict": "contradicted",
    "conflicting_answer": "contradicted",
    "conflicting answer": "contradicted",
}
APPROVE_DECISIONS = {"approved", "approve", "accept", "accepted", "use", "include"}
PREFILL_DECISIONS = {
    "approved_prefill_needs_spotcheck",
    "prefill",
    "prefilled",
    "auto_prefill",
}
SKIP_DECISIONS = {"skip", "reject", "rejected", "exclude", "bad", "invalid"}


def normalize_status(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_")
    text = " ".join(text.split())
    if text in STATUS_ALIASES:
        return STATUS_ALIASES[text]
    return STATUS_ALIASES.get(text.replace(" ", "_"))


def normalize_decision(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    text = " ".join(text.split())
    text = text.replace(" ", "_")
    if text in APPROVE_DECISIONS:
        return "approved"
    if text in PREFILL_DECISIONS:
        return "prefill"
    if text in SKIP_DECISIONS:
        return "skipped"
    if text in {"pending", ""}:
        return "pending"
    return text


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


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    return read_jsonl(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_training_row(row: dict[str, Any], approved_status: str) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "premise": row.get("evidence_span"),
        "hypothesis": row.get("candidate_answer"),
        "label": SUPPORT_TO_NLI[approved_status],
        "metadata": {
            "source": "finregbench_phase2_reviewed_errors",
            "approved_support_status": approved_status,
            "original_expected_status": row.get("expected_status"),
            "predicted_status": row.get("predicted_status"),
            "error_type": row.get("error_type"),
            "priority": row.get("priority"),
            "source_id": row.get("source_id"),
            "generation_method": row.get("generation_method"),
            "artifact_level": row.get("artifact_level"),
            "artifact_flags": row.get("artifact_flags"),
            "review_notes": row.get("review_notes") or "",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviewed",
        type=Path,
        default=Path("FinRegBench/data/phase2_error_mining/error_review_packet.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FinRegBench/data/phase2_error_mining/targeted_training_seed_reviewed.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("FinRegBench/data/phase2_error_mining/reviewed_training_seed_summary.json"),
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Treat pending rows as skipped instead of failing.",
    )
    parser.add_argument(
        "--accept-prefill",
        action="store_true",
        help="Accept approved_prefill_needs_spotcheck rows as training rows.",
    )
    args = parser.parse_args()

    rows = read_rows(args.reviewed)
    training_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []

    for row in rows:
        decision = normalize_decision(row.get("review_decision"))
        approved_status = normalize_status(row.get("approved_support_status"))

        if decision == "pending":
            pending.append(row)
            continue
        if decision == "skipped":
            skipped.append(row)
            continue
        if decision == "prefill" and not args.accept_prefill:
            pending.append(row)
            continue
        if decision == "prefill":
            decision = "approved"
        if decision != "approved" or approved_status is None:
            invalid.append(row)
            continue

        training_row = build_training_row(row, approved_status)
        if normalize_decision(row.get("review_decision")) == "prefill":
            training_row["metadata"]["review_prefill"] = True
            training_row["metadata"]["prefill_risk_flags"] = row.get("prefill_risk_flags") or ""
        training_rows.append(training_row)

    if pending and not args.allow_pending:
        raise SystemExit(
            f"{len(pending)} rows are still pending. Fill review_decision or rerun with --allow-pending."
        )
    if invalid:
        raise SystemExit(
            f"{len(invalid)} rows have invalid review decisions or approved labels."
        )

    write_jsonl(args.output, training_rows)
    write_json(
        args.summary,
        {
            "input_rows": len(rows),
            "approved_rows": len(training_rows),
            "skipped_rows": len(skipped),
            "pending_rows": len(pending),
            "label_counts": dict(sorted(Counter(row["label"] for row in training_rows).items())),
            "source_counts": dict(
                sorted(Counter(row["metadata"].get("source_id") for row in training_rows).items())
            ),
            "error_type_counts": dict(
                sorted(Counter(row["metadata"].get("error_type") for row in training_rows).items())
            ),
        },
    )


if __name__ == "__main__":
    main()
