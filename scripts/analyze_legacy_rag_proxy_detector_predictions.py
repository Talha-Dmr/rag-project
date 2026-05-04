#!/usr/bin/env python3
"""Analyze detector predictions on legacy RAG QA proxy inputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def f1_bucket(value: float) -> str:
    if value >= 0.9:
        return "f1_0.90_1.00"
    if value >= 0.5:
        return "f1_0.50_0.89"
    if value > 0.0:
        return "f1_0.01_0.49"
    return "f1_0"


def support_status(prediction: dict[str, Any]) -> str:
    return str(prediction.get("support_status") or prediction.get("label") or "unknown")


def prediction_source(prediction: dict[str, Any]) -> str:
    return str(prediction.get("prediction_source") or "unknown")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--max-cases", type=int, default=80)
    args = parser.parse_args()

    input_rows = read_jsonl(args.inputs)
    prediction_rows = read_jsonl(args.predictions)
    input_by_id = {str(row.get("id")): row for row in input_rows}

    missing = [row.get("id") for row in input_rows if str(row.get("id")) not in {
        str(pred.get("id")) for pred in prediction_rows
    }]
    joined: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        row_id = str(prediction.get("id"))
        source = input_by_id.get(row_id)
        if source is None:
            continue
        joined.append({"input": source, "prediction": prediction})

    status_counts = Counter()
    source_counts = Counter()
    exact_by_status: dict[str, Counter[str]] = defaultdict(Counter)
    f1_by_status: dict[str, Counter[str]] = defaultdict(Counter)
    arch_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    risk_cases: list[dict[str, Any]] = []

    f1_values: list[float] = []
    exact_values: list[float] = []
    for item in joined:
        source = item["input"]
        prediction = item["prediction"]
        status = support_status(prediction)
        pred_source = prediction_source(prediction)
        exact = float(source.get("qa_exact_match", 0.0) or 0.0)
        token_f1 = float(source.get("qa_token_f1", 0.0) or 0.0)
        architecture = str((source.get("metadata") or {}).get("architecture") or "unknown")

        status_counts[status] += 1
        source_counts[pred_source] += 1
        exact_by_status[status][str(exact)] += 1
        f1_by_status[status][f1_bucket(token_f1)] += 1
        arch_status_counts[architecture][status] += 1
        f1_values.append(token_f1)
        exact_values.append(exact)

        high_risk = (
            (status == "supported" and token_f1 < 0.5)
            or (status != "supported" and token_f1 >= 0.9)
            or (status == "contradicted" and token_f1 >= 0.5)
        )
        if high_risk:
            risk_cases.append(
                {
                    "id": source.get("id"),
                    "architecture": architecture,
                    "support_status": status,
                    "prediction_source": pred_source,
                    "qa_exact_match": exact,
                    "qa_token_f1": token_f1,
                    "query": source.get("query"),
                    "reference_answer": source.get("reference_answer"),
                    "candidate_answer": source.get("candidate_answer"),
                }
            )

    report = {
        "rows": len(joined),
        "missing_prediction_ids": missing[:50],
        "support_status_counts": dict(status_counts),
        "prediction_source_counts": dict(source_counts),
        "f1_bucket_by_support_status": {
            key: dict(value) for key, value in f1_by_status.items()
        },
        "exact_match_by_support_status": {
            key: dict(value) for key, value in exact_by_status.items()
        },
        "architecture_status_counts": {
            key: dict(value) for key, value in arch_status_counts.items()
        },
        "qa_metrics": {
            "exact_match_mean": sum(exact_values) / len(exact_values) if exact_values else 0.0,
            "token_f1_mean": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        },
        "risk_case_count": len(risk_cases),
        "risk_case_policy": (
            "supported with low QA F1, or non-supported with high QA F1; proxy only."
        ),
    }
    write_json(args.report, report)
    if args.cases:
        write_jsonl(args.cases, risk_cases[: args.max_cases])
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
