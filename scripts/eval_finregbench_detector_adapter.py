#!/usr/bin/env python3
"""Evaluate detector predictions on the FinRegBench detector format.

The expected dataset is produced by:

    scripts/prepare_finregbench_detector_format.py

This adapter intentionally evaluates two views:

1. Three-way answer verification:
   supported / unsupported / contradicted
2. Binary groundedness:
   supported / not_supported

The second view is useful for a retrieval gate or hallucination detector, while
the first keeps unsupported and contradictory answers separate.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORT_STATUSES = {"supported", "unsupported", "contradicted"}
NLI_TO_SUPPORT_STATUS = {
    "entailment": "supported",
    "neutral": "unsupported",
    "contradiction": "contradicted",
}
LEGACY_STATUS_ALIASES = {
    "support": "supported",
    "supported": "supported",
    "entailment": "supported",
    "entailed": "supported",
    "yes": "supported",
    "true": "supported",
    "unsupported": "unsupported",
    "not_supported": "unsupported",
    "not supported": "unsupported",
    "neutral": "unsupported",
    "missing_evidence": "unsupported",
    "missing evidence": "unsupported",
    "no": "unsupported",
    "false": "unsupported",
    "contradicted": "contradicted",
    "contradiction": "contradicted",
    "conflicting_answer": "contradicted",
    "conflicting answer": "contradicted",
    "conflict": "contradicted",
}
DEFAULT_SLICE_FIELDS = [
    "metadata.source_id",
    "metadata.jurisdiction",
    "metadata.difficulty",
    "metadata.generation_method",
    "labels.ambiguity_status",
    "labels.nli_label",
]


@dataclass(frozen=True)
class EvalRow:
    row_id: str
    expected_status: str
    predicted_status: str
    record: dict[str, Any]
    prediction: dict[str, Any]

    @property
    def expected_binary(self) -> str:
        return to_binary(self.expected_status)

    @property
    def predicted_binary(self) -> str:
        return to_binary(self.predicted_status)


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


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_nested(record: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def normalize_status(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    normalized = " ".join(normalized.split())
    normalized = normalized.replace("_", " ")
    aliased = LEGACY_STATUS_ALIASES.get(normalized)
    if aliased:
        return aliased
    normalized = normalized.replace(" ", "_")
    if normalized in SUPPORT_STATUSES:
        return normalized
    return NLI_TO_SUPPORT_STATUS.get(normalized)


def to_binary(status: str) -> str:
    return "supported" if status == "supported" else "not_supported"


def expected_status(record: dict[str, Any]) -> str:
    candidates = [
        get_nested(record, "labels.support_status"),
        get_nested(record, "labels.nli_label"),
        record.get("support_status"),
        record.get("label"),
        record.get("expected_label"),
    ]
    for candidate in candidates:
        status = normalize_status(candidate)
        if status is not None:
            return status
    raise ValueError(f"{record.get('id', '<missing-id>')}: missing expected label")


def predicted_status(prediction: dict[str, Any]) -> str:
    direct_fields = [
        "support_status",
        "predicted_support_status",
        "prediction",
        "predicted_label",
        "label",
        "nli_label",
    ]
    for field in direct_fields:
        status = normalize_status(prediction.get(field))
        if status is not None:
            return status

    for scores_key in ("support_status_scores", "scores", "label_scores"):
        scores = prediction.get(scores_key)
        if isinstance(scores, dict) and scores:
            best_label = max(scores.items(), key=lambda item: float(item[1]))[0]
            status = normalize_status(best_label)
            if status is not None:
                return status

    raise ValueError(
        f"{prediction.get('id', '<missing-id>')}: prediction must contain a support label or scores"
    )


def make_eval_rows(
    dataset_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]
) -> list[EvalRow]:
    dataset_by_id = {str(row.get("id")): row for row in dataset_rows}
    predictions_by_id = {str(row.get("id")): row for row in prediction_rows}

    missing_predictions = sorted(set(dataset_by_id) - set(predictions_by_id))
    extra_predictions = sorted(set(predictions_by_id) - set(dataset_by_id))
    if missing_predictions:
        preview = ", ".join(missing_predictions[:5])
        raise ValueError(
            f"missing predictions for {len(missing_predictions)} rows; first ids: {preview}"
        )
    if extra_predictions:
        preview = ", ".join(extra_predictions[:5])
        raise ValueError(
            f"predictions contain {len(extra_predictions)} unknown ids; first ids: {preview}"
        )

    rows: list[EvalRow] = []
    for row_id, record in dataset_by_id.items():
        prediction = predictions_by_id[row_id]
        rows.append(
            EvalRow(
                row_id=row_id,
                expected_status=expected_status(record),
                predicted_status=predicted_status(prediction),
                record=record,
                prediction=prediction,
            )
        )
    return rows


def accuracy(rows: list[EvalRow], *, binary: bool) -> float | None:
    if not rows:
        return None
    if binary:
        correct = sum(row.expected_binary == row.predicted_binary for row in rows)
    else:
        correct = sum(row.expected_status == row.predicted_status for row in rows)
    return correct / len(rows)


def confusion(rows: list[EvalRow], *, binary: bool) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        expected = row.expected_binary if binary else row.expected_status
        predicted = row.predicted_binary if binary else row.predicted_status
        matrix[expected][predicted] += 1
    return {
        expected: dict(sorted(predictions.items()))
        for expected, predictions in sorted(matrix.items())
    }


def precision_recall_f1(rows: list[EvalRow], labels: list[str], *, binary: bool) -> dict[str, Any]:
    per_label: dict[str, dict[str, float | int | None]] = {}
    macro_f1_values: list[float] = []

    for label in labels:
        tp = fp = fn = 0
        support = 0
        for row in rows:
            expected = row.expected_binary if binary else row.expected_status
            predicted = row.predicted_binary if binary else row.predicted_status
            if expected == label:
                support += 1
            if expected == label and predicted == label:
                tp += 1
            elif expected != label and predicted == label:
                fp += 1
            elif expected == label and predicted != label:
                fn += 1

        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        )
        if f1 is not None:
            macro_f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "per_label": per_label,
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values)
        if macro_f1_values
        else None,
    }


def slice_reports(rows: list[EvalRow], slice_fields: list[str]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for field in slice_fields:
        grouped: dict[str, list[EvalRow]] = defaultdict(list)
        for row in rows:
            value = get_nested(row.record, field, "unknown")
            if isinstance(value, list):
                value = ",".join(str(item) for item in value)
            grouped[str(value)].append(row)

        for value, slice_rows in sorted(grouped.items()):
            reports.append(
                {
                    "slice_field": field,
                    "slice_value": value,
                    "count": len(slice_rows),
                    "three_way_accuracy": accuracy(slice_rows, binary=False),
                    "binary_accuracy": accuracy(slice_rows, binary=True),
                    "expected_counts": dict(
                        sorted(Counter(row.expected_status for row in slice_rows).items())
                    ),
                    "predicted_counts": dict(
                        sorted(Counter(row.predicted_status for row in slice_rows).items())
                    ),
                }
            )
    return reports


def error_rows(rows: list[EvalRow], *, limit: int) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for row in rows:
        if row.expected_status == row.predicted_status:
            continue
        errors.append(
            {
                "id": row.row_id,
                "expected_status": row.expected_status,
                "predicted_status": row.predicted_status,
                "expected_binary": row.expected_binary,
                "predicted_binary": row.predicted_binary,
                "query": get_nested(row.record, "input.query") or row.record.get("query"),
                "candidate_answer": get_nested(row.record, "input.candidate_answer")
                or row.record.get("candidate_answer"),
                "evidence_span": get_nested(row.record, "input.evidence_span")
                or row.record.get("evidence_span"),
                "source_id": get_nested(row.record, "metadata.source_id")
                or row.record.get("doc_id"),
                "difficulty": get_nested(row.record, "metadata.difficulty")
                or row.record.get("difficulty"),
                "generation_method": get_nested(row.record, "metadata.generation_method")
                or row.record.get("generation_method"),
                "ambiguity_status": get_nested(row.record, "labels.ambiguity_status"),
                "ambiguity_type": get_nested(row.record, "labels.ambiguity_type")
                or row.record.get("ambiguity_type"),
            }
        )
        if len(errors) >= limit:
            break
    return errors


def build_report(rows: list[EvalRow], slice_fields: list[str], error_limit: int) -> dict[str, Any]:
    three_way_labels = ["supported", "unsupported", "contradicted"]
    binary_labels = ["supported", "not_supported"]
    negative_rows = [row for row in rows if row.expected_status != "supported"]
    negative_prediction_leakage = (
        sum(row.predicted_status == "supported" for row in negative_rows) / len(negative_rows)
        if negative_rows
        else None
    )

    return {
        "total_rows": len(rows),
        "expected_counts": dict(sorted(Counter(row.expected_status for row in rows).items())),
        "predicted_counts": dict(sorted(Counter(row.predicted_status for row in rows).items())),
        "three_way": {
            "accuracy": accuracy(rows, binary=False),
            "confusion": confusion(rows, binary=False),
            **precision_recall_f1(rows, three_way_labels, binary=False),
        },
        "binary_supported": {
            "accuracy": accuracy(rows, binary=True),
            "confusion": confusion(rows, binary=True),
            **precision_recall_f1(rows, binary_labels, binary=True),
        },
        "negative_only_unsupported_vs_contradicted": {
            "rows": len(negative_rows),
            "accuracy": accuracy(negative_rows, binary=False),
            "supported_prediction_leakage": negative_prediction_leakage,
            "confusion": confusion(negative_rows, binary=False),
            **precision_recall_f1(
                negative_rows, ["unsupported", "contradicted", "supported"], binary=False
            ),
        },
        "slices": slice_reports(rows, slice_fields),
        "sample_errors": error_rows(rows, limit=error_limit),
    }


def export_detector_inputs(dataset_rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in dataset_rows:
            payload = {
                "id": record.get("id"),
                "query": get_nested(record, "input.query") or record.get("query"),
                "candidate_answer": get_nested(record, "input.candidate_answer")
                or record.get("candidate_answer"),
                "evidence_span": get_nested(record, "input.evidence_span")
                or record.get("evidence_span"),
                "metadata": record.get("metadata", {}),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def export_oracle_predictions(dataset_rows: list[dict[str, Any]], output_path: Path) -> None:
    """Export label-copy predictions for adapter smoke testing only."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in dataset_rows:
            payload = {
                "id": record.get("id"),
                "support_status": expected_status(record),
                "prediction_source": "oracle_labels_for_adapter_smoke_test",
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_eval.jsonl"),
        help="Detector-format FinRegBench JSONL.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Detector predictions JSONL. Each row must include id and a label or scores.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_eval_report.json"),
        help="Evaluation report JSON output.",
    )
    parser.add_argument(
        "--slice-csv",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_eval_slices.csv"),
        help="Flat slice report CSV output.",
    )
    parser.add_argument(
        "--export-inputs",
        type=Path,
        help="Write detector input JSONL and exit without evaluating predictions.",
    )
    parser.add_argument(
        "--export-oracle-predictions",
        type=Path,
        help=(
            "Write label-copy predictions for adapter smoke testing and exit. "
            "Do not use this as a detector result."
        ),
    )
    parser.add_argument(
        "--slice-field",
        action="append",
        default=[],
        help="Additional dotted field to report as a slice.",
    )
    parser.add_argument("--error-limit", type=int, default=50)
    args = parser.parse_args()

    dataset_rows = read_jsonl(args.dataset)
    if args.export_inputs:
        export_detector_inputs(dataset_rows, args.export_inputs)
        return
    if args.export_oracle_predictions:
        export_oracle_predictions(dataset_rows, args.export_oracle_predictions)
        return

    if args.predictions is None:
        raise SystemExit("--predictions is required unless --export-inputs is used")
    if not args.predictions.exists():
        raise SystemExit(
            f"Prediction file not found: {args.predictions}\n"
            "First create predictions with your detector, or export inputs with:\n"
            f"  python {Path(__file__).as_posix()} --dataset {args.dataset.as_posix()} "
            "--export-inputs FinRegBench/data/finreg_3000_detector_inputs.jsonl\n"
            "For adapter smoke testing only, you can create oracle predictions with:\n"
            f"  python {Path(__file__).as_posix()} --dataset {args.dataset.as_posix()} "
            "--export-oracle-predictions FinRegBench/data/finreg_3000_detector_predictions.jsonl"
        )

    prediction_rows = read_jsonl(args.predictions)
    rows = make_eval_rows(dataset_rows, prediction_rows)
    slice_fields = DEFAULT_SLICE_FIELDS + args.slice_field
    report = build_report(rows, slice_fields=slice_fields, error_limit=args.error_limit)
    write_json(args.report, report)
    write_csv(
        args.slice_csv,
        report["slices"],
        fieldnames=[
            "slice_field",
            "slice_value",
            "count",
            "three_way_accuracy",
            "binary_accuracy",
            "expected_counts",
            "predicted_counts",
        ],
    )


if __name__ == "__main__":
    main()
