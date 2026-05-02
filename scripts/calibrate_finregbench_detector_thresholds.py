#!/usr/bin/env python3
"""Calibrate detector decision thresholds on FinRegBench Phase 2 runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ["supported", "unsupported", "contradicted"]


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


def get_nested(record: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def expected_status(record: dict[str, Any]) -> str:
    return str(get_nested(record, "labels.support_status") or record.get("support_status"))


def scores(prediction: dict[str, Any]) -> dict[str, float]:
    raw = prediction.get("support_status_scores") or {}
    return {
        "supported": float(raw.get("supported", 0.0) or 0.0),
        "unsupported": float(raw.get("unsupported", 0.0) or 0.0),
        "contradicted": float(raw.get("contradicted", 0.0) or 0.0),
    }


def join_rows(
    dataset_path: Path,
    predictions_path: Path,
    run_name: str,
) -> list[dict[str, Any]]:
    dataset = {str(row.get("id")): row for row in read_jsonl(dataset_path)}
    predictions = {str(row.get("id")): row for row in read_jsonl(predictions_path)}
    missing = sorted(set(dataset) - set(predictions))
    extra = sorted(set(predictions) - set(dataset))
    if missing:
        raise ValueError(f"{run_name}: missing {len(missing)} predictions; first={missing[:5]}")
    if extra:
        raise ValueError(f"{run_name}: {len(extra)} extra predictions; first={extra[:5]}")

    rows: list[dict[str, Any]] = []
    for row_id, record in dataset.items():
        prediction = predictions[row_id]
        row_scores = scores(prediction)
        rows.append(
            {
                "id": row_id,
                "run_name": run_name,
                "expected": expected_status(record),
                "argmax": max(row_scores.items(), key=lambda item: item[1])[0],
                "scores": row_scores,
                "source_id": get_nested(record, "metadata.source_id") or "unknown",
                "generation_method": get_nested(record, "metadata.generation_method") or "unknown",
                "artifact_level": get_nested(record, "metadata.artifact_level") or "unknown",
                "ambiguity_status": get_nested(record, "labels.ambiguity_status") or "unknown",
            }
        )
    return rows


def predict_with_thresholds(
    row: dict[str, Any],
    *,
    contradiction_threshold: float,
    supported_threshold: float,
    contradiction_margin: float,
    supported_margin: float,
) -> str:
    row_scores = row["scores"]
    supported = row_scores["supported"]
    unsupported = row_scores["unsupported"]
    contradicted = row_scores["contradicted"]
    best_non_contradiction = max(supported, unsupported)
    best_non_supported = max(unsupported, contradicted)

    if (
        contradicted >= contradiction_threshold
        and contradicted - best_non_contradiction >= contradiction_margin
    ):
        return "contradicted"

    if supported >= supported_threshold and supported - best_non_supported >= supported_margin:
        return "supported"

    return "unsupported"


def confusion(rows: list[dict[str, Any]], pred_key: str) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        matrix[row["expected"]][row[pred_key]] += 1
    return {
        expected: dict(sorted(preds.items()))
        for expected, preds in sorted(matrix.items())
    }


def metric_report(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    per_label: dict[str, dict[str, float | int | None]] = {}
    f1_values: list[float] = []
    correct = sum(1 for row in rows if row["expected"] == row[pred_key])

    for label in LABELS:
        tp = fp = fn = support = 0
        for row in rows:
            expected = row["expected"]
            predicted = row[pred_key]
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
            f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    negative_rows = [row for row in rows if row["expected"] != "supported"]
    supported_leakage = (
        sum(1 for row in negative_rows if row[pred_key] == "supported") / len(negative_rows)
        if negative_rows
        else None
    )

    return {
        "rows": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_label": per_label,
        "supported_prediction_leakage": supported_leakage,
        "predicted_counts": dict(sorted(Counter(row[pred_key] for row in rows).items())),
        "confusion": confusion(rows, pred_key),
    }


def with_threshold_predictions(
    rows: list[dict[str, Any]],
    *,
    contradiction_threshold: float,
    supported_threshold: float,
    contradiction_margin: float,
    supported_margin: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        new_row["threshold_prediction"] = predict_with_thresholds(
            row,
            contradiction_threshold=contradiction_threshold,
            supported_threshold=supported_threshold,
            contradiction_margin=contradiction_margin,
            supported_margin=supported_margin,
        )
        output.append(new_row)
    return output


def threshold_grid(values: str) -> list[float]:
    return [float(item) for item in values.split(",") if item.strip()]


def score_candidate(report: dict[str, Any], objective: str) -> float:
    per_label = report["per_label"]
    if objective == "macro_f1":
        return float(report["macro_f1"] or 0.0)
    if objective == "contradiction_recall":
        return float(per_label["contradicted"]["recall"] or 0.0)
    if objective == "low_supported_leakage":
        leakage = float(report["supported_prediction_leakage"] or 0.0)
        macro_f1 = float(report["macro_f1"] or 0.0)
        return macro_f1 - leakage
    raise ValueError(f"Unknown objective: {objective}")


def balanced_run_score(rows: list[dict[str, Any]], pred_key: str) -> float:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("run_name", "unknown"))].append(row)
    if not groups:
        return 0.0
    scores = []
    for group_rows in groups.values():
        report = metric_report(group_rows, pred_key)
        scores.append(float(report["macro_f1"] or 0.0))
    return sum(scores) / len(scores)


def calibrate(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for contradiction_threshold in threshold_grid(args.contradiction_thresholds):
        for supported_threshold in threshold_grid(args.supported_thresholds):
            for contradiction_margin in threshold_grid(args.contradiction_margins):
                for supported_margin in threshold_grid(args.supported_margins):
                    predicted_rows = with_threshold_predictions(
                        rows,
                        contradiction_threshold=contradiction_threshold,
                        supported_threshold=supported_threshold,
                        contradiction_margin=contradiction_margin,
                        supported_margin=supported_margin,
                    )
                    report = metric_report(predicted_rows, "threshold_prediction")
                    run_balanced_macro_f1 = balanced_run_score(
                        predicted_rows, "threshold_prediction"
                    )
                    candidates.append(
                        {
                            "thresholds": {
                                "contradiction_threshold": contradiction_threshold,
                                "supported_threshold": supported_threshold,
                                "contradiction_margin": contradiction_margin,
                                "supported_margin": supported_margin,
                            },
                            "score_macro_f1": score_candidate(report, "macro_f1"),
                            "score_run_balanced_macro_f1": run_balanced_macro_f1,
                            "score_contradiction_recall": score_candidate(
                                report, "contradiction_recall"
                            ),
                            "score_low_supported_leakage": score_candidate(
                                report, "low_supported_leakage"
                            ),
                            "report": report,
                        }
                    )
    return candidates


def top_candidates(candidates: list[dict[str, Any]], objective: str, limit: int) -> list[dict[str, Any]]:
    score_key = f"score_{objective}"
    return sorted(candidates, key=lambda row: row[score_key], reverse=True)[:limit]


def slice_reports(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("run_name", "source_id", "generation_method", "artifact_level", "ambiguity_status"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row.get(field, "unknown"))].append(row)
        output[field] = {
            value: metric_report(group_rows, pred_key)
            for value, group_rows in sorted(groups.items())
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset path. Repeat once per prediction file.",
    )
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Normalized prediction JSONL path. Repeat once per dataset.",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=[],
        help="Optional run name. Repeat once per dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FinRegBench/data/phase2_runs/threshold_calibration.json"),
    )
    parser.add_argument(
        "--contradiction-thresholds",
        default="0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60",
    )
    parser.add_argument(
        "--supported-thresholds",
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60",
    )
    parser.add_argument("--contradiction-margins", default="-0.05,0.00,0.03,0.05,0.08,0.10")
    parser.add_argument("--supported-margins", default="-0.05,0.00,0.03,0.05,0.08,0.10")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if len(args.dataset) != len(args.predictions):
        raise SystemExit("--dataset and --predictions must be repeated the same number of times")

    rows: list[dict[str, Any]] = []
    for index, (dataset, predictions) in enumerate(zip(args.dataset, args.predictions)):
        run_name = (
            args.run_name[index]
            if index < len(args.run_name)
            else Path(dataset).stem
        )
        rows.extend(join_rows(Path(dataset), Path(predictions), run_name))

    argmax_report = metric_report(rows, "argmax")
    candidates = calibrate(rows, args)

    best_macro = top_candidates(candidates, "macro_f1", args.top_k)
    best_run_balanced = top_candidates(candidates, "run_balanced_macro_f1", args.top_k)
    best_contradiction = top_candidates(candidates, "contradiction_recall", args.top_k)
    best_low_leakage = top_candidates(candidates, "low_supported_leakage", args.top_k)

    chosen = best_run_balanced[0]
    chosen_rows = with_threshold_predictions(rows, **chosen["thresholds"])

    write_json(
        args.output,
        {
            "rows": len(rows),
            "input_runs": [
                {
                    "dataset": dataset,
                    "predictions": predictions,
                    "run_name": args.run_name[index]
                    if index < len(args.run_name)
                    else Path(dataset).stem,
                }
                for index, (dataset, predictions) in enumerate(
                    zip(args.dataset, args.predictions)
                )
            ],
            "argmax_report": argmax_report,
            "best_by_macro_f1": best_macro,
            "best_by_run_balanced_macro_f1": best_run_balanced,
            "best_by_contradiction_recall": best_contradiction,
            "best_by_low_supported_leakage": best_low_leakage,
            "chosen_thresholds": chosen["thresholds"],
            "chosen_report": chosen["report"],
            "chosen_slice_reports": slice_reports(chosen_rows, "threshold_prediction"),
        },
    )


if __name__ == "__main__":
    main()
