#!/usr/bin/env python3
"""Sweep contradiction probability thresholds on FinReg detector eval outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LABELS = ["entailment", "neutral", "contradiction"]


def load_details(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_model[row["model"]].append(row)
    return dict(rows_by_model)


def decide(row: dict[str, Any], contradiction_threshold: float) -> str:
    scores = row.get("scores") or {}
    if float(scores.get("contradiction", 0.0)) >= contradiction_threshold:
        return "contradiction"

    non_contra = {
        "entailment": float(scores.get("entailment", 0.0)),
        "neutral": float(scores.get("neutral", 0.0)),
    }
    return max(non_contra.items(), key=lambda item: item[1])[0]


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def metrics_for(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    predictions = [(row["expected"], decide(row, threshold)) for row in rows]
    total = len(predictions)
    correct = sum(1 for expected, predicted in predictions if expected == predicted)
    confusion = Counter(predictions)

    per_class: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = confusion[(label, label)]
        fp = sum(confusion[(other, label)] for other in LABELS if other != label)
        fn = sum(confusion[(label, other)] for other in LABELS if other != label)
        per_class[label] = prf(tp, fp, fn)

    non_contradiction = sum(1 for expected, _ in predictions if expected != "contradiction")
    false_contradictions = sum(
        1
        for expected, predicted in predictions
        if expected != "contradiction" and predicted == "contradiction"
    )

    return {
        "threshold": threshold,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(per_class[label]["f1"] for label in LABELS) / len(LABELS),
        "contradiction_precision": per_class["contradiction"]["precision"],
        "contradiction_recall": per_class["contradiction"]["recall"],
        "contradiction_f1": per_class["contradiction"]["f1"],
        "false_contradiction_rate": (
            false_contradictions / non_contradiction if non_contradiction else 0.0
        ),
        "predicted_counts": dict(Counter(predicted for _, predicted in predictions)),
        "confusion": {f"{src}->{dst}": count for (src, dst), count in confusion.items()},
    }


def round_floats(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, digits) for v in value]
    return value


def threshold_values(start: float, stop: float, step: float) -> list[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 6))
        current += step
    return values


def select_operating_point(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    max_false_contradiction_rate: float,
) -> dict[str, Any]:
    candidates = [metrics_for(rows, threshold) for threshold in thresholds]

    viable = [
        row for row in candidates
        if (
            row["false_contradiction_rate"] <= max_false_contradiction_rate
            and row["contradiction_recall"] > 0
        )
    ]
    if viable:
        selected = max(
            viable,
            key=lambda row: (
                row["contradiction_f1"],
                row["contradiction_recall"],
                row["macro_f1"],
                -row["threshold"],
            ),
        )
        selected["selection_status"] = "safe_with_nonzero_contradiction_recall"
        return selected

    safe = [
        row
        for row in candidates
        if row["false_contradiction_rate"] <= max_false_contradiction_rate
    ]
    if safe:
        selected = max(
            safe,
            key=lambda row: (
                row["macro_f1"],
                row["accuracy"],
                -row["threshold"],
            ),
        )
        selected["selection_status"] = "safe_but_zero_contradiction_recall"
        return selected

    selected = min(
        candidates,
        key=lambda row: (
            row["false_contradiction_rate"],
            -row["macro_f1"],
            row["threshold"],
        ),
    )
    selected["selection_status"] = "no_safe_threshold"
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--details",
        default="evaluation_results/finreg_detector_eval_v1/details.jsonl",
        help="Detector eval details JSONL containing scores.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results/finreg_detector_eval_v1/threshold_sweep.json",
        help="Output JSON summary.",
    )
    parser.add_argument("--start", type=float, default=0.20)
    parser.add_argument("--stop", type=float, default=0.80)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--max-false-contradiction-rate",
        type=float,
        default=0.10,
        help="Safety ceiling for recommending a contradiction threshold.",
    )
    args = parser.parse_args()

    rows_by_model = load_details(Path(args.details))
    thresholds = threshold_values(args.start, args.stop, args.step)

    output: dict[str, Any] = {
        "details": args.details,
        "thresholds": thresholds,
        "models": {},
    }

    for model, rows in rows_by_model.items():
        sweep = [round_floats(metrics_for(rows, threshold)) for threshold in thresholds]
        selected = round_floats(
            select_operating_point(
                rows,
                thresholds,
                max_false_contradiction_rate=args.max_false_contradiction_rate,
            )
        )
        output["models"][model] = {
            "selected_operating_point": selected,
            "sweep": sweep,
        }

        print(
            json.dumps(
                {
                    "model": model,
                    "selection_status": selected["selection_status"],
                    "selected_threshold": selected["threshold"],
                    "macro_f1": selected["macro_f1"],
                    "contradiction_recall": selected["contradiction_recall"],
                    "false_contradiction_rate": selected["false_contradiction_rate"],
                },
                sort_keys=True,
            )
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
