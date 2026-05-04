#!/usr/bin/env python3
"""Mine FinRegBench Phase 2 detector errors for targeted training/review."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUPPORT_TO_NLI = {
    "supported": "entailment",
    "unsupported": "neutral",
    "contradicted": "contradiction",
}


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


def stable_key(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def expected_status(record: dict[str, Any]) -> str:
    return str(get_nested(record, "labels.support_status") or record.get("support_status"))


def scores(prediction: dict[str, Any]) -> dict[str, float]:
    raw = prediction.get("support_status_scores") or {}
    return {
        "supported": float(raw.get("supported", 0.0) or 0.0),
        "unsupported": float(raw.get("unsupported", 0.0) or 0.0),
        "contradicted": float(raw.get("contradicted", 0.0) or 0.0),
    }


def predicted_status(prediction: dict[str, Any]) -> str:
    direct = prediction.get("support_status")
    if direct:
        return str(direct)
    row_scores = scores(prediction)
    return max(row_scores.items(), key=lambda item: item[1])[0]


def join_run(dataset_path: Path, predictions_path: Path, run_name: str) -> list[dict[str, Any]]:
    dataset = {str(row.get("id")): row for row in read_jsonl(dataset_path)}
    predictions = {str(row.get("id")): row for row in read_jsonl(predictions_path)}
    missing = sorted(set(dataset) - set(predictions))
    extra = sorted(set(predictions) - set(dataset))
    if missing:
        raise ValueError(f"{run_name}: missing {len(missing)} predictions; first={missing[:5]}")
    if extra:
        raise ValueError(f"{run_name}: {len(extra)} extra predictions; first={extra[:5]}")

    joined: list[dict[str, Any]] = []
    for row_id, record in dataset.items():
        prediction = predictions[row_id]
        expected = expected_status(record)
        predicted = predicted_status(prediction)
        row_scores = scores(prediction)
        expected_score = row_scores.get(expected, 0.0)
        predicted_score = row_scores.get(predicted, 0.0)
        sorted_scores = sorted(row_scores.items(), key=lambda item: item[1], reverse=True)
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        joined.append(
            {
                "id": row_id,
                "run_name": run_name,
                "expected_status": expected,
                "predicted_status": predicted,
                "is_error": expected != predicted,
                "error_type": f"{expected}_to_{predicted}" if expected != predicted else "correct",
                "expected_score": expected_score,
                "predicted_score": predicted_score,
                "prediction_margin": predicted_score - second_score,
                "expected_score_gap": predicted_score - expected_score,
                "scores": row_scores,
                "record": record,
                "prediction": prediction,
            }
        )
    return joined


def priority(row: dict[str, Any]) -> int:
    error_type = row["error_type"]
    if error_type in {
        "contradicted_to_supported",
        "unsupported_to_supported",
        "supported_to_contradicted",
    }:
        return 3
    if error_type in {
        "contradicted_to_unsupported",
        "supported_to_unsupported",
        "unsupported_to_contradicted",
    }:
        return 2
    return 1


def review_row(row: dict[str, Any]) -> dict[str, Any]:
    record = row["record"]
    return {
        "id": row["id"],
        "run_name": row["run_name"],
        "priority": priority(row),
        "error_type": row["error_type"],
        "expected_status": row["expected_status"],
        "predicted_status": row["predicted_status"],
        "scores": row["scores"],
        "expected_score_gap": row["expected_score_gap"],
        "prediction_margin": row["prediction_margin"],
        "query": get_nested(record, "input.query") or record.get("query"),
        "candidate_answer": get_nested(record, "input.candidate_answer")
        or record.get("candidate_answer"),
        "evidence_span": get_nested(record, "input.evidence_span")
        or record.get("evidence_span"),
        "source_id": get_nested(record, "metadata.source_id"),
        "generation_method": get_nested(record, "metadata.generation_method"),
        "artifact_level": get_nested(record, "metadata.artifact_level"),
        "artifact_flags": get_nested(record, "metadata.artifact_flags", []),
        "review_decision": "pending",
        "approved_support_status": None,
        "review_notes": "",
    }


def nli_training_row(row: dict[str, Any]) -> dict[str, Any]:
    record = row["record"]
    expected = row["expected_status"]
    return {
        "id": row["id"],
        "premise": get_nested(record, "input.evidence_span") or record.get("evidence_span"),
        "hypothesis": get_nested(record, "input.candidate_answer")
        or record.get("candidate_answer"),
        "label": SUPPORT_TO_NLI[expected],
        "metadata": {
            "source": "finregbench_phase2_error_mining",
            "run_name": row["run_name"],
            "error_type": row["error_type"],
            "expected_status": expected,
            "predicted_status": row["predicted_status"],
            "source_id": get_nested(record, "metadata.source_id"),
            "generation_method": get_nested(record, "metadata.generation_method"),
            "artifact_level": get_nested(record, "metadata.artifact_level"),
            "artifact_flags": get_nested(record, "metadata.artifact_flags", []),
            "requires_human_review": True,
        },
    }


def summarize(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if row["is_error"]]
    return {
        "total_rows": len(rows),
        "total_errors": len(errors),
        "error_rate": len(errors) / len(rows) if rows else None,
        "run_counts": dict(sorted(Counter(row["run_name"] for row in rows).items())),
        "error_type_counts": dict(sorted(Counter(row["error_type"] for row in errors).items())),
        "selected_rows": len(selected),
        "selected_error_type_counts": dict(
            sorted(Counter(row["error_type"] for row in selected).items())
        ),
        "selected_priority_counts": dict(
            sorted(Counter(str(priority(row)) for row in selected).items())
        ),
    }


def select_errors(rows: list[dict[str, Any]], limit: int, seed: str) -> list[dict[str, Any]]:
    errors = [row for row in rows if row["is_error"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in errors:
        grouped[row["error_type"]].append(row)

    selected_by_id: dict[str, dict[str, Any]] = {}
    per_type = max(1, limit // max(1, len(grouped)))
    for error_type, group_rows in grouped.items():
        ordered = sorted(
            group_rows,
            key=lambda row: (
                -priority(row),
                -float(row["expected_score_gap"]),
                stable_key(row["id"], f"{seed}:{error_type}"),
            ),
        )
        for row in ordered[:per_type]:
            selected_by_id[row["id"]] = row

    if len(selected_by_id) < limit:
        remaining = [row for row in errors if row["id"] not in selected_by_id]
        remaining = sorted(
            remaining,
            key=lambda row: (
                -priority(row),
                -float(row["expected_score_gap"]),
                stable_key(row["id"], f"{seed}:fill"),
            ),
        )
        for row in remaining[: limit - len(selected_by_id)]:
            selected_by_id[row["id"]] = row

    selected = list(selected_by_id.values())
    return sorted(
        selected,
        key=lambda row: (
            -priority(row),
            row["error_type"],
            stable_key(row["id"], f"{seed}:final"),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--predictions", action="append", required=True)
    parser.add_argument("--run-name", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("FinRegBench/data/phase2_error_mining"),
    )
    parser.add_argument("--limit", type=int, default=240)
    parser.add_argument("--seed", default="finregbench_phase2_error_mining_v1")
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
        rows.extend(join_run(Path(dataset), Path(predictions), run_name))

    selected = select_errors(rows, args.limit, args.seed)
    review_rows = [review_row(row) for row in selected]
    training_rows = [nli_training_row(row) for row in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "error_review_queue.jsonl", review_rows)
    write_jsonl(args.output_dir / "targeted_training_seed_unreviewed.jsonl", training_rows)
    write_json(
        args.output_dir / "error_mining_summary.json",
        summarize(rows, selected),
    )


if __name__ == "__main__":
    main()
