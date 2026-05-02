#!/usr/bin/env python3
"""Run FinRegBench Phase 2 packs through a detector and evaluate predictions.

This script is a thin integration layer around the Phase 2 pack files and
eval_finregbench_detector_adapter.py.  It can either:

1. evaluate an existing prediction JSONL file, or
2. run a detector command template that writes predictions, then evaluate them.

The detector output is normalized before evaluation so small schema differences
do not silently corrupt the score interpretation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKS = {
    "smoke_300": "smoke_300.jsonl",
    "contradiction_stress_520": "contradiction_stress_520.jsonl",
    "review_180": "review_180.jsonl",
}

LABEL_ALIASES = {
    "support": "supported",
    "supported": "supported",
    "entailment": "supported",
    "entailed": "supported",
    "grounded": "supported",
    "answer_supported": "supported",
    "true": "supported",
    "yes": "supported",
    "unsupported": "unsupported",
    "not_supported": "unsupported",
    "not supported": "unsupported",
    "neutral": "unsupported",
    "missing_evidence": "unsupported",
    "missing evidence": "unsupported",
    "ungrounded": "unsupported",
    "unknown": "unsupported",
    "false": "unsupported",
    "no": "unsupported",
    "contradicted": "contradicted",
    "contradiction": "contradicted",
    "conflicting_answer": "contradicted",
    "conflicting answer": "contradicted",
    "conflict": "contradicted",
}

DIRECT_LABEL_FIELDS = [
    "support_status",
    "predicted_support_status",
    "prediction",
    "predicted_label",
    "label",
    "nli_label",
    "verdict",
    "decision",
    "class",
]
SCORE_FIELDS = [
    "support_status_scores",
    "scores",
    "label_scores",
    "probabilities",
    "probs",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    normalized = " ".join(normalized.split())
    normalized = normalized.replace("_", " ")
    if normalized in LABEL_ALIASES:
        return LABEL_ALIASES[normalized]
    normalized = normalized.replace(" ", "_")
    return LABEL_ALIASES.get(normalized)


def normalize_scores(scores: Any) -> dict[str, float] | None:
    if not isinstance(scores, dict) or not scores:
        return None

    normalized: dict[str, float] = {}
    for raw_label, raw_score in scores.items():
        label = normalize_label(raw_label)
        if label is None:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        normalized[label] = max(score, normalized.get(label, float("-inf")))

    return normalized or None


def extract_support_status(row: dict[str, Any]) -> tuple[str | None, dict[str, float] | None]:
    for field in DIRECT_LABEL_FIELDS:
        label = normalize_label(row.get(field))
        if label is not None:
            scores = None
            for score_field in SCORE_FIELDS:
                scores = normalize_scores(row.get(score_field))
                if scores is not None:
                    break
            return label, scores

    for score_field in SCORE_FIELDS:
        scores = normalize_scores(row.get(score_field))
        if scores is None:
            continue
        label = max(scores.items(), key=lambda item: item[1])[0]
        return label, scores

    return None, None


def normalize_prediction_rows(
    rows: list[dict[str, Any]], output_path: Path, validation_path: Path
) -> None:
    normalized_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        row_id = row.get("id") or row.get("record_id") or row.get("example_id")
        label, scores = extract_support_status(row)

        if row_id is None or label is None:
            failures.append(
                {
                    "line": index,
                    "id": row_id,
                    "reason": "missing id or unmapped support label",
                    "available_fields": sorted(row.keys()),
                    "row": row,
                }
            )
            continue

        normalized = {
            "id": str(row_id),
            "support_status": label,
            "raw_prediction": row,
        }
        if scores is not None:
            normalized["support_status_scores"] = scores
        normalized_rows.append(normalized)

    validation = {
        "input_rows": len(rows),
        "normalized_rows": len(normalized_rows),
        "failed_rows": len(failures),
        "failures": failures[:25],
        "accepted_label_fields": DIRECT_LABEL_FIELDS,
        "accepted_score_fields": SCORE_FIELDS,
        "canonical_labels": ["supported", "unsupported", "contradicted"],
    }
    write_json(validation_path, validation)

    if failures:
        raise SystemExit(
            f"Could not normalize {len(failures)} prediction rows. "
            f"See {validation_path}."
        )

    write_jsonl(output_path, normalized_rows)


def run_command_template(command_template: str, *, dataset: Path, inputs: Path, output: Path) -> None:
    command = command_template.format(
        dataset=str(dataset),
        input=str(inputs),
        inputs=str(inputs),
        output=str(output),
        predictions=str(output),
    )
    completed = subprocess.run(command, shell=True)
    if completed.returncode != 0:
        raise SystemExit(f"Detector command failed with exit code {completed.returncode}: {command}")


def run_adapter_export(dataset: Path, inputs: Path) -> None:
    command = [
        sys.executable,
        "scripts/eval_finregbench_detector_adapter.py",
        "--dataset",
        str(dataset),
        "--export-inputs",
        str(inputs),
    ]
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(f"Input export failed with exit code {completed.returncode}")


def run_adapter_eval(dataset: Path, predictions: Path, report: Path, slice_csv: Path) -> None:
    command = [
        sys.executable,
        "scripts/eval_finregbench_detector_adapter.py",
        "--dataset",
        str(dataset),
        "--predictions",
        str(predictions),
        "--report",
        str(report),
        "--slice-csv",
        str(slice_csv),
    ]
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SystemExit(f"Evaluation failed with exit code {completed.returncode}")


def resolve_pack(pack_dir: Path, pack: str) -> Path:
    if pack in PACKS:
        return pack_dir / PACKS[pack]
    path = Path(pack)
    if path.exists():
        return path
    raise SystemExit(f"Unknown pack {pack!r}. Use one of {sorted(PACKS)} or a dataset path.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack-dir",
        type=Path,
        default=Path("FinRegBench/data/phase2_pack"),
    )
    parser.add_argument(
        "--pack",
        choices=sorted(PACKS),
        default="smoke_300",
        help="Phase 2 pack to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Dataset JSONL override. If set, --pack is ignored.",
    )
    parser.add_argument(
        "--detector-command",
        help=(
            "Command template that writes raw predictions. Available placeholders: "
            "{input}, {inputs}, {dataset}, {output}, {predictions}."
        ),
    )
    parser.add_argument(
        "--raw-predictions",
        type=Path,
        help="Existing raw detector predictions to normalize and evaluate.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("FinRegBench/data/phase2_runs"),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Output run name. Defaults to the pack name.",
    )
    args = parser.parse_args()

    dataset = args.dataset or resolve_pack(args.pack_dir, args.pack)
    run_name = args.run_name or dataset.stem
    output_dir = args.work_dir / run_name
    inputs_path = output_dir / "detector_inputs.jsonl"
    raw_predictions_path = args.raw_predictions or output_dir / "raw_predictions.jsonl"
    normalized_predictions_path = output_dir / "predictions_normalized.jsonl"
    validation_path = output_dir / "prediction_normalization_report.json"
    report_path = output_dir / "eval_report.json"
    slice_csv_path = output_dir / "eval_slices.csv"

    run_adapter_export(dataset, inputs_path)

    if args.detector_command:
        run_command_template(
            args.detector_command,
            dataset=dataset,
            inputs=inputs_path,
            output=raw_predictions_path,
        )
    elif args.raw_predictions is None:
        raise SystemExit("--detector-command or --raw-predictions is required")

    raw_rows = read_jsonl(raw_predictions_path)
    normalize_prediction_rows(raw_rows, normalized_predictions_path, validation_path)
    run_adapter_eval(dataset, normalized_predictions_path, report_path, slice_csv_path)

    print(f"Wrote normalized predictions: {normalized_predictions_path}")
    print(f"Wrote normalization report: {validation_path}")
    print(f"Wrote eval report: {report_path}")
    print(f"Wrote slice report: {slice_csv_path}")


if __name__ == "__main__":
    main()
