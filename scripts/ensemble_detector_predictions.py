#!/usr/bin/env python3
"""Combine detector prediction JSONL files by weighted score averaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ORDER = ["supported", "unsupported", "contradicted"]
STATUS_TO_NLI = {
    "supported": "entailment",
    "unsupported": "neutral",
    "contradicted": "contradiction",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def parse_weighted_path(value: str) -> tuple[Path, float]:
    if "=" not in value:
        return Path(value), 1.0
    path_text, weight_text = value.rsplit("=", 1)
    return Path(path_text), float(weight_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supported-bias", type=float, default=0.0)
    parser.add_argument("--unsupported-bias", type=float, default=0.0)
    parser.add_argument("--contradicted-bias", type=float, default=0.0)
    args = parser.parse_args()

    weighted_paths = [parse_weighted_path(value) for value in args.prediction]
    by_id: dict[str, list[tuple[dict[str, Any], float]]] = {}
    ordered_ids: list[str] = []
    for path, weight in weighted_paths:
        for row in read_jsonl(path):
            row_id = str(row.get("id"))
            if row_id not in by_id:
                ordered_ids.append(row_id)
                by_id[row_id] = []
            by_id[row_id].append((row, weight))

    output_rows = []
    for row_id in ordered_ids:
        rows = by_id[row_id]
        if len(rows) != len(weighted_paths):
            raise ValueError(f"Missing prediction for id={row_id}")

        scores = {key: 0.0 for key in ORDER}
        total_weight = 0.0
        for row, weight in rows:
            row_scores = row.get("support_status_scores") or {}
            for key in ORDER:
                scores[key] += float(row_scores.get(key, 0.0) or 0.0) * weight
            total_weight += weight
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        for key in ORDER:
            scores[key] /= total_weight

        scores["supported"] += args.supported_bias
        scores["unsupported"] += args.unsupported_bias
        scores["contradicted"] += args.contradicted_bias

        support_status = max(ORDER, key=lambda key: scores[key])
        nli_label = STATUS_TO_NLI[support_status]
        output_rows.append(
            {
                "id": row_id,
                "support_status": support_status,
                "label": nli_label,
                "support_status_scores": scores,
                "scores": {
                    "entailment": scores["supported"],
                    "neutral": scores["unsupported"],
                    "contradiction": scores["contradicted"],
                },
                "confidence": float(scores[support_status]),
            }
        )

    write_jsonl(args.output, output_rows)
    print(f"Wrote ensemble predictions to {args.output}")


if __name__ == "__main__":
    main()
