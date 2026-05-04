#!/usr/bin/env python3
"""Compare detector loader paths on the same FinRegBench examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.hallucination_detector import HallucinationDetector  # noqa: E402


STATUS_TO_LABEL = {
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


def pick_examples(rows: list[dict[str, Any]], per_status: int) -> list[dict[str, Any]]:
    selected = []
    counts = {"supported": 0, "unsupported": 0, "contradicted": 0}
    for row in rows:
        status = row.get("labels", {}).get("support_status")
        if status in counts and counts[status] < per_status:
            selected.append(row)
            counts[status] += 1
        if all(count >= per_status for count in counts.values()):
            break
    return selected


def predict_detector(
    detector: HallucinationDetector,
    row: dict[str, Any],
    *,
    reverse: bool = False,
) -> dict[str, Any]:
    payload = row["input"]
    premise = str(payload.get("evidence_span") or "")
    hypothesis = str(payload.get("candidate_answer") or "")
    if reverse:
        premise, hypothesis = hypothesis, premise
    return detector.detect(
        premise,
        hypothesis,
        return_scores=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("FinRegBench/data/phase2_pack/smoke_300.jsonl"))
    parser.add_argument("--per-status", type=int, default=3)
    parser.add_argument(
        "--export-model",
        type=Path,
        default=Path("models/hallucination_detector_adamw_lora_targeted_multipleqa_phase2_1_mixed_v2"),
    )
    parser.add_argument(
        "--checkpoint-model",
        type=Path,
        default=Path("models/checkpoints/adamw_lora_targeted_multipleqa_phase2_1_mixed_v2/best_model"),
    )
    parser.add_argument(
        "--base-model",
        default="models/checkpoints/adamw_lora_targeted_multipleqa_phase2_1_mixed_v2/base_model_unwrapped",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--reverse", action="store_true", help="Swap premise/hypothesis at inference")
    args = parser.parse_args()

    examples = pick_examples(read_jsonl(args.dataset), args.per_status)
    detectors = {
        "export_bundle": HallucinationDetector(
            str(args.export_model),
            base_model=args.base_model,
            device=args.device,
            max_length=128,
            batch_size=8,
        ),
        "checkpoint_model_pt": HallucinationDetector(
            str(args.checkpoint_model),
            base_model=args.base_model,
            device=args.device,
            max_length=128,
            batch_size=8,
        ),
    }

    for row in examples:
        expected_status = row["labels"]["support_status"]
        print(f"\n{id(row)} {row['id']} expected={expected_status}/{STATUS_TO_LABEL[expected_status]}")
        for name, detector in detectors.items():
            pred = predict_detector(detector, row, reverse=args.reverse)
            scores = pred.get("scores") or {}
            print(
                name,
                "label=",
                pred.get("label"),
                "confidence=",
                round(float(pred.get("confidence") or 0.0), 4),
                "scores=",
                {k: round(float(v), 4) for k, v in scores.items()},
            )


if __name__ == "__main__":
    main()
