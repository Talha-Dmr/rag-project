#!/usr/bin/env python3
"""Run a FinRegBench hybrid detector.

The hybrid path uses the deterministic lexical/artifact verifier for clear
cases, then falls back to the neural hallucination detector on low-confidence
rows.  This keeps the FinRegBench artifact signal explicit while preserving a
path for model-based decisions where the verifier is uncertain.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.rag.artifact_verifier import predict_support_status  # noqa: E402
from run_finregbench_detector_model import (  # noqa: E402
    NLI_TO_SUPPORT_STATUS,
    load_detector_class,
    normalize_nli_label,
    support_scores_from_nli,
)


LOGGER = logging.getLogger("run_finregbench_hybrid_detector")
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


def max_score(scores: dict[str, Any]) -> float:
    if not scores:
        return 0.0
    return max(float(value or 0.0) for value in scores.values())


def lexical_prediction(row: dict[str, Any]) -> dict[str, Any]:
    prediction = predict_support_status(
        query=str(row.get("query") or ""),
        candidate_answer=str(row.get("candidate_answer") or ""),
        evidence_span=str(row.get("evidence_span") or ""),
    )
    support_status = prediction["support_status"]
    scores = prediction["support_status_scores"]
    return {
        "id": row.get("id"),
        "label": SUPPORT_TO_NLI[support_status],
        "support_status": support_status,
        "scores": {
            "entailment": float(scores.get("supported", 0.0)),
            "neutral": float(scores.get("unsupported", 0.0)),
            "contradiction": float(scores.get("contradicted", 0.0)),
        },
        "support_status_scores": scores,
        "confidence": max_score(scores),
        "is_hallucination": support_status != "supported",
        "prediction_source": "lexical_artifact_verifier",
        "lexical_features": prediction["features"],
    }


def make_hypothesis(row: dict[str, Any], mode: str) -> str:
    answer = str(row.get("candidate_answer") or "")
    query = str(row.get("query") or "")
    if mode == "question_answer":
        return f"Question: {query}\nAnswer: {answer}".strip()
    if mode == "answer_only":
        return answer
    raise ValueError(f"Unknown hypothesis mode: {mode}")


def neural_predictions(
    rows: list[dict[str, Any]],
    *,
    model_path: Path,
    base_model: str | None,
    device: str | None,
    max_length: int,
    batch_size: int,
    hypothesis_mode: str,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    detector_class = load_detector_class()
    detector = detector_class(
        model_path=str(model_path),
        base_model=base_model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    premises = [str(row.get("evidence_span") or "") for row in rows]
    hypotheses = [make_hypothesis(row, hypothesis_mode) for row in rows]
    detections = detector.detect_batch(premises, hypotheses, return_scores=True)
    if len(detections) != len(rows):
        raise RuntimeError(f"Detector returned {len(detections)} rows for {len(rows)} inputs")

    predictions: list[dict[str, Any]] = []
    for row, detection in zip(rows, detections):
        nli_label = normalize_nli_label(detection.get("label"))
        support_status = NLI_TO_SUPPORT_STATUS[nli_label]
        nli_scores = {
            key: float(value)
            for key, value in (detection.get("scores") or {}).items()
        }
        predictions.append(
            {
                "id": row.get("id"),
                "label": nli_label,
                "support_status": support_status,
                "scores": nli_scores,
                "support_status_scores": support_scores_from_nli(nli_scores),
                "confidence": float(detection.get("confidence", 0.0) or 0.0),
                "is_hallucination": bool(detection.get("is_hallucination", False)),
                "prediction_source": "neural_low_confidence_fallback",
            }
        )
    return predictions


def build_predictions(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    lexical_rows = [lexical_prediction(row) for row in rows]
    fallback_indices = [
        index
        for index, prediction in enumerate(lexical_rows)
        if prediction["confidence"] < args.lexical_confidence_threshold
    ]

    if not fallback_indices or args.lexical_only:
        LOGGER.info("Using lexical verifier for all %s rows", len(rows))
        return lexical_rows

    if args.model_path is None:
        raise SystemExit("--model-path is required unless --lexical-only is set")

    fallback_inputs = [rows[index] for index in fallback_indices]
    LOGGER.info(
        "Using neural fallback for %s/%s rows below lexical confidence %.3f",
        len(fallback_inputs),
        len(rows),
        args.lexical_confidence_threshold,
    )
    fallback_outputs = neural_predictions(
        fallback_inputs,
        model_path=args.model_path,
        base_model=args.base_model,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        hypothesis_mode=args.hypothesis_mode,
    )

    merged = list(lexical_rows)
    for index, neural_prediction in zip(fallback_indices, fallback_outputs):
        neural_prediction["lexical_prediction"] = lexical_rows[index]
        neural_prediction["lexical_confidence"] = lexical_rows[index]["confidence"]
        merged[index] = neural_prediction
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--inputs", dest="input", type=Path, required=True)
    parser.add_argument("--output", "--predictions", dest="output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--hypothesis-mode",
        choices=["answer_only", "question_answer"],
        default="answer_only",
    )
    parser.add_argument(
        "--lexical-confidence-threshold",
        type=float,
        default=0.42,
        help="Rows below this lexical confidence are sent to the neural fallback.",
    )
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Disable neural fallback and emit lexical verifier predictions only.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    rows = read_jsonl(args.input)
    predictions = build_predictions(rows, args)
    write_jsonl(args.output, predictions)
    LOGGER.info("Wrote predictions to %s", args.output)


if __name__ == "__main__":
    main()
