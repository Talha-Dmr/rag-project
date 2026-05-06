#!/usr/bin/env python3
"""Run the project hallucination detector on FinRegBench detector inputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any, Type


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger("run_finregbench_detector_model")

NLI_TO_SUPPORT_STATUS = {
    "entailment": "supported",
    "neutral": "unsupported",
    "contradiction": "contradicted",
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


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise ImportError("PyYAML is required when --config is used") from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected YAML object")
    return loaded


def resolve_detector_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if args.config:
        loaded = load_yaml(args.config)
        config.update(loaded.get("hallucination_detector", {}) or {})

    model_path = args.model_path or config.get("model_path")
    if not model_path:
        raise SystemExit("--model-path is required unless --config provides hallucination_detector.model_path")

    base_model = args.base_model or config.get("base_model")
    return {
        "model_path": str(model_path),
        "base_model": base_model,
        "mc_dropout_samples": int(args.mc_dropout_samples or config.get("mc_dropout_samples", 1)),
        "logit_sampling_config": config.get("logit_sampling") if args.use_config_sampling else None,
        "artifact_verifier_config": (
            {
                "enabled": True,
                "confidence_threshold": args.artifact_verifier_threshold,
            }
            if args.artifact_verifier
            else config.get("artifact_verifier")
        ),
        "device": args.device,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
    }


def load_detector_class() -> Type[Any]:
    module_path = PROJECT_ROOT / "src" / "rag" / "hallucination_detector.py"
    spec = importlib.util.spec_from_file_location("finreg_hallucination_detector", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load detector module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HallucinationDetector


def normalize_nli_label(label: Any) -> str:
    text = str(label or "").strip().lower().replace("-", "_")
    text = " ".join(text.split())
    text = text.replace("_", " ")
    aliases = {
        "entailment": "entailment",
        "entailed": "entailment",
        "supports": "entailment",
        "supported": "entailment",
        "support": "entailment",
        "neutral": "neutral",
        "not enough info": "neutral",
        "not_enough_info": "neutral",
        "unknown": "neutral",
        "nei": "neutral",
        "contradiction": "contradiction",
        "contradicted": "contradiction",
        "refutes": "contradiction",
        "conflict": "contradiction",
    }
    normalized = aliases.get(text) or aliases.get(text.replace(" ", "_"))
    if normalized is None:
        raise ValueError(f"Unsupported detector label: {label!r}")
    return normalized


def support_scores_from_nli(scores: dict[str, Any]) -> dict[str, float]:
    return {
        "supported": float(scores.get("entailment", 0.0) or 0.0),
        "unsupported": float(scores.get("neutral", 0.0) or 0.0),
        "contradicted": float(scores.get("contradiction", 0.0) or 0.0),
    }


def make_hypothesis(row: dict[str, Any], mode: str) -> str:
    answer = str(row.get("candidate_answer") or "")
    query = str(row.get("query") or "")
    if mode == "question_answer":
        return f"Question: {query}\nAnswer: {answer}".strip()
    if mode == "answer_only":
        return answer
    raise ValueError(f"Unknown hypothesis mode: {mode}")


def build_predictions(
    detector: Any,
    rows: list[dict[str, Any]],
    *,
    hypothesis_mode: str,
) -> list[dict[str, Any]]:
    premises = [str(row.get("evidence_span") or "") for row in rows]
    hypotheses = [make_hypothesis(row, hypothesis_mode) for row in rows]

    LOGGER.info("Running detector on %s rows", len(rows))
    detections = detector.detect_batch(premises, hypotheses, return_scores=True)
    if len(detections) != len(rows):
        raise RuntimeError(
            f"Detector returned {len(detections)} rows for {len(rows)} inputs"
        )

    predictions: list[dict[str, Any]] = []
    for row, detection in zip(rows, detections):
        nli_label = normalize_nli_label(detection.get("label"))
        support_status = NLI_TO_SUPPORT_STATUS[nli_label]
        nli_scores = {
            key: float(value)
            for key, value in (detection.get("scores") or {}).items()
        }
        prediction = {
            "id": row.get("id"),
            "label": nli_label,
            "support_status": support_status,
            "scores": nli_scores,
            "support_status_scores": support_scores_from_nli(nli_scores),
            "confidence": float(detection.get("confidence", 0.0) or 0.0),
            "is_hallucination": bool(detection.get("is_hallucination", False)),
            "prediction_source": detection.get("prediction_source", "neural_detector"),
        }
        for key in (
            "artifact_verifier",
            "artifact_verifier_confidence_threshold",
            "uncertainty_entropy",
            "uncertainty_variance",
            "contradiction_variance",
            "uncertainty_logit_mi",
            "uncertainty_logit_variance",
            "uncertainty_logit_entropy",
            "uncertainty_rep_mi",
            "uncertainty_rep_variance",
            "uncertainty_rep_entropy",
        ):
            if key in detection:
                if key.startswith("uncertainty") or key == "contradiction_variance":
                    prediction[key] = float(detection[key])
                else:
                    prediction[key] = detection[key]
        predictions.append(prediction)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "--inputs", dest="input", type=Path, required=True)
    parser.add_argument("--output", "--predictions", dest="output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional gating config with hallucination_detector section.",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mc-dropout-samples", type=int)
    parser.add_argument(
        "--use-config-sampling",
        action="store_true",
        help="Use logit_sampling settings from --config. Slower but matches gating config uncertainty mode.",
    )
    parser.add_argument(
        "--artifact-verifier",
        action="store_true",
        help=(
            "Enable the lexical artifact verifier before neural decisions for "
            "FinRegBench hybrid evals. This is an explicit experiment flag, not "
            "the default production detector path."
        ),
    )
    parser.add_argument("--artifact-verifier-threshold", type=float, default=0.42)
    parser.add_argument(
        "--hypothesis-mode",
        choices=["answer_only", "question_answer"],
        default="answer_only",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    detector_config = resolve_detector_config(args)
    LOGGER.info("Detector config: %s", detector_config)

    detector_class = load_detector_class()
    detector = detector_class(**detector_config)
    rows = read_jsonl(args.input)
    predictions = build_predictions(
        detector,
        rows,
        hypothesis_mode=args.hypothesis_mode,
    )
    write_jsonl(args.output, predictions)
    LOGGER.info("Wrote predictions to %s", args.output)


if __name__ == "__main__":
    main()
