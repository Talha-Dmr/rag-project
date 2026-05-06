#!/usr/bin/env python3
"""Evaluate hallucination detectors on the small FinReg detector eval set."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.hallucination_detector import HallucinationDetector

LABELS = ["entailment", "neutral", "contradiction"]
LABEL_ID_TO_NAME = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("label")
            if isinstance(label, int):
                row["label"] = LABEL_ID_TO_NAME[label]
            if row.get("label") not in LABELS:
                raise ValueError(f"Invalid label at {path}:{line_no}: {label!r}")
            rows.append(row)
    return rows


def default_candidates() -> list[tuple[str, str]]:
    return [
        ("current_fever_deberta_v3_base", "electra_deberta/final_fever_deberta_v3_base_model"),
        ("balanced_recovery_v2", "detector-assets/models/checkpoints/adamw_lora_balanced_recovery_v2/best_model"),
        (
            "targeted_multipleqa_recovery_v2",
            "detector-assets/models/checkpoints/adamw_lora_targeted_multipleqa_recovery_v2/best_model",
        ),
        (
            "phase2_1_mixed_v2",
            "detector-assets/models/checkpoints/adamw_lora_targeted_multipleqa_phase2_1_mixed_v2/best_model",
        ),
    ]


def parse_candidates(values: list[str] | None) -> list[tuple[str, str]]:
    if not values:
        return default_candidates()

    parsed: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Candidate must be name=path, got: {value}")
        name, path = value.split("=", 1)
        parsed.append((name.strip(), path.strip()))
    return parsed


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for row in records if row["expected"] == row["predicted"])
    expected_counts = Counter(row["expected"] for row in records)
    predicted_counts = Counter(row["predicted"] for row in records)
    confusion = Counter((row["expected"], row["predicted"]) for row in records)

    per_class: dict[str, dict[str, float]] = {}
    for label in LABELS:
        tp = confusion[(label, label)]
        fp = sum(confusion[(other, label)] for other in LABELS if other != label)
        fn = sum(confusion[(label, other)] for other in LABELS if other != label)
        per_class[label] = prf(tp, fp, fn)

    macro_f1 = sum(per_class[label]["f1"] for label in LABELS) / len(LABELS)
    macro_precision = sum(per_class[label]["precision"] for label in LABELS) / len(LABELS)
    macro_recall = sum(per_class[label]["recall"] for label in LABELS) / len(LABELS)
    non_contradiction = sum(1 for row in records if row["expected"] != "contradiction")
    false_contradictions = sum(
        1
        for row in records
        if row["expected"] != "contradiction" and row["predicted"] == "contradiction"
    )

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "expected_counts": dict(expected_counts),
        "predicted_counts": dict(predicted_counts),
        "confusion": {f"{src}->{dst}": count for (src, dst), count in confusion.items()},
        "per_class": per_class,
        "contradiction_recall": per_class["contradiction"]["recall"],
        "false_contradiction_rate": (
            false_contradictions / non_contradiction if non_contradiction else 0.0
        ),
    }


def round_floats(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_floats(v, digits) for v in value]
    return value


def evaluate_candidate(
    name: str,
    model_path: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detector = HallucinationDetector(
        model_path=model_path,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        mc_dropout_samples=1,
        logit_sampling_config={"enabled": False},
    )

    details: list[dict[str, Any]] = []
    for row in rows:
        result = detector.detect(row["premise"], row["hypothesis"], return_scores=True)
        predicted = str(result["label"])
        scores = {key: float(val) for key, val in result.get("scores", {}).items()}
        details.append(
            {
                "model": name,
                "id": row["id"],
                "expected": row["label"],
                "predicted": predicted,
                "ok": predicted == row["label"],
                "confidence": float(result["confidence"]),
                "scores": scores,
                "metadata": row.get("metadata", {}),
            }
        )

    metrics = compute_metrics(details)
    summary = {"model": name, "model_path": model_path, **metrics}
    return round_floats(summary), [round_floats(item) for item in details]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/domain_finreg/detector_eval_finreg_v1.jsonl",
        help="JSONL file with premise/hypothesis/label rows.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        help="Detector candidate as name=path. Defaults compare current FEVER and local asset candidates.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/finreg_detector_eval_v1",
        help="Directory for summary.json and details.jsonl.",
    )
    parser.add_argument("--device", default="cpu", help="Device for detector inference.")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set HF_HUB_OFFLINE=1 during evaluation.",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    rows = load_rows(Path(args.data))
    candidates = parse_candidates(args.candidate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []

    for name, model_path in candidates:
        print(f"Evaluating {name}: {model_path}", flush=True)
        try:
            summary, details = evaluate_candidate(name, model_path, rows, args)
            summaries.append(summary)
            all_details.extend(details)
            print(
                json.dumps(
                    {
                        "model": name,
                        "accuracy": summary["accuracy"],
                        "macro_f1": summary["macro_f1"],
                        "contradiction_recall": summary["contradiction_recall"],
                        "false_contradiction_rate": summary["false_contradiction_rate"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            error_summary = {"model": name, "model_path": model_path, "error": repr(exc)}
            summaries.append(error_summary)
            print(json.dumps(error_summary, sort_keys=True), flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "details.jsonl").open("w", encoding="utf-8") as f:
        for row in all_details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {output_dir / 'summary.json'}")
    print(f"Wrote {output_dir / 'details.jsonl'}")


if __name__ == "__main__":
    main()
