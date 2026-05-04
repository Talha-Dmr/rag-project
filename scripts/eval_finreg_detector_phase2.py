#!/usr/bin/env python3
"""
Evaluate detector outputs against the FinReg Detector Phase 2 benchmark.

Inputs:
- benchmark JSONL with gold labels
- one or more per-question detector output JSONL files

Outputs:
- summary JSON
- per-variant metrics JSON
- comparison markdown
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


RISKY_LABELS = {"unsupported", "contradicted", "partial", "ambiguous"}
POSITIVE_LABELS = {"contradicted"}
NEUTRALISH_LABELS = {"unsupported", "partial", "ambiguous"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector outputs on Phase 2 benchmark")
    parser.add_argument(
        "--benchmark",
        required=True,
        help="Path to benchmark JSONL with gold labels",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Variant input in the form alias=path/to/per_question.jsonl. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/finreg_detector_phase2_eval",
        help="Output directory",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_variant_specs(raw_specs: List[str]) -> List[Tuple[str, Path]]:
    resolved: List[Tuple[str, Path]] = []
    for raw in raw_specs:
        if "=" not in raw:
            raise SystemExit(f"Invalid --variant value: {raw}")
        alias, path_str = raw.split("=", 1)
        resolved.append((alias.strip(), Path(path_str.strip())))
    if not resolved:
        raise SystemExit("At least one --variant alias=path is required.")
    return resolved


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def label_to_contradiction_target(label: str) -> int:
    return 1 if label in POSITIVE_LABELS else 0


def predicted_contradiction_from_label(label: str) -> int:
    return 1 if (label or "").strip().lower() == "contradiction" else 0


def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def compute_ece(probs: List[float], labels: List[int], bins: int = 10) -> float:
    if not probs:
        return 0.0
    total = len(probs)
    ece = 0.0
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        members = [
            (p, y)
            for p, y in zip(probs, labels)
            if (p >= lo and p < hi) or (idx == bins - 1 and p == 1.0)
        ]
        if not members:
            continue
        mean_prob = sum(p for p, _ in members) / len(members)
        mean_acc = sum(y for _, y in members) / len(members)
        ece += (len(members) / total) * abs(mean_prob - mean_acc)
    return ece


def compute_brier(probs: List[float], labels: List[int]) -> float:
    if not probs:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def build_reliability_table(probs: List[float], labels: List[int], bins: int = 10) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx in range(bins):
        lo = idx / bins
        hi = (idx + 1) / bins
        members = [
            (p, y)
            for p, y in zip(probs, labels)
            if (p >= lo and p < hi) or (idx == bins - 1 and p == 1.0)
        ]
        if not members:
            rows.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "count": 0,
                "mean_prob": 0.0,
                "empirical_rate": 0.0,
            })
            continue
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "count": len(members),
            "mean_prob": sum(p for p, _ in members) / len(members),
            "empirical_rate": sum(y for _, y in members) / len(members),
        })
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def score_summary(rows: List[Dict[str, Any]], score_key: str) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[row["gold_label"]].append(safe_float(row.get(score_key)))
    return {label: mean(vals) for label, vals in grouped.items()}


def evaluate_variant(
    benchmark_rows: List[Dict[str, Any]],
    detector_rows: List[Dict[str, Any]],
    alias: str,
) -> Dict[str, Any]:
    detector_by_qid = {row.get("question_id"): row for row in detector_rows if row.get("question_id")}
    joined_rows: List[Dict[str, Any]] = []
    missing = 0

    for bench in benchmark_rows:
        provenance = bench.get("provenance") or {}
        source_id = provenance.get("source_id", "")
        question_id = source_id.split("::")[-1] if "::" in source_id else source_id
        if not question_id:
            continue
        detector = detector_by_qid.get(question_id)
        if detector is None:
            missing += 1
            continue
        joined_rows.append({
            "question_id": question_id,
            "gold_label": bench["gold_label"],
            "slice": bench.get("slice", ""),
            "difficulty": bench.get("difficulty", ""),
            "predicted_detector_label": detector.get("predicted_detector_label", ""),
            "contradiction_prob_mean": safe_float(detector.get("contradiction_prob_mean")),
            "hallucination_prob_topk": safe_float(detector.get("hallucination_prob_topk")),
            "detector_conflict": safe_float(detector.get("detector_conflict")),
            "detector_conflict_consensus": safe_float(detector.get("detector_conflict_consensus")),
            "entailment_prob_mean": safe_float(detector.get("entailment_prob_mean")),
            "neutral_prob_mean": safe_float(detector.get("neutral_prob_mean")),
            "source_consistency": safe_float(detector.get("source_consistency")),
        })

    y_true = [label_to_contradiction_target(row["gold_label"]) for row in joined_rows]
    y_pred = [
        predicted_contradiction_from_label(row["predicted_detector_label"])
        for row in joined_rows
    ]
    contradiction_probs = [row["contradiction_prob_mean"] for row in joined_rows]

    binary = compute_binary_metrics(y_true, y_pred)
    reliability = build_reliability_table(contradiction_probs, y_true)

    contradiction_neutral_confusion = sum(
        1
        for row in joined_rows
        if row["gold_label"] in NEUTRALISH_LABELS and row["predicted_detector_label"] == "contradiction"
    )
    supported_precision_den = sum(
        1 for row in joined_rows if row["predicted_detector_label"] != "contradiction"
    )
    supported_precision_num = sum(
        1
        for row in joined_rows
        if row["predicted_detector_label"] != "contradiction" and row["gold_label"] == "supported"
    )
    supported_precision = (
        supported_precision_num / supported_precision_den if supported_precision_den else 0.0
    )

    slice_breakdown: Dict[str, Dict[str, Any]] = {}
    by_slice: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        by_slice[row["slice"]].append(row)
    for slice_name, rows in by_slice.items():
        slice_true = [label_to_contradiction_target(row["gold_label"]) for row in rows]
        slice_pred = [
            predicted_contradiction_from_label(row["predicted_detector_label"])
            for row in rows
        ]
        slice_breakdown[slice_name] = {
            "count": len(rows),
            "contradiction_prob_mean": mean(row["contradiction_prob_mean"] for row in rows),
            "hallucination_prob_topk_mean": mean(row["hallucination_prob_topk"] for row in rows),
            "detector_conflict_mean": mean(row["detector_conflict"] for row in rows),
            "binary": compute_binary_metrics(slice_true, slice_pred),
        }

    return {
        "variant": alias,
        "matched_rows": len(joined_rows),
        "missing_rows": missing,
        "binary_contradiction_metrics": binary,
        "supported_precision": supported_precision,
        "contradiction_vs_neutral_confusion": contradiction_neutral_confusion,
        "ece_contradiction_prob": compute_ece(contradiction_probs, y_true),
        "brier_contradiction_prob": compute_brier(contradiction_probs, y_true),
        "score_summary_by_label": {
            "contradiction_prob_mean": score_summary(joined_rows, "contradiction_prob_mean"),
            "hallucination_prob_topk": score_summary(joined_rows, "hallucination_prob_topk"),
            "detector_conflict": score_summary(joined_rows, "detector_conflict"),
            "entailment_prob_mean": score_summary(joined_rows, "entailment_prob_mean"),
            "neutral_prob_mean": score_summary(joined_rows, "neutral_prob_mean"),
        },
        "reliability_table": reliability,
        "slice_breakdown": slice_breakdown,
    }


def render_markdown(results: List[Dict[str, Any]], benchmark_path: str) -> str:
    lines: List[str] = []
    variant_list = ", ".join(f"`{row['variant']}`" for row in results)
    lines.append("# FinReg Detector Phase 2 Evaluation")
    lines.append("")
    lines.append(f"- Benchmark: `{benchmark_path}`")
    lines.append(f"- Variants: {variant_list}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Variant | Matched | Missing | F1 Contradiction | Recall | Precision | Supported Precision | ECE | Brier | C-vs-N Confusion |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in results:
        binary = row["binary_contradiction_metrics"]
        lines.append(
            f"| {row['variant']} | {row['matched_rows']} | {row['missing_rows']} | "
            f"{binary['f1']:.3f} | {binary['recall']:.3f} | {binary['precision']:.3f} | "
            f"{row['supported_precision']:.3f} | {row['ece_contradiction_prob']:.3f} | "
            f"{row['brier_contradiction_prob']:.3f} | {row['contradiction_vs_neutral_confusion']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `F1 Contradiction` is computed from hard predicted detector label vs gold `contradicted`.")
    lines.append("- `Supported Precision` measures how often non-contradiction predictions land on truly supported rows.")
    lines.append("- `C-vs-N Confusion` counts neutralish gold rows predicted as contradiction.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    benchmark_rows = load_jsonl(Path(args.benchmark))
    benchmark_rows = [row for row in benchmark_rows if row.get("gold_label")]
    variant_specs = resolve_variant_specs(args.variant)

    results: List[Dict[str, Any]] = []
    for alias, path in variant_specs:
        detector_rows = load_jsonl(path)
        results.append(evaluate_variant(benchmark_rows, detector_rows, alias))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "benchmark": str(Path(args.benchmark)),
        "variants": [row["variant"] for row in results],
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for row in results:
        (output_dir / f"{row['variant']}_metrics.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (output_dir / "report.md").write_text(
        render_markdown(results, args.benchmark),
        encoding="utf-8",
    )
    print(f"Wrote Phase 2 detector evaluation to {output_dir}")


if __name__ == "__main__":
    main()
