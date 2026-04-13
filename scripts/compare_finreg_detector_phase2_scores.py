#!/usr/bin/env python3
"""
Compare aggregation/calibration score views on the FinReg Detector Phase 2 benchmark.

This script measures score separation by gold label and runs a simple threshold sweep for
contradiction detection on supported-vs-contradicted comparisons.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASE_SCORE_KEYS = (
    "contradiction_prob_mean",
    "hallucination_prob_topk",
    "detector_conflict",
    "detector_conflict_consensus",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Phase 2 detector aggregation scores")
    parser.add_argument("--benchmark", required=True, help="Benchmark JSONL path")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant input in the form alias=path/to/per_question.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/finreg_detector_phase2_scores",
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


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def derive_scores(row: Dict[str, Any]) -> Dict[str, float]:
    entailment = safe_float(row.get("entailment_prob_mean"))
    neutral = safe_float(row.get("neutral_prob_mean"))
    contradiction = safe_float(row.get("contradiction_prob_mean"))
    return {
        "contradiction_prob_mean": contradiction,
        "hallucination_prob_topk": safe_float(row.get("hallucination_prob_topk")),
        "detector_conflict": safe_float(row.get("detector_conflict")),
        "detector_conflict_consensus": safe_float(row.get("detector_conflict_consensus")),
        "entailment_contradiction_margin": contradiction - entailment,
        "neutral_contradiction_margin": contradiction - neutral,
    }


def threshold_metrics(scores: List[float], labels: List[int], threshold: float) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    for score, label in zip(scores, labels):
        pred = 1 if score >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def build_joined_rows(benchmark_rows: List[Dict[str, Any]], detector_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    detector_by_qid = {row.get("question_id"): row for row in detector_rows if row.get("question_id")}
    joined: List[Dict[str, Any]] = []
    for bench in benchmark_rows:
        provenance = bench.get("provenance") or {}
        source_id = provenance.get("source_id", "")
        question_id = source_id.split("::")[-1] if "::" in source_id else source_id
        if not question_id:
            continue
        detector = detector_by_qid.get(question_id)
        if detector is None:
            continue
        scores = derive_scores(detector)
        joined.append({
            "question_id": question_id,
            "gold_label": bench.get("gold_label", ""),
            **scores,
        })
    return joined


def score_means_by_label(joined_rows: List[Dict[str, Any]], score_key: str) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in joined_rows:
        grouped[row["gold_label"]].append(safe_float(row.get(score_key)))
    return {label: mean(vals) for label, vals in grouped.items()}


def evaluate_score_key(joined_rows: List[Dict[str, Any]], score_key: str) -> Dict[str, Any]:
    contradiction_rows = [
        row for row in joined_rows if row["gold_label"] in {"supported", "contradicted"}
    ]
    labels = [1 if row["gold_label"] == "contradicted" else 0 for row in contradiction_rows]
    scores = [safe_float(row.get(score_key)) for row in contradiction_rows]

    candidate_thresholds = sorted(set(scores))
    if not candidate_thresholds:
        best = {"threshold": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    else:
        best = max(
            (threshold_metrics(scores, labels, thr) for thr in candidate_thresholds),
            key=lambda row: (row["f1"], row["recall"], -row["fp"]),
        )

    return {
        "score_key": score_key,
        "matched_rows": len(joined_rows),
        "supported_vs_contradicted_rows": len(contradiction_rows),
        "means_by_label": score_means_by_label(joined_rows, score_key),
        "best_threshold_supported_vs_contradicted": best,
    }


def render_markdown(variant_results: Dict[str, List[Dict[str, Any]]], benchmark_path: str) -> str:
    lines: List[str] = []
    lines.append("# FinReg Detector Phase 2 Score Comparison")
    lines.append("")
    lines.append(f"- Benchmark: `{benchmark_path}`")
    lines.append("")
    for variant, rows in variant_results.items():
        lines.append(f"## {variant}")
        lines.append("")
        lines.append("| Score | Matched | S-vs-C Rows | Best Threshold | Best F1 | Precision | Recall |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            best = row["best_threshold_supported_vs_contradicted"]
            lines.append(
                f"| {row['score_key']} | {row['matched_rows']} | {row['supported_vs_contradicted_rows']} | "
                f"{best['threshold']:.4f} | {best['f1']:.3f} | {best['precision']:.3f} | {best['recall']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    benchmark_rows = load_jsonl(Path(args.benchmark))
    variant_specs = resolve_variant_specs(args.variant)
    all_results: Dict[str, List[Dict[str, Any]]] = {}

    score_keys = list(BASE_SCORE_KEYS) + [
        "entailment_contradiction_margin",
        "neutral_contradiction_margin",
    ]

    for alias, path in variant_specs:
        detector_rows = load_jsonl(path)
        joined_rows = build_joined_rows(benchmark_rows, detector_rows)
        all_results[alias] = [evaluate_score_key(joined_rows, key) for key in score_keys]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "variants": all_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        render_markdown(all_results, args.benchmark),
        encoding="utf-8",
    )
    print(f"Wrote score comparison to {output_dir}")


if __name__ == "__main__":
    main()
