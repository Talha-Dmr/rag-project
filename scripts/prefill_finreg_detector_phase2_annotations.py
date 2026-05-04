#!/usr/bin/env python3
"""
Create a semi-automatic annotation prefill package for Detector Optimization Phase 2.

This script does NOT write final gold labels. Instead it fills:
- suggested_label
- suggested_error_type
- review_priority
- review_reason
- benchmark-compatible draft rows with blank gold labels

The goal is to reduce manual effort without leaking detector predictions into the final gold set.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


CONTAMINATION_MARKERS = (
    "job title:",
    "project manager",
    "human resources department",
    "essential duties:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefill Phase 2 finreg detector annotations")
    parser.add_argument(
        "--subset",
        default="evaluation_results/finreg_detector_manualeval_v3/stratified_eval_subset.jsonl",
        help="Stratified eval subset JSONL",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant input in the form alias=path/to/per_question.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/finreg_detector_phase2_prefill",
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


def contains_contamination(text: str) -> bool:
    lower = (text or "").lower()
    return any(marker in lower for marker in CONTAMINATION_MARKERS)


def infer_slice(row: Dict[str, Any]) -> str:
    question = (row.get("question") or "").lower()
    answer = (row.get("generated_answer") or "").lower()
    bucket = (row.get("bucket") or "").lower()

    if contains_contamination(answer):
        return "source_mix_up"
    if "threshold" in question or "threshold" in answer or "number" in question:
        return "wrong_number_or_threshold"
    if "outdated" in question or "stale" in answer:
        return "outdated_regulation"
    if "differ" in question or "diverge" in question or "compare" in question:
        return "cross_document_conflict"
    if bucket == "high_contradiction_signal":
        return "direct_contradiction"
    if bucket == "suspicious_cases":
        return "cross_chunk_mismatch"
    if bucket == "high_entailment":
        return "supported_control"
    return "unsupported_overreach"


def infer_suggested_label(row: Dict[str, Any]) -> str:
    answer = row.get("generated_answer") or ""
    bucket = (row.get("bucket") or "").lower()
    contradiction_prob = safe_float(row.get("contradiction_prob_mean"))
    hallucination_topk = safe_float(row.get("hallucination_prob_topk"))

    if contains_contamination(answer):
        return "unsupported"
    if "i don't know" in answer.lower() or "based on the provided context" in answer.lower():
        return "ambiguous"
    if bucket == "high_entailment" and contradiction_prob < 0.03:
        return "supported"
    if bucket == "high_contradiction_signal":
        return "contradicted"
    if bucket == "suspicious_cases" and hallucination_topk >= 0.10:
        return "unsupported"
    if contradiction_prob >= 0.10:
        return "contradicted"
    if hallucination_topk >= 0.08:
        return "unsupported"
    return "partial"


def infer_error_type(row: Dict[str, Any], suggested_label: str) -> str:
    question = (row.get("question") or "").lower()
    answer = row.get("generated_answer") or ""
    if suggested_label == "supported":
        return "none"
    if contains_contamination(answer):
        return "fabricated_fact"
    if "threshold" in question or "number" in question:
        return "wrong_number_or_threshold"
    if "differ" in question or "diverge" in question or "compare" in question:
        return "cross_document_conflict"
    if suggested_label == "partial":
        return "incomplete_reasoning"
    return "misinterpretation"


def infer_review_priority(row: Dict[str, Any], suggested_label: str) -> str:
    answer = row.get("generated_answer") or ""
    bucket = (row.get("bucket") or "").lower()
    if contains_contamination(answer):
        return "p0"
    if bucket in {"high_contradiction_signal", "suspicious_cases"}:
        return "p0"
    if suggested_label in {"contradicted", "unsupported"}:
        return "p1"
    if suggested_label == "partial":
        return "p2"
    return "p3"


def infer_review_reason(row: Dict[str, Any], suggested_label: str) -> str:
    reasons: List[str] = []
    answer = row.get("generated_answer") or ""
    if contains_contamination(answer):
        reasons.append("generation_contamination")
    bucket = (row.get("bucket") or "").lower()
    if bucket:
        reasons.append(f"bucket={bucket}")
    predicted = row.get("predicted_detector_label") or ""
    if predicted:
        reasons.append(f"detector_pred={predicted}")
    reasons.append(f"suggested_label={suggested_label}")
    return "; ".join(reasons)


def build_prefill_row(row: Dict[str, Any]) -> Dict[str, Any]:
    suggested_label = infer_suggested_label(row)
    return {
        "id": row.get("id", ""),
        "question_id": row.get("question_id", ""),
        "detector_variant": row.get("detector_variant", ""),
        "bucket": row.get("bucket", ""),
        "question": row.get("question", ""),
        "generated_answer": row.get("generated_answer", ""),
        "suggested_label": suggested_label,
        "suggested_error_type": infer_error_type(row, suggested_label),
        "review_priority": infer_review_priority(row, suggested_label),
        "review_reason": infer_review_reason(row, suggested_label),
        "proposed_slice": infer_slice(row),
        "predicted_detector_label": row.get("predicted_detector_label", ""),
        "contradiction_prob_mean": safe_float(row.get("contradiction_prob_mean")),
        "hallucination_prob_topk": safe_float(row.get("hallucination_prob_topk")),
        "detector_conflict": safe_float(row.get("detector_conflict")),
        "retrieval_max_score": safe_float(row.get("retrieval_max_score")),
        "retrieval_mean_score": safe_float(row.get("retrieval_mean_score")),
        "gold_label": "",
        "gold_error_type": "",
        "review_notes": "",
    }


def build_benchmark_prefill_row(row: Dict[str, Any], prefill: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"phase2::{row.get('id', '')}",
        "question": row.get("question", ""),
        "answer": row.get("generated_answer", ""),
        "gold_label": "",
        "slice": prefill["proposed_slice"],
        "difficulty": "hard" if prefill["review_priority"] in {"p0", "p1"} else "medium",
        "expected_detector_zone": row.get("bucket", "").replace("suspicious_cases", "suspicious_case"),
        "evidence": [],
        "provenance": {
            "source_type": "manual_eval_v3",
            "source_id": row.get("id", ""),
            "detector_variant": row.get("detector_variant", ""),
            "notes": prefill["review_reason"],
        },
        "annotation": {
            "suggested_label": prefill["suggested_label"],
            "suggested_error_type": prefill["suggested_error_type"],
            "should_abstain": prefill["suggested_label"] in {"unsupported", "contradicted", "ambiguous"},
            "notes": "",
        },
    }


def main() -> None:
    args = parse_args()
    subset_rows = load_jsonl(Path(args.subset))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefill_rows = [build_prefill_row(row) for row in subset_rows]
    benchmark_rows = [
        build_benchmark_prefill_row(row, prefill)
        for row, prefill in zip(subset_rows, prefill_rows)
    ]

    csv_path = output_dir / "annotation_prefill.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prefill_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prefill_rows)

    benchmark_path = output_dir / "benchmark_prefill.jsonl"
    with benchmark_path.open("w", encoding="utf-8") as handle:
        for row in benchmark_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_subset": args.subset,
        "rows": len(prefill_rows),
        "review_priority_counts": {},
        "suggested_label_counts": {},
    }
    for row in prefill_rows:
        summary["review_priority_counts"][row["review_priority"]] = (
            summary["review_priority_counts"].get(row["review_priority"], 0) + 1
        )
        summary["suggested_label_counts"][row["suggested_label"]] = (
            summary["suggested_label_counts"].get(row["suggested_label"], 0) + 1
        )

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote annotation prefill package to {output_dir}")


if __name__ == "__main__":
    main()
