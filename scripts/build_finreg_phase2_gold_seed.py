#!/usr/bin/env python3
"""
Build a reviewed gold-seed benchmark from:
- benchmark_prefill.jsonl
- reviewer_notes_draft.csv

Only rows with non-empty gold_label are emitted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FinReg Phase 2 gold seed benchmark")
    parser.add_argument(
        "--benchmark-prefill",
        default="evaluation_results/finreg_detector_phase2_prefill_smoke/benchmark_prefill.jsonl",
        help="Benchmark prefill JSONL path",
    )
    parser.add_argument(
        "--review-csv",
        default="evaluation_results/finreg_detector_phase2_priority_review_smoke/reviewer_notes_draft.csv",
        help="Reviewed CSV with gold fields",
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks/finreg_detector_phase2/v1/benchmark_v1_gold_seed.jsonl",
        help="Output benchmark JSONL path",
    )
    parser.add_argument(
        "--variant-input",
        action="append",
        default=[],
        help=(
            "Optional fallback per-question input in the form alias=path/to/per_question.jsonl. "
            "Can be passed multiple times."
        ),
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


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def resolve_variant_specs(raw_specs: List[str]) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    defaults = [
        "fever_local=evaluation_results/finreg_detector_manualeval_v3/fever_local/per_question.jsonl",
        "targeted_v2=evaluation_results/finreg_detector_manualeval_v3/targeted_v2/per_question.jsonl",
    ]
    for raw in raw_specs or defaults:
        if "=" not in raw:
            raise SystemExit(f"Invalid --variant-input value: {raw}")
        alias, path_str = raw.split("=", 1)
        resolved[alias.strip()] = Path(path_str.strip())
    return resolved


def derive_slice(gold_label: str, gold_error_type: str) -> str:
    if gold_label == "supported":
        return "supported_control"
    if gold_label == "contradicted":
        return "direct_contradiction"
    mapping = {
        "fabricated_fact": "source_mix_up",
        "cross_document_conflict": "cross_document_conflict",
        "wrong_number_or_threshold": "wrong_number_or_threshold",
        "outdated_regulation": "outdated_regulation",
        "misinterpretation": "unsupported_overreach",
        "incomplete_reasoning": "multiple_qa_mixed_support",
    }
    return mapping.get(gold_error_type, "unsupported_overreach")


def derive_expected_zone(gold_label: str) -> str:
    if gold_label == "supported":
        return "high_entailment"
    if gold_label == "contradicted":
        return "high_contradiction_signal"
    if gold_label == "unsupported":
        return "suspicious_case"
    return "uncertain_zone"


def derive_difficulty(review_priority: str) -> str:
    return {
        "p0": "hard",
        "p1": "hard",
        "p2": "medium",
        "p3": "easy",
    }.get(review_priority, "medium")


def build_fallback_row(
    detector_row: Dict[str, Any],
    reviewed: Dict[str, Any],
) -> Dict[str, Any]:
    gold_label = reviewed.get("gold_label", "").strip()
    gold_error_type = reviewed.get("gold_error_type", "").strip()
    evidence = []
    for chunk in (detector_row.get("retrieved_chunks") or [])[:3]:
        evidence.append({
            "source_id": chunk.get("title") or chunk.get("source") or "unknown",
            "snippet": chunk.get("content_preview") or chunk.get("content", "")[:300],
            "page": chunk.get("page"),
        })

    return {
        "id": f"phase2::{detector_row.get('id', '')}",
        "question": detector_row.get("question", reviewed.get("question", "")),
        "answer": detector_row.get("generated_answer", ""),
        "gold_label": gold_label,
        "slice": derive_slice(gold_label, gold_error_type),
        "difficulty": derive_difficulty(reviewed.get("review_priority", "")),
        "expected_detector_zone": derive_expected_zone(gold_label),
        "evidence": evidence,
        "provenance": {
            "source_type": "manual_eval_v3",
            "source_id": detector_row.get("id", ""),
            "detector_variant": detector_row.get("detector_variant", ""),
            "notes": f"fallback_from_per_question; detector_pred={detector_row.get('predicted_detector_label', '')}",
        },
        "annotation": {
            "gold_error_type": gold_error_type,
            "error_type": gold_error_type or "none",
            "should_abstain": gold_label in {"unsupported", "ambiguous"},
            "notes": reviewed.get("review_notes", "").strip(),
            "suggested_label": reviewed.get("suggested_label", "").strip(),
            "suggested_error_type": reviewed.get("suggested_error_type", "").strip(),
        },
    }


def main() -> None:
    args = parse_args()
    benchmark_rows = load_jsonl(Path(args.benchmark_prefill))
    review_rows = load_csv(Path(args.review_csv))
    variant_specs = resolve_variant_specs(args.variant_input)
    review_by_id = {
        row.get("id", ""): row
        for row in review_rows
        if (row.get("gold_label") or "").strip()
    }
    fallback_rows: Dict[str, Dict[str, Any]] = {}
    for alias, path in variant_specs.items():
        for row in load_jsonl(path):
            source_id = row.get("id", "")
            if source_id:
                fallback_rows[source_id] = row

    output_rows: List[Dict[str, Any]] = []
    matched_ids = set()
    for row in benchmark_rows:
        provenance = row.get("provenance") or {}
        source_id = provenance.get("source_id", "")
        reviewed = review_by_id.get(source_id)
        if not reviewed:
            continue
        matched_ids.add(source_id)

        merged = dict(row)
        merged["gold_label"] = reviewed.get("gold_label", "").strip()

        annotation = dict(merged.get("annotation") or {})
        annotation["gold_error_type"] = reviewed.get("gold_error_type", "").strip()
        annotation["notes"] = reviewed.get("review_notes", "").strip()
        merged["annotation"] = annotation

        output_rows.append(merged)

    for source_id, reviewed in review_by_id.items():
        if source_id in matched_ids:
            continue
        fallback = fallback_rows.get(source_id)
        if not fallback:
            continue
        output_rows.append(build_fallback_row(fallback, reviewed))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output_rows)} reviewed gold seed rows to {output_path}")


if __name__ == "__main__":
    main()
