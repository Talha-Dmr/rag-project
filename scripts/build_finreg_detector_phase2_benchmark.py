#!/usr/bin/env python3
"""
Build the FinReg detector Phase 2 benchmark JSONL from manual annotation artifacts.

This script converts the existing manual-eval sheet into the new benchmark schema and
emits authoring placeholders when a target bucket is under-filled.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


LABEL_TARGETS = {
    "supported": 40,
    "unsupported": 35,
    "contradicted": 45,
    "partial": 25,
    "ambiguous": 25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FinReg detector benchmark for Detector Optimization Phase 2"
    )
    parser.add_argument(
        "--annotations",
        default="evaluation_results/finreg_detector_manualeval_v3/manual_annotation_sheet.csv",
        help="Path to the filled manual annotation CSV",
    )
    parser.add_argument(
        "--output",
        default="data/benchmarks/finreg_detector_phase2/v1/benchmark_v1_from_manual_eval.jsonl",
        help="Output JSONL path",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def normalize_label(raw: str) -> str:
    return (raw or "").strip().lower()


def infer_slice(row: Dict[str, Any]) -> str:
    bucket = (row.get("bucket") or "").strip().lower()
    label = normalize_label(row.get("label", ""))
    error_type = (row.get("error_type") or "").strip().lower()
    notes = (row.get("notes") or "").strip().lower()

    if label == "supported":
        return "supported_control"
    if error_type == "wrong_number_or_threshold":
        return "wrong_number_or_threshold"
    if error_type == "outdated_regulation":
        return "outdated_regulation"
    if error_type == "cross_document_conflict":
        return "cross_document_conflict"
    if "source mix" in notes or "source mix-up" in notes or "source mixup" in notes:
        return "source_mix_up"
    if "multiple" in notes or "multi-hop" in notes or "compound" in notes:
        return "multiple_qa_mixed_support"
    if bucket == "high_contradiction_signal" and label == "contradicted":
        return "direct_contradiction"
    if bucket == "suspicious_cases" and label in {"contradicted", "unsupported"}:
        return "cross_chunk_mismatch"
    if label == "unsupported":
        return "unsupported_overreach"
    if label == "partial":
        return "multiple_qa_mixed_support"
    if label == "ambiguous":
        return "cross_chunk_mismatch"
    return "unsupported_overreach"


def infer_difficulty(row: Dict[str, Any], slice_name: str) -> str:
    bucket = (row.get("bucket") or "").strip().lower()
    if slice_name in {"cross_document_conflict", "cross_chunk_mismatch", "multiple_qa_mixed_support"}:
        return "hard"
    if bucket in {"suspicious_cases", "high_contradiction_signal"}:
        return "hard"
    if normalize_label(row.get("label", "")) == "supported":
        return "easy"
    return "medium"


def infer_expected_zone(row: Dict[str, Any], label: str) -> str:
    bucket = (row.get("bucket") or "").strip().lower()
    if bucket == "high_entailment" or label == "supported":
        return "high_entailment"
    if bucket == "high_contradiction_signal":
        return "high_contradiction_signal"
    if bucket == "suspicious_cases":
        return "suspicious_case"
    return "uncertain_zone"


def build_record(row: Dict[str, Any]) -> Dict[str, Any]:
    label = normalize_label(row.get("label", ""))
    slice_name = infer_slice(row)
    error_type = (row.get("error_type") or "").strip() or "none"
    should_abstain = label in {"unsupported", "contradicted", "ambiguous"}
    return {
        "id": f"phase2::{row['id']}",
        "question": row.get("question", ""),
        "answer": row.get("generated_answer", ""),
        "gold_label": label,
        "slice": slice_name,
        "difficulty": infer_difficulty(row, slice_name),
        "expected_detector_zone": infer_expected_zone(row, label),
        "evidence": [],
        "provenance": {
            "source_type": "manual_eval_v3",
            "source_id": row.get("id", ""),
            "detector_variant": row.get("detector_variant", ""),
            "notes": (
                f"bucket={row.get('bucket', '')}; "
                f"predicted_label={row.get('predicted_detector_label', '')}"
            ),
        },
        "annotation": {
            "error_type": error_type,
            "should_abstain": should_abstain,
            "notes": row.get("notes", ""),
        },
    }


def build_placeholder(label: str, index: int) -> Dict[str, Any]:
    slice_defaults = {
        "supported": "supported_control",
        "unsupported": "unsupported_overreach",
        "contradicted": "direct_contradiction",
        "partial": "multiple_qa_mixed_support",
        "ambiguous": "cross_chunk_mismatch",
    }
    zone_defaults = {
        "supported": "high_entailment",
        "unsupported": "uncertain_zone",
        "contradicted": "high_contradiction_signal",
        "partial": "uncertain_zone",
        "ambiguous": "uncertain_zone",
    }
    return {
        "id": f"phase2::placeholder::{label}::{index:02d}",
        "question": "",
        "answer": "",
        "gold_label": label,
        "slice": slice_defaults[label],
        "difficulty": "hard" if label in {"contradicted", "ambiguous", "partial"} else "medium",
        "expected_detector_zone": zone_defaults[label],
        "evidence": [],
        "provenance": {
            "source_type": "authored_phase2",
            "source_id": "",
            "detector_variant": "",
            "notes": "Authoring placeholder emitted because this label bucket was under-filled.",
        },
        "annotation": {
            "error_type": "none" if label == "supported" else "",
            "should_abstain": label in {"unsupported", "contradicted", "ambiguous"},
            "notes": "",
        },
    }


def main() -> None:
    args = parse_args()
    rows = load_annotations(Path(args.annotations))
    labeled_rows = [
        row for row in rows
        if normalize_label(row.get("label", "")) in LABEL_TARGETS
    ]

    output_rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in labeled_rows:
        grouped[normalize_label(row.get("label", ""))].append(row)

    for label, target in LABEL_TARGETS.items():
        source_rows = grouped.get(label, [])
        for row in source_rows[:target]:
            output_rows.append(build_record(row))
        if len(source_rows) < target:
            for idx in range(1, (target - len(source_rows)) + 1):
                output_rows.append(build_placeholder(label, idx))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output_rows)} benchmark rows to {output_path}")
    if not labeled_rows:
        print("No labeled rows found; emitted placeholder-only benchmark scaffold.")


if __name__ == "__main__":
    main()
