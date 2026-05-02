#!/usr/bin/env python3
"""Split FinRegBench detector rows into artifact-easy and hard-candidate sets."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
SCOPE_MARKERS = {
    "only",
    "all",
    "never",
    "always",
    "unless",
    "except",
    "must",
    "shall",
    "prohibited",
    "required",
}
INVENTED_DETAIL_MARKERS = {
    "email",
    "address",
    "xml",
    "template",
    "vendor",
    "private",
    "provider",
    "watermark",
    "printed",
    "notices",
    "font",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_nested(record: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def record_text(record: dict[str, Any], key: str) -> str:
    return str(get_nested(record, f"input.{key}") or record.get(key) or "")


def words(value: str) -> set[str]:
    return {match.group(0).lower() for match in WORD_RE.finditer(value)}


def numbers(value: str) -> set[str]:
    return {match.group(0).replace(",", ".") for match in NUMBER_RE.finditer(value)}


def support_status(record: dict[str, Any]) -> str:
    return str(get_nested(record, "labels.support_status") or record.get("label") or "unknown")


def generation_method(record: dict[str, Any]) -> str:
    return str(get_nested(record, "metadata.generation_method") or record.get("generation_method") or "unknown")


def source_id(record: dict[str, Any]) -> str:
    return str(get_nested(record, "metadata.source_id") or record.get("doc_id") or "unknown")


def artifact_flags(record: dict[str, Any]) -> dict[str, bool]:
    candidate = record_text(record, "candidate_answer")
    evidence = record_text(record, "evidence_span")
    candidate_lower = " ".join(candidate.lower().split())
    evidence_lower = " ".join(evidence.lower().split())
    candidate_words = words(candidate)
    evidence_words = words(evidence)
    candidate_numbers = numbers(candidate)
    evidence_numbers = numbers(evidence)

    return {
        "exact_evidence_copy": bool(candidate_lower and candidate_lower == evidence_lower),
        "candidate_contains_evidence": bool(evidence_lower and evidence_lower in candidate_lower),
        "evidence_contains_candidate": bool(candidate_lower and candidate_lower in evidence_lower),
        "inserted_scope_marker": bool((candidate_words - evidence_words) & SCOPE_MARKERS),
        "inserted_invented_detail_marker": bool(
            (candidate_words - evidence_words) & INVENTED_DETAIL_MARKERS
        ),
        "number_mismatch": bool(candidate_numbers and evidence_numbers and candidate_numbers != evidence_numbers),
    }


def classify_artifact_level(record: dict[str, Any]) -> tuple[str, list[str]]:
    flags = artifact_flags(record)
    active_flags = [flag for flag, enabled in flags.items() if enabled]
    label = support_status(record)
    method = generation_method(record)

    easy = False
    if label == "supported" and flags["exact_evidence_copy"]:
        easy = True
    elif label == "unsupported" and (
        flags["inserted_invented_detail_marker"] or method == "invented_unstated_detail"
    ):
        easy = True
    elif label == "contradicted" and (
        flags["inserted_scope_marker"] or flags["number_mismatch"]
    ):
        easy = True

    return ("artifact_easy" if easy else "hard_candidate", active_flags)


def annotate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for record in rows:
        artifact_level, flags = classify_artifact_level(record)
        new_record = dict(record)
        metadata = dict(new_record.get("metadata") or {})
        metadata["artifact_level"] = artifact_level
        metadata["artifact_flags"] = flags
        new_record["metadata"] = metadata
        annotated.append(new_record)
    return annotated


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    level_counts: Counter[str] = Counter()
    level_label_counts: dict[str, Counter[str]] = {}
    level_source_counts: dict[str, Counter[str]] = {}
    level_method_counts: dict[str, Counter[str]] = {}

    for record in rows:
        level = str(get_nested(record, "metadata.artifact_level"))
        level_counts[level] += 1
        level_label_counts.setdefault(level, Counter())[support_status(record)] += 1
        level_source_counts.setdefault(level, Counter())[source_id(record)] += 1
        level_method_counts.setdefault(level, Counter())[generation_method(record)] += 1

    return {
        "total_rows": len(rows),
        "artifact_level_counts": dict(sorted(level_counts.items())),
        "artifact_level_label_counts": {
            level: dict(sorted(counts.items()))
            for level, counts in sorted(level_label_counts.items())
        },
        "artifact_level_source_counts": {
            level: dict(sorted(counts.items()))
            for level, counts in sorted(level_source_counts.items())
        },
        "artifact_level_generation_method_counts": {
            level: dict(sorted(counts.items()))
            for level, counts in sorted(level_method_counts.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_eval.jsonl"),
    )
    parser.add_argument(
        "--annotated-output",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_eval_artifact_annotated.jsonl"),
    )
    parser.add_argument(
        "--easy-output",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_artifact_easy.jsonl"),
    )
    parser.add_argument(
        "--hard-output",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_hard_candidate.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_artifact_split_summary.json"),
    )
    args = parser.parse_args()

    rows = annotate_rows(read_jsonl(args.dataset))
    easy_rows = [
        row for row in rows if get_nested(row, "metadata.artifact_level") == "artifact_easy"
    ]
    hard_rows = [
        row for row in rows if get_nested(row, "metadata.artifact_level") == "hard_candidate"
    ]

    write_jsonl(args.annotated_output, rows)
    write_jsonl(args.easy_output, easy_rows)
    write_jsonl(args.hard_output, hard_rows)
    write_json(args.summary, summarize(rows))


if __name__ == "__main__":
    main()
