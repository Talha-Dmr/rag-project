#!/usr/bin/env python3
"""Audit FinRegBench detector rows for generation artifacts.

The current draft benchmark is auto-generated.  This audit highlights simple
surface patterns that can make the set easier than a real answer-verification
benchmark, such as exact copied evidence, inserted scope words, and invented
detail markers.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
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


def text(record: dict[str, Any], field: str) -> str:
    return str(get_nested(record, f"input.{field}") or record.get(field) or "")


def words(value: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(value)]


def numbers(value: str) -> set[str]:
    return {match.group(0).replace(",", ".") for match in NUMBER_RE.finditer(value)}


def support_status(record: dict[str, Any]) -> str:
    return str(get_nested(record, "labels.support_status") or record.get("label") or "unknown")


def generation_method(record: dict[str, Any]) -> str:
    return str(get_nested(record, "metadata.generation_method") or record.get("generation_method") or "unknown")


def source_id(record: dict[str, Any]) -> str:
    return str(get_nested(record, "metadata.source_id") or record.get("doc_id") or "unknown")


def artifact_flags(record: dict[str, Any]) -> dict[str, bool]:
    candidate = text(record, "candidate_answer")
    evidence = text(record, "evidence_span")
    candidate_lower = " ".join(candidate.lower().split())
    evidence_lower = " ".join(evidence.lower().split())
    candidate_words = set(words(candidate))
    evidence_words = set(words(evidence))

    inserted_scope_markers = (candidate_words - evidence_words) & SCOPE_MARKERS
    inserted_invented_markers = (candidate_words - evidence_words) & INVENTED_DETAIL_MARKERS
    candidate_numbers = numbers(candidate)
    evidence_numbers = numbers(evidence)

    return {
        "exact_evidence_copy": bool(candidate_lower and candidate_lower == evidence_lower),
        "candidate_contains_evidence": bool(evidence_lower and evidence_lower in candidate_lower),
        "evidence_contains_candidate": bool(candidate_lower and candidate_lower in evidence_lower),
        "inserted_scope_marker": bool(inserted_scope_markers),
        "inserted_invented_detail_marker": bool(inserted_invented_markers),
        "number_mismatch": bool(candidate_numbers and evidence_numbers and candidate_numbers != evidence_numbers),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flag_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    by_method: dict[str, Counter[str]] = defaultdict(Counter)

    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in rows:
        label = support_status(record)
        method = generation_method(record)
        source = source_id(record)
        flags = artifact_flags(record)

        label_counts[label] += 1
        method_counts[method] += 1
        source_counts[source] += 1

        for flag, enabled in flags.items():
            if not enabled:
                continue
            flag_counts[flag] += 1
            by_label[label][flag] += 1
            by_method[method][flag] += 1
            if len(examples[flag]) < 5:
                examples[flag].append(
                    {
                        "id": record.get("id"),
                        "support_status": label,
                        "generation_method": method,
                        "source_id": source,
                        "candidate_answer": text(record, "candidate_answer"),
                        "evidence_span": text(record, "evidence_span"),
                    }
                )

    total = len(rows)
    return {
        "total_rows": total,
        "label_counts": dict(sorted(label_counts.items())),
        "generation_method_counts": dict(sorted(method_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "artifact_counts": dict(sorted(flag_counts.items())),
        "artifact_rates": {
            flag: count / total if total else None
            for flag, count in sorted(flag_counts.items())
        },
        "artifacts_by_label": {
            label: dict(sorted(counts.items())) for label, counts in sorted(by_label.items())
        },
        "artifacts_by_generation_method": {
            method: dict(sorted(counts.items())) for method, counts in sorted(by_method.items())
        },
        "examples": dict(sorted(examples.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_eval.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_artifact_audit.json"),
    )
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    write_json(args.output, summarize(rows))


if __name__ == "__main__":
    main()
