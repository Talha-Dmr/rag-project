#!/usr/bin/env python3
"""Convert FinRegBench draft JSONL files into a detector-friendly eval format.

The source benchmark is an answer-verification/NLI set with labels:
entailment, contradiction, neutral.  This converter keeps those labels and adds
canonical detector labels used by the FinReg evaluation workflow.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NLI_TO_SUPPORT_STATUS = {
    "entailment": "supported",
    "contradiction": "contradicted",
    "neutral": "unsupported",
}

NLI_TO_BINARY_SUPPORTED = {
    "entailment": True,
    "contradiction": False,
    "neutral": False,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected object row")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def normalize_ambiguity(record: dict[str, Any]) -> str:
    ambiguity_type = str(record.get("ambiguity_type", "") or "").strip().lower()
    if ambiguity_type in {"", "none", "null", "n/a"}:
        return "unambiguous"
    return "ambiguous"


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    nli_label = str(record.get("label", "")).strip().lower()
    if nli_label not in NLI_TO_SUPPORT_STATUS:
        raise ValueError(
            f"{record.get('id', '<missing-id>')}: unsupported label {record.get('label')!r}"
        )

    support_status = NLI_TO_SUPPORT_STATUS[nli_label]
    ambiguity_status = normalize_ambiguity(record)
    detector_labels = [support_status, ambiguity_status]

    if support_status != "supported":
        detector_labels.append("needs_review_or_retrieval")
    if support_status == "contradicted":
        detector_labels.append("conflicting_answer")
    if support_status == "unsupported":
        detector_labels.append("missing_evidence")

    return {
        "id": record.get("id"),
        "split": record.get("split"),
        "task": "finreg_answer_verification",
        "input": {
            "query": record.get("query"),
            "candidate_answer": record.get("candidate_answer"),
            "evidence_span": record.get("evidence_span"),
        },
        "labels": {
            "nli_label": nli_label,
            "support_status": support_status,
            "binary_supported": NLI_TO_BINARY_SUPPORTED[nli_label],
            "ambiguity_status": ambiguity_status,
            "ambiguity_type": record.get("ambiguity_type"),
            "detector_labels": detector_labels,
        },
        "metadata": {
            "source_id": record.get("doc_id"),
            "source_title": record.get("doc_title"),
            "jurisdiction": record.get("jurisdiction"),
            "source_path": record.get("source_path"),
            "source_pages": record.get("source_pages"),
            "topic": record.get("topic"),
            "difficulty": record.get("difficulty"),
            "generation_method": record.get("generation_method"),
            "quality_score": record.get("quality_score"),
            "review_status": record.get("review_status"),
            "expected_label": record.get("expected_label"),
        },
        "raw": record,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    support_counts: Counter[str] = Counter()
    ambiguity_counts: Counter[str] = Counter()
    source_label_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for record in records:
        labels = record["labels"]
        metadata = record["metadata"]
        nli_label = labels["nli_label"]
        source_id = metadata.get("source_id") or "unknown"

        label_counts[nli_label] += 1
        split_counts[record.get("split") or "unknown"] += 1
        source_counts[source_id] += 1
        support_counts[labels["support_status"]] += 1
        ambiguity_counts[labels["ambiguity_status"]] += 1
        source_label_counts[source_id][nli_label] += 1

    return {
        "total_rows": len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "support_status_counts": dict(sorted(support_counts.items())),
        "ambiguity_status_counts": dict(sorted(ambiguity_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "source_label_counts": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(source_label_counts.items())
        },
        "format": {
            "task": "finreg_answer_verification",
            "input_fields": ["query", "candidate_answer", "evidence_span"],
            "primary_labels": [
                "nli_label",
                "support_status",
                "binary_supported",
                "ambiguity_status",
                "detector_labels",
            ],
        },
    }


def convert_file(input_path: Path, output_path: Path, summary_path: Path | None) -> None:
    source_records = read_jsonl(input_path)
    converted = [convert_record(record) for record in source_records]
    write_jsonl(output_path, converted)

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    **summarize(converted),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_draft.jsonl"),
        help="Source FinRegBench draft JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_eval.jsonl"),
        help="Detector-format JSONL output.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_eval_summary.json"),
        help="Summary JSON output. Use an empty string to skip.",
    )
    args = parser.parse_args()

    summary_path = args.summary if str(args.summary) else None
    convert_file(args.input, args.output, summary_path)


if __name__ == "__main__":
    main()
