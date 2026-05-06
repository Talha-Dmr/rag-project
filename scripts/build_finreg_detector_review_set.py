#!/usr/bin/env python3
"""Build a balanced manual-review set from the FinReg detector candidate pool."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("entailment", "neutral", "contradiction")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_sample(
    rows: list[dict[str, Any]],
    per_label: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    selected: list[dict[str, Any]] = []
    for label in LABELS:
        candidates = list(by_label[label])
        if len(candidates) < per_label:
            raise ValueError(f"Not enough rows for {label}: requested {per_label}, found {len(candidates)}")

        # Sort first for deterministic sampling across Python/hash changes.
        candidates.sort(
            key=lambda r: (
                r.get("metadata", {}).get("source_org", ""),
                r.get("metadata", {}).get("theme", ""),
                r.get("id", ""),
            )
        )
        selected.extend(rng.sample(candidates, k=per_label))

    rng.shuffle(selected)
    return selected


def to_review_record(row: dict[str, Any], review_id: str) -> dict[str, Any]:
    metadata = row.get("metadata", {})
    return {
        "review_id": review_id,
        "candidate_id": row.get("id", ""),
        "premise": row.get("premise", ""),
        "hypothesis": row.get("hypothesis", ""),
        "auto_label": row.get("label", ""),
        "review_label": "",
        "keep": "",
        "notes": "",
        "source_org": metadata.get("source_org", ""),
        "theme": metadata.get("theme", ""),
        "pair_type": metadata.get("pair_type", ""),
        "transform": metadata.get("transform", ""),
        "source_file": metadata.get("source_file", ""),
        "review_status": "pending",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "review_id",
        "candidate_id",
        "premise",
        "hypothesis",
        "auto_label",
        "review_label",
        "keep",
        "notes",
        "source_org",
        "theme",
        "pair_type",
        "transform",
        "source_file",
        "review_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "labels": dict(Counter(row["auto_label"] for row in rows)),
        "source_orgs": dict(Counter(row["source_org"] for row in rows)),
        "themes": dict(Counter(row["theme"] for row in rows)),
        "pair_types": dict(Counter(row["pair_type"] for row in rows)),
        "note": "Manual-review set. Fill review_label, keep, and notes before training.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/domain_finreg/detector_candidate_pool_v11.jsonl",
        help="Candidate pool JSONL",
    )
    parser.add_argument("--per-label", type=int, default=30)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--output-dir",
        default="data/domain_finreg/manual_review/detector_v3",
    )
    args = parser.parse_args()

    candidate_rows = load_jsonl(Path(args.input))
    sampled = stratified_sample(candidate_rows, per_label=args.per_label, seed=args.seed)
    review_rows = [
        to_review_record(row, review_id=f"frdr_v3_{idx:04d}")
        for idx, row in enumerate(sampled, start=1)
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "review_set.jsonl"
    csv_path = output_dir / "review_set.csv"
    summary_path = output_dir / "summary.json"

    write_jsonl(jsonl_path, review_rows)
    write_csv(csv_path, review_rows)
    summary_path.write_text(json.dumps(summarize(review_rows), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(review_rows)} review rows")
    print(f"- JSONL: {jsonl_path}")
    print(f"- CSV:   {csv_path}")
    print(f"- Summary: {summary_path}")


if __name__ == "__main__":
    main()
