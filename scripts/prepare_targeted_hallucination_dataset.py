#!/usr/bin/env python3
"""
Prepare a targeted continuation dataset for the hallucination detector.

Default target slice:
- entailment: multipleQAs supported examples
- neutral: multipleQAs mismatch examples
- contradiction: multipleQAs fabricated examples

This is meant for short continuation training after a broader base run.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_bucket_key(row: dict) -> str | None:
    label = row["label"]
    meta = row.get("metadata") or {}
    ann_type = meta.get("ann_type")

    if ann_type != "multipleQAs":
        return None
    if label == 0 and not meta.get("fabricated") and not meta.get("mismatch"):
        return "entailment_multipleqas"
    if label == 1 and meta.get("mismatch"):
        return "neutral_multipleqas_mismatch"
    if label == 2 and meta.get("fabricated"):
        return "contradiction_multipleqas_fabricated"
    return None


def build_targeted_split(rows: list[dict], seed: int) -> tuple[list[dict], dict[str, int]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = select_bucket_key(row)
        if key is not None:
            buckets[key].append(row)

    if len(buckets) != 3:
        raise ValueError(f"Expected 3 populated buckets, got {sorted(buckets)}")

    rng = random.Random(seed)
    min_size = min(len(bucket_rows) for bucket_rows in buckets.values())

    selected: list[dict] = []
    bucket_sizes: dict[str, int] = {}
    for key, bucket_rows in sorted(buckets.items()):
        pool = list(bucket_rows)
        rng.shuffle(pool)
        chosen = pool[:min_size]
        selected.extend(chosen)
        bucket_sizes[key] = len(chosen)

    rng.shuffle(selected)
    return selected, bucket_sizes


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict]) -> dict[str, object]:
    label_counts = Counter(row["label"] for row in rows)
    ann_counts = Counter((row.get("metadata") or {}).get("ann_type") for row in rows)
    return {
        "total": len(rows),
        "label_counts": dict(label_counts),
        "ann_type_counts": dict(ann_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare targeted detector continuation dataset")
    parser.add_argument("--input-dir", required=True, help="Directory with train/val/test jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory to write targeted split")
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report: dict[str, object] = {"seed": args.seed, "input_dir": str(input_dir)}

    for split in ["train", "val", "test"]:
        rows = load_rows(input_dir / f"{split}.jsonl")
        selected, bucket_sizes = build_targeted_split(rows, seed=args.seed)
        write_jsonl(output_dir / f"{split}.jsonl", selected)
        report[split] = {
            "bucket_sizes": bucket_sizes,
            "summary": summarize(selected),
        }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote targeted dataset to {output_dir}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
