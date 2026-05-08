#!/usr/bin/env python3
"""Build the FinReg detector hardmix training split.

The hardmix split keeps the original FinRegBench detector data and adds reviewed
real-RAG-like detector examples from data/domain_finreg/manual_review/detector_v3.
The controlled benchmark cases are not used for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_key(row: dict[str, Any]) -> str:
    raw = row.get("review_id") or row.get("candidate_id") or json.dumps(row, sort_keys=True)
    return hashlib.sha1(str(raw).encode("utf-8")).hexdigest()


def convert_reviewed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("keep", "")).strip().lower() != "yes":
            continue
        label = str(row.get("review_label") or "").strip().lower()
        if label not in LABEL_TO_ID:
            continue
        converted.append(
            {
                "id": row.get("review_id"),
                "premise": row.get("premise"),
                "hypothesis": row.get("hypothesis"),
                "label": LABEL_TO_ID[label],
                "metadata": {
                    "source": "manual_review_detector_v3_codex_v1",
                    "candidate_id": row.get("candidate_id"),
                    "review_label": label,
                    "source_org": row.get("source_org"),
                    "theme": row.get("theme"),
                    "pair_type": row.get("pair_type"),
                    "source_file": row.get("source_file"),
                    "review_status": row.get("review_status"),
                },
            }
        )
    return sorted(converted, key=stable_key)


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(row["label"] for row in rows).items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("data/training/finregbench_detector"))
    parser.add_argument(
        "--reviewed",
        type=Path,
        default=Path("data/domain_finreg/manual_review/detector_v3/reviewed_set_codex_v1.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/training/finregbench_detector_v3_hardmix"))
    parser.add_argument("--hard-val-size", type=int, default=18)
    parser.add_argument("--oversample", type=int, default=6)
    args = parser.parse_args()

    base_train = read_jsonl(args.base_dir / "train.jsonl")
    base_val = read_jsonl(args.base_dir / "val.jsonl")
    reviewed_rows = convert_reviewed_rows(read_jsonl(args.reviewed))

    hard_val = reviewed_rows[: args.hard_val_size]
    hard_train = reviewed_rows[args.hard_val_size :]

    hard_train_augmented: list[dict[str, Any]] = []
    for aug_idx in range(args.oversample):
        for row in hard_train:
            augmented = dict(row)
            augmented["id"] = f"{row['id']}_aug{aug_idx + 1}"
            augmented["metadata"] = dict(row.get("metadata", {}), hardmix_aug=aug_idx + 1)
            hard_train_augmented.append(augmented)

    train_rows = base_train + hard_train_augmented
    val_rows = base_val + hard_val

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "val.jsonl", val_rows)
    shutil.copyfile(args.base_dir / "test.jsonl", args.output_dir / "test.jsonl")
    shutil.copyfile(args.base_dir / "test_heldout_doc.jsonl", args.output_dir / "test_heldout_doc.jsonl")

    summary = {
        "base_data_dir": str(args.base_dir),
        "reviewed_source": str(args.reviewed),
        "output_dir": str(args.output_dir),
        "strategy": (
            "base FinRegBench train/val plus reviewed detector_v3 hard examples; "
            "hard train oversampled; controlled benchmark not used for training"
        ),
        "counts": {
            "base_train": len(base_train),
            "hard_train_unique": len(hard_train),
            "hard_train_after_oversample": len(hard_train_augmented),
            "train_total": len(train_rows),
            "base_val": len(base_val),
            "hard_val": len(hard_val),
            "val_total": len(val_rows),
        },
        "label_counts": {
            "train": label_counts(train_rows),
            "val": label_counts(val_rows),
            "hard_train_unique": label_counts(hard_train),
            "hard_val": label_counts(hard_val),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
