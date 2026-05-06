#!/usr/bin/env python3
"""Build a balanced FinReg detector NLI dataset from reviewed candidate rows."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("entailment", "neutral", "contradiction")
LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}


def load_reviewed_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_nli_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "id": row["review_id"],
        "premise": row["premise"],
        "hypothesis": row["hypothesis"],
        "label": row["review_label"],
        "metadata": {
            "candidate_id": row["candidate_id"],
            "source_org": row["source_org"],
            "theme": row["theme"],
            "pair_type": row["pair_type"],
            "transform": row["transform"],
            "source_file": row["source_file"],
            "review_status": row["review_status"],
            "review_notes": row["notes"],
            "builder": "build_finreg_reviewed_nli_dataset.py",
        },
    }


def select_balanced_rows(
    reviewed_rows: list[dict[str, str]],
    max_per_label: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviewed_rows:
        if row.get("keep") != "yes":
            continue
        label = row.get("review_label", "")
        if label in LABELS:
            by_label[label].append(to_nli_row(row))

    missing = [label for label in LABELS if not by_label[label]]
    if missing:
        raise ValueError(f"No kept rows for labels: {missing}")

    per_label = min(len(by_label[label]) for label in LABELS)
    if max_per_label is not None:
        per_label = min(per_label, max_per_label)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in LABELS:
        rows = sorted(by_label[label], key=lambda r: r["id"])
        selected.extend(rng.sample(rows, k=per_label))

    rng.shuffle(selected)
    return selected


def split_stratified(
    rows: list[dict[str, Any]],
    val_per_label: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for label in LABELS:
        label_rows = list(by_label[label])
        rng.shuffle(label_rows)
        if len(label_rows) <= val_per_label:
            raise ValueError(f"Not enough {label} rows for val_per_label={val_per_label}")
        val_rows.extend(label_rows[:val_per_label])
        train_rows.extend(label_rows[val_per_label:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], numeric_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = dict(row)
            if numeric_labels:
                out["label"] = LABEL_TO_ID[out["label"]]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "source_orgs": dict(Counter(row["metadata"]["source_org"] for row in rows)),
        "themes": dict(Counter(row["metadata"]["theme"] for row in rows)),
        "pair_types": dict(Counter(row["metadata"]["pair_type"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviewed-input",
        default="data/domain_finreg/manual_review/detector_v3/reviewed_set_codex_v1.csv",
    )
    parser.add_argument("--output-dir", default="data/training/nli_dataset_finreg_detector_reviewed_v1")
    parser.add_argument("--test-source", default="data/domain_finreg/detector_eval_finreg_v1.jsonl")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--val-per-label", type=int, default=4)
    parser.add_argument(
        "--numeric-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write labels as 0/1/2. The training loader also accepts string labels.",
    )
    args = parser.parse_args()

    reviewed_rows = load_reviewed_rows(Path(args.reviewed_input))
    selected_rows = select_balanced_rows(reviewed_rows, args.max_per_label, args.seed)
    train_rows, val_rows = split_stratified(selected_rows, args.val_per_label, args.seed)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows, args.numeric_labels)
    write_jsonl(output_dir / "val.jsonl", val_rows, args.numeric_labels)

    test_source = Path(args.test_source)
    if not test_source.exists():
        raise FileNotFoundError(test_source)
    shutil.copyfile(test_source, output_dir / "test.jsonl")

    test_rows = []
    with (output_dir / "test.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_rows.append(json.loads(line))

    summary = {
        "reviewed_input": args.reviewed_input,
        "seed": args.seed,
        "max_per_label": args.max_per_label,
        "val_per_label": args.val_per_label,
        "numeric_labels": args.numeric_labels,
        "selected": summarize(selected_rows),
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "test_source": str(test_source),
        "test": summarize(test_rows),
        "note": "Built only from keep=yes reviewed rows. Test remains the held-out FinReg eval set.",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
