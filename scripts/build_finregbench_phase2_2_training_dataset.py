#!/usr/bin/env python3
"""Build Phase 2.2 training dataset with FinRegBench targeted error seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def stable_key(row: dict[str, Any], seed: str) -> str:
    row_id = str(row.get("id") or json.dumps(row, sort_keys=True, ensure_ascii=False))
    return hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()


def label(row: dict[str, Any]) -> str:
    return str(row.get("label") or "unknown")


def mark_seed(row: dict[str, Any]) -> dict[str, Any]:
    new_row = json.loads(json.dumps(row, ensure_ascii=False))
    metadata = dict(new_row.get("metadata") or {})
    metadata["phase2_2_source"] = "finregbench_targeted_error_seed"
    metadata["phase2_2_weight_group"] = "targeted_seed"
    new_row["metadata"] = metadata
    return new_row


def mark_base(row: dict[str, Any]) -> dict[str, Any]:
    new_row = json.loads(json.dumps(row, ensure_ascii=False))
    metadata = dict(new_row.get("metadata") or {})
    metadata["phase2_2_source"] = metadata.get("phase2_2_source") or "base_training_data"
    metadata["phase2_2_weight_group"] = metadata.get("phase2_2_weight_group") or "base"
    new_row["metadata"] = metadata
    return new_row


def stratified_split(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[label(row)].append(row)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for group_label, group_rows in sorted(groups.items()):
        ordered = sorted(group_rows, key=lambda row: stable_key(row, f"{seed}:{group_label}"))
        total = len(ordered)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        train.extend(ordered[:train_end])
        val.extend(ordered[train_end:val_end])
        test.extend(ordered[val_end:])

    train = sorted(train, key=lambda row: stable_key(row, f"{seed}:train"))
    val = sorted(val, key=lambda row: stable_key(row, f"{seed}:val"))
    test = sorted(test, key=lambda row: stable_key(row, f"{seed}:test"))
    return train, val, test


def cap_seed_rows(
    seed_rows: list[dict[str, Any]],
    *,
    max_seed_rows: int | None,
    seed_multiplier: int,
    seed: str,
) -> list[dict[str, Any]]:
    ordered = sorted(seed_rows, key=lambda row: stable_key(row, f"{seed}:seed_cap"))
    if max_seed_rows is not None:
        ordered = ordered[:max_seed_rows]

    expanded: list[dict[str, Any]] = []
    for copy_index in range(seed_multiplier):
        for row in ordered:
            new_row = json.loads(json.dumps(row, ensure_ascii=False))
            metadata = dict(new_row.get("metadata") or {})
            metadata["phase2_2_seed_copy_index"] = copy_index
            new_row["metadata"] = metadata
            expanded.append(new_row)
    return expanded


def dataset_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter()
    error_type_counts = Counter()
    prefill_counts = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        source_counts[str(metadata.get("phase2_2_weight_group") or "unknown")] += 1
        if metadata.get("error_type"):
            error_type_counts[str(metadata.get("error_type"))] += 1
        if metadata.get("review_prefill"):
            prefill_counts["review_prefill"] += 1

    return {
        "total_examples": len(rows),
        "label_distribution": dict(sorted(Counter(label(row) for row in rows).items())),
        "phase2_2_source_distribution": dict(sorted(source_counts.items())),
        "error_type_distribution": dict(sorted(error_type_counts.items())),
        "review_prefill_count": int(prefill_counts.get("review_prefill", 0)),
    }


def load_base_split(base_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        read_jsonl(base_dir / "train.jsonl"),
        read_jsonl(base_dir / "val.jsonl"),
        read_jsonl(base_dir / "test.jsonl"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed-data",
        type=Path,
        default=Path("FinRegBench/data/phase2_error_mining/targeted_training_seed_prefilled.jsonl"),
    )
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=Path("data/training/nli_dataset_ambigqa_mini_targeted_multipleqa"),
        help="Optional existing train/val/test directory. If missing/empty, seed-only splits are created.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/nli_dataset_finregbench_phase2_2"),
    )
    parser.add_argument("--seed-multiplier", type=int, default=1)
    parser.add_argument("--max-seed-rows", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--random-seed", default="finregbench_phase2_2_training_v1")
    parser.add_argument(
        "--seed-placement",
        choices=["train_only", "stratified"],
        default="train_only",
        help="When base data exists, put seed in train only by default to avoid contaminating eval splits.",
    )
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise SystemExit("train/val/test ratios must sum to 1.0")

    seed_rows = [mark_seed(row) for row in read_jsonl(args.seed_data)]
    seed_rows = cap_seed_rows(
        seed_rows,
        max_seed_rows=args.max_seed_rows,
        seed_multiplier=max(1, args.seed_multiplier),
        seed=args.random_seed,
    )

    base_train, base_val, base_test = load_base_split(args.base_data_dir)
    base_exists = bool(base_train or base_val or base_test)
    base_train = [mark_base(row) for row in base_train]
    base_val = [mark_base(row) for row in base_val]
    base_test = [mark_base(row) for row in base_test]

    if base_exists and args.seed_placement == "train_only":
        train_rows = base_train + seed_rows
        val_rows = base_val
        test_rows = base_test
    else:
        seed_train, seed_val, seed_test = stratified_split(
            seed_rows,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.random_seed,
        )
        train_rows = base_train + seed_train
        val_rows = base_val + seed_val
        test_rows = base_test + seed_test

    random.seed(args.random_seed)
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "val.jsonl", val_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)

    write_json(
        args.output_dir / "dataset_stats.json",
        {
            "base_data_dir": str(args.base_data_dir),
            "base_data_found": base_exists,
            "seed_data": str(args.seed_data),
            "seed_rows_loaded_after_multiplier": len(seed_rows),
            "seed_multiplier": args.seed_multiplier,
            "max_seed_rows": args.max_seed_rows,
            "seed_placement": args.seed_placement,
            "train": dataset_stats(train_rows),
            "val": dataset_stats(val_rows),
            "test": dataset_stats(test_rows),
            "total_examples": len(train_rows) + len(val_rows) + len(test_rows),
        },
    )


if __name__ == "__main__":
    main()
