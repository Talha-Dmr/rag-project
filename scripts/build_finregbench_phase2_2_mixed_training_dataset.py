#!/usr/bin/env python3
"""Build a mixed Phase 2.2 training dataset.

This combines a balanced FinRegBench background pool with the targeted
error-mined seed. Phase 2 eval packs are excluded from the background pool by
ID, so smoke/stress runs remain useful for regression checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABEL_BY_STATUS = {
    "supported": "entailment",
    "unsupported": "neutral",
    "contradicted": "contradiction",
}


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
    row_id = str(row.get("id") or json.dumps(row, ensure_ascii=False, sort_keys=True))
    return hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()


def row_label(row: dict[str, Any]) -> str:
    return str(row.get("label") or "unknown")


def pack_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            if row.get("id"):
                ids.add(str(row["id"]))
    return ids


def detector_row_to_nli(row: dict[str, Any]) -> dict[str, Any]:
    input_payload = row.get("input") or {}
    labels = row.get("labels") or {}
    metadata = row.get("metadata") or {}
    raw = row.get("raw") or {}
    support_status = str(labels.get("support_status") or "")
    nli_label = str(labels.get("nli_label") or LABEL_BY_STATUS.get(support_status) or "")
    if nli_label not in {"entailment", "neutral", "contradiction"}:
        raise ValueError(f"Cannot map label for row {row.get('id')}")

    out_metadata = dict(metadata)
    out_metadata.update(
        {
            "source": "finregbench_phase2_2_background",
            "phase2_2_source": "finregbench_background_excluding_eval_packs",
            "phase2_2_weight_group": "background",
            "support_status": support_status,
            "nli_label": nli_label,
        }
    )
    if raw.get("split"):
        out_metadata["original_split"] = raw.get("split")

    return {
        "id": str(row.get("id")),
        "premise": str(input_payload.get("evidence_span") or ""),
        "hypothesis": str(input_payload.get("candidate_answer") or ""),
        "label": nli_label,
        "metadata": out_metadata,
    }


def mark_seed(row: dict[str, Any], copy_index: int) -> dict[str, Any]:
    new_row = json.loads(json.dumps(row, ensure_ascii=False))
    metadata = dict(new_row.get("metadata") or {})
    metadata["phase2_2_source"] = "finregbench_targeted_error_seed"
    metadata["phase2_2_weight_group"] = "targeted_seed"
    metadata["phase2_2_seed_copy_index"] = copy_index
    new_row["metadata"] = metadata
    return new_row


def stratified_take(
    rows: list[dict[str, Any]],
    *,
    per_label: int,
    seed: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_label(row)].append(row)

    selected: list[dict[str, Any]] = []
    for label, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: stable_key(row, f"{seed}:{label}"))
        selected.extend(ordered[:per_label])
    return selected


def stratified_split(
    rows: list[dict[str, Any]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_label(row)].append(row)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: stable_key(row, f"{seed}:split:{label}"))
        train_end = int(len(ordered) * train_ratio)
        val_end = train_end + int(len(ordered) * val_ratio)
        train.extend(ordered[:train_end])
        val.extend(ordered[train_end:val_end])
        test.extend(ordered[val_end:])
    return train, val, test


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter()
    artifact_counts = Counter()
    error_counts = Counter()
    for row in rows:
        metadata = row.get("metadata") or {}
        source_counts[str(metadata.get("phase2_2_weight_group") or "unknown")] += 1
        if metadata.get("artifact_level"):
            artifact_counts[str(metadata.get("artifact_level"))] += 1
        if metadata.get("error_type"):
            error_counts[str(metadata.get("error_type"))] += 1
    return {
        "rows": len(rows),
        "label_counts": dict(sorted(Counter(row_label(row) for row in rows).items())),
        "weight_group_counts": dict(sorted(source_counts.items())),
        "artifact_level_counts": dict(sorted(artifact_counts.items())),
        "error_type_counts": dict(sorted(error_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--background-data",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_eval_artifact_annotated.jsonl"),
    )
    parser.add_argument(
        "--seed-data",
        type=Path,
        default=Path("FinRegBench/data/phase2_error_mining/targeted_training_seed_prefilled.jsonl"),
    )
    parser.add_argument(
        "--exclude-pack",
        type=Path,
        action="append",
        default=[
            Path("FinRegBench/data/phase2_pack/smoke_300.jsonl"),
            Path("FinRegBench/data/phase2_pack/contradiction_stress_520.jsonl"),
            Path("FinRegBench/data/phase2_pack/review_180.jsonl"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/nli_dataset_finregbench_phase2_2_mixed"),
    )
    parser.add_argument("--background-per-label", type=int, default=420)
    parser.add_argument("--seed-multiplier", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.88)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--test-ratio", type=float, default=0.04)
    parser.add_argument("--random-seed", default="finregbench_phase2_2_mixed_v1")
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise SystemExit("train/val/test ratios must sum to 1.0")

    excluded_ids = pack_ids(args.exclude_pack)
    background_rows = []
    for row in read_jsonl(args.background_data):
        if str(row.get("id")) in excluded_ids:
            continue
        background_rows.append(detector_row_to_nli(row))
    background_rows = stratified_take(
        background_rows,
        per_label=args.background_per_label,
        seed=args.random_seed,
    )

    seed_source = read_jsonl(args.seed_data)
    seed_rows: list[dict[str, Any]] = []
    for copy_index in range(max(1, args.seed_multiplier)):
        seed_rows.extend(mark_seed(row, copy_index) for row in seed_source)

    mixed_rows = background_rows + seed_rows
    train_rows, val_rows, test_rows = stratified_split(
        mixed_rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.random_seed,
    )

    random.seed(args.random_seed)
    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)

    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "val.jsonl", val_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)
    write_json(
        args.output_dir / "dataset_stats.json",
        {
            "background_data": str(args.background_data),
            "seed_data": str(args.seed_data),
            "excluded_pack_ids": len(excluded_ids),
            "background_rows_selected": len(background_rows),
            "seed_rows_selected_after_multiplier": len(seed_rows),
            "background_per_label": args.background_per_label,
            "seed_multiplier": args.seed_multiplier,
            "train": stats(train_rows),
            "val": stats(val_rows),
            "test": stats(test_rows),
            "total_rows": len(train_rows) + len(val_rows) + len(test_rows),
        },
    )
    print(f"Wrote mixed Phase 2.2 dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
