#!/usr/bin/env python3
"""Convert FinRegBench rows into NLI detector train/val/test JSONL files."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABEL_TO_ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def page_key(row: dict[str, Any]) -> str:
    pages = row.get("source_pages") or []
    first_page = pages[0] if pages else "unknown"
    return f"{row.get('doc_id', 'unknown')}:{first_page}"


def split_grouped_rows(
    rows: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("split ratios must sum to a positive value")
    ratios = {
        "train": train_ratio / ratio_sum,
        "val": val_ratio / ratio_sum,
        "test": test_ratio / ratio_sum,
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[page_key(row)].append(row)

    groups = list(grouped.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    total_target = {name: ratios[name] * len(rows) for name in ratios}
    label_totals = Counter(row["label"] for row in rows)
    label_targets = {
        split: {label: ratios[split] * count for label, count in label_totals.items()}
        for split in ratios
    }
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in ratios}
    split_label_counts: dict[str, Counter[str]] = {
        name: Counter() for name in ratios
    }

    for group in groups:
        group_labels = Counter(row["label"] for row in group)

        def score(split: str) -> float:
            total_remaining = total_target[split] - len(splits[split])
            label_remaining = sum(
                label_targets[split][label] - split_label_counts[split][label]
                for label in group_labels
            )
            overflow_penalty = max(0.0, len(splits[split]) + len(group) - total_target[split])
            return total_remaining + label_remaining - 2.0 * overflow_penalty

        chosen = max(("train", "val", "test"), key=score)
        splits[chosen].extend(group)
        split_label_counts[chosen].update(group_labels)

    return splits


def to_nli_row(row: dict[str, Any], include_query: bool = True) -> dict[str, Any]:
    label = row["label"]
    if label not in LABEL_TO_ID:
        raise ValueError(f"Unknown label: {label}")

    if include_query:
        hypothesis = (
            f"Question: {row['query']}\n"
            f"Candidate answer: {row['candidate_answer']}"
        )
    else:
        hypothesis = row["candidate_answer"]

    return {
        "premise": row["evidence_span"],
        "hypothesis": hypothesis,
        "label": LABEL_TO_ID[label],
        "metadata": {
            "source": "finregbench",
            "original_id": row["id"],
            "original_label": label,
            "risk_label": 0 if label == "entailment" else 1,
            "doc_id": row.get("doc_id"),
            "source_pages": row.get("source_pages", []),
            "generation_method": row.get("generation_method"),
            "review_status": row.get("review_status"),
        },
    }


def summarize(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["metadata"]["original_label"] for row in rows)
    pages = {page_key_from_nli(row) for row in rows}
    premise_lengths = [len(row["premise"]) for row in rows]
    hypothesis_lengths = [len(row["hypothesis"]) for row in rows]
    return {
        "rows": len(rows),
        "labels": dict(labels),
        "page_groups": len(pages),
        "premise_chars_max": max(premise_lengths) if premise_lengths else 0,
        "hypothesis_chars_max": max(hypothesis_lengths) if hypothesis_lengths else 0,
    }


def page_key_from_nli(row: dict[str, Any]) -> str:
    meta = row.get("metadata") or {}
    pages = meta.get("source_pages") or []
    first_page = pages[0] if pages else "unknown"
    return f"{meta.get('doc_id', 'unknown')}:{first_page}"


def ensure_no_page_overlap(splits: dict[str, list[dict[str, Any]]]) -> None:
    page_sets = {
        split: {page_key_from_nli(row) for row in rows}
        for split, rows in splits.items()
    }
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    overlaps = {
        f"{left}_{right}": len(page_sets[left] & page_sets[right])
        for left, right in pairs
    }
    bad = {name: count for name, count in overlaps.items() if count}
    if bad:
        raise SystemExit(f"page-group leakage across splits: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../FinRegBench/data/finreg_3000_draft.jsonl"),
        help="Path to FinRegBench JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/finregbench_detector"),
        help="Directory for train.jsonl, val.jsonl, and test.jsonl",
    )
    parser.add_argument(
        "--heldout-input",
        type=Path,
        default=Path("../FinRegBench/data/finreg_heldout_cbe_test.jsonl"),
        help=(
            "Optional heldout FinRegBench JSONL. When present, it is converted "
            "to test_heldout_doc.jsonl without being mixed into train/val/test."
        ),
    )
    parser.add_argument(
        "--skip-heldout",
        action="store_true",
        help="Do not convert the optional heldout document test file",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Use only candidate_answer as hypothesis instead of question + answer",
    )
    args = parser.parse_args()

    source_rows = read_jsonl(args.input)
    missing = [row.get("id") for row in source_rows if row.get("label") not in LABEL_TO_ID]
    if missing:
        raise SystemExit(f"Rows with invalid labels: {missing[:5]}")

    raw_splits = split_grouped_rows(
        source_rows,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    nli_splits = {
        split: [to_nli_row(row, include_query=not args.answer_only) for row in split_rows]
        for split, split_rows in raw_splits.items()
    }
    ensure_no_page_overlap(nli_splits)

    for split, split_rows in nli_splits.items():
        write_jsonl(args.output_dir / f"{split}.jsonl", split_rows)

    heldout_rows: list[dict[str, Any]] = []
    heldout_output = args.output_dir / "test_heldout_doc.jsonl"
    if not args.skip_heldout and args.heldout_input.exists():
        raw_heldout_rows = read_jsonl(args.heldout_input)
        heldout_missing = [
            row.get("id") for row in raw_heldout_rows
            if row.get("label") not in LABEL_TO_ID
        ]
        if heldout_missing:
            raise SystemExit(f"Heldout rows with invalid labels: {heldout_missing[:5]}")

        heldout_rows = [
            to_nli_row(row, include_query=not args.answer_only)
            for row in raw_heldout_rows
        ]
        write_jsonl(heldout_output, heldout_rows)

    summary = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "label_mapping": LABEL_TO_ID,
        "hypothesis_format": "candidate_answer" if args.answer_only else "question_and_candidate_answer",
        "splits": {
            split: summarize(split, split_rows)
            for split, split_rows in nli_splits.items()
        },
    }
    if heldout_rows:
        summary["heldout_input"] = str(args.heldout_input)
        summary["heldout_output"] = str(heldout_output)
        summary["splits"]["test_heldout_doc"] = summarize("test_heldout_doc", heldout_rows)

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
