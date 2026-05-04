#!/usr/bin/env python3
"""
Export high-priority review subsets from the Phase 2 annotation prefill package.

Default behavior:
- include `p0` and `p1`
- write both CSV and benchmark-style JSONL
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export high-priority Phase 2 review subset")
    parser.add_argument(
        "--prefill-csv",
        default="evaluation_results/finreg_detector_phase2_prefill_smoke/annotation_prefill.csv",
        help="Prefill CSV path",
    )
    parser.add_argument(
        "--benchmark-prefill",
        default="evaluation_results/finreg_detector_phase2_prefill_smoke/benchmark_prefill.jsonl",
        help="Benchmark prefill JSONL path",
    )
    parser.add_argument(
        "--priorities",
        default="p0,p1",
        help="Comma-separated review priorities to include",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/finreg_detector_phase2_priority_review",
        help="Output directory",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    wanted: Set[str] = {item.strip() for item in args.priorities.split(",") if item.strip()}

    csv_rows = load_csv(Path(args.prefill_csv))
    benchmark_rows = load_jsonl(Path(args.benchmark_prefill))

    filtered_csv = [row for row in csv_rows if row.get("review_priority") in wanted]
    wanted_ids = {row.get("id", "") for row in filtered_csv}

    filtered_benchmark = []
    for row in benchmark_rows:
        provenance = row.get("provenance") or {}
        if provenance.get("source_id", "") in wanted_ids:
            filtered_benchmark.append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_out = output_dir / "priority_review.csv"
    if filtered_csv:
        with csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(filtered_csv[0].keys()))
            writer.writeheader()
            writer.writerows(filtered_csv)
    else:
        with csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "review_priority"])

    jsonl_out = output_dir / "priority_review.jsonl"
    with jsonl_out.open("w", encoding="utf-8") as handle:
        for row in filtered_benchmark:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "prefill_csv": args.prefill_csv,
        "benchmark_prefill": args.benchmark_prefill,
        "priorities": sorted(wanted),
        "priority_rows": len(filtered_csv),
        "benchmark_rows": len(filtered_benchmark),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote priority review subset to {output_dir}")


if __name__ == "__main__":
    main()
