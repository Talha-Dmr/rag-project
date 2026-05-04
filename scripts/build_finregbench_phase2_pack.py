#!/usr/bin/env python3
"""Build a Phase 2 FinRegBench detector benchmark pack.

Inputs should come from scripts/split_finregbench_by_artifacts.py so each row
has metadata.artifact_level and metadata.artifact_flags.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


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


def get_nested(record: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def set_nested(record: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = record
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def stable_key(record: dict[str, Any], seed: str) -> str:
    row_id = str(record.get("id", ""))
    return hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()


def deterministic_sample(rows: list[dict[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: stable_key(row, seed))
    return ordered[: min(count, len(ordered))]


def support_status(record: dict[str, Any]) -> str:
    return str(get_nested(record, "labels.support_status") or "unknown")


def source_id(record: dict[str, Any]) -> str:
    return str(get_nested(record, "metadata.source_id") or "unknown")


def generation_method(record: dict[str, Any]) -> str:
    return str(get_nested(record, "metadata.generation_method") or "unknown")


def artifact_level(record: dict[str, Any]) -> str:
    return str(get_nested(record, "metadata.artifact_level") or "unknown")


def add_pack_metadata(
    rows: list[dict[str, Any]], *, pack_name: str, review_required: bool
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        new_row = json.loads(json.dumps(row, ensure_ascii=False))
        metadata = dict(new_row.get("metadata") or {})
        metadata["phase2_pack"] = pack_name
        metadata["phase2_review_required"] = review_required
        new_row["metadata"] = metadata
        output.append(new_row)
    return output


def grouped(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def build_smoke_300(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    easy_rows = [row for row in rows if artifact_level(row) == "artifact_easy"]
    by_label = grouped(easy_rows, support_status)
    selected: list[dict[str, Any]] = []
    for label in ("supported", "unsupported", "contradicted"):
        selected.extend(deterministic_sample(by_label.get(label, []), 100, f"{seed}:smoke:{label}"))
    return add_pack_metadata(selected, pack_name="smoke_300", review_required=False)


def build_contradiction_stress(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if artifact_level(row) == "hard_candidate" and support_status(row) == "contradicted"
    ]
    selected = sorted(selected, key=lambda row: str(row.get("id", "")))
    return add_pack_metadata(
        selected, pack_name="contradiction_stress_520", review_required=False
    )


def build_review_180(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    selected_by_id: dict[str, dict[str, Any]] = {}

    # 30 rows for each support label from the artifact-easy pool.
    easy_rows = [row for row in rows if artifact_level(row) == "artifact_easy"]
    by_label = grouped(easy_rows, support_status)
    for label in ("supported", "unsupported", "contradicted"):
        for row in deterministic_sample(
            by_label.get(label, []), 30, f"{seed}:review:easy:{label}"
        ):
            selected_by_id[str(row.get("id"))] = row

    # 60 rows from the hard contradiction stress slice.
    hard_contradictions = [
        row
        for row in rows
        if artifact_level(row) == "hard_candidate" and support_status(row) == "contradicted"
    ]
    for row in deterministic_sample(hard_contradictions, 60, f"{seed}:review:hard_contradicted"):
        selected_by_id[str(row.get("id"))] = row

    # 30 source-balance rows from CCPA, useful because the full set is Basel-heavy.
    ccpa_rows = [row for row in rows if source_id(row) == "ccpa"]
    for row in deterministic_sample(ccpa_rows, 30, f"{seed}:review:ccpa"):
        selected_by_id[str(row.get("id"))] = row

    selected = list(selected_by_id.values())
    if len(selected) < 180:
        remaining = [row for row in rows if str(row.get("id")) not in selected_by_id]
        selected.extend(
            deterministic_sample(remaining, 180 - len(selected), f"{seed}:review:fill")
        )

    selected = sorted(selected[:180], key=lambda row: stable_key(row, f"{seed}:review:final"))
    return add_pack_metadata(selected, pack_name="review_180", review_required=True)


def build_gold_seed_template(review_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in review_rows:
        candidate = {
            "id": row.get("id"),
            "source_pack": "review_180",
            "original_support_status": support_status(row),
            "approved_support_status": None,
            "review_decision": "pending",
            "review_notes": "",
            "record": row,
        }
        output.append(candidate)
    return output


def count_by(rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> dict[str, int]:
    return dict(sorted(Counter(key_fn(row) for row in rows).items()))


def summarize_pack(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "label_counts": count_by(rows, support_status),
        "source_counts": count_by(rows, source_id),
        "generation_method_counts": count_by(rows, generation_method),
        "artifact_level_counts": count_by(rows, artifact_level),
    }


def build_summary(
    source_rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "source_rows": len(source_rows),
        "output_dir": str(output_dir),
        "packs": {
            "smoke_300": summarize_pack(smoke_rows),
            "contradiction_stress_520": summarize_pack(stress_rows),
            "review_180": summarize_pack(review_rows),
        },
        "recommended_usage": {
            "smoke_300": "Fast regression and format smoke testing; artifact-heavy by design.",
            "contradiction_stress_520": "Stress test for subtle contradiction detection only.",
            "review_180": "Manual review queue for building an approved gold seed.",
            "gold_seed_template": "Fill approved_support_status and review_decision after human review.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_eval_artifact_annotated.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("FinRegBench/data/phase2_pack"),
    )
    parser.add_argument("--seed", default="finregbench_phase2_v1")
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    smoke_rows = build_smoke_300(rows, args.seed)
    stress_rows = build_contradiction_stress(rows)
    review_rows = build_review_180(rows, args.seed)
    gold_seed_template = build_gold_seed_template(review_rows)

    write_jsonl(args.output_dir / "smoke_300.jsonl", smoke_rows)
    write_jsonl(args.output_dir / "contradiction_stress_520.jsonl", stress_rows)
    write_jsonl(args.output_dir / "review_180.jsonl", review_rows)
    write_jsonl(args.output_dir / "gold_seed_template.jsonl", gold_seed_template)
    write_json(
        args.output_dir / "phase2_pack_summary.json",
        build_summary(rows, smoke_rows, stress_rows, review_rows, args.output_dir),
    )


if __name__ == "__main__":
    main()
