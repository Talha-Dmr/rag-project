#!/usr/bin/env python3
"""
Analyze evidence-conflict / aleatoric patterns from eval_grounding_proxy detail dumps.

Input: JSONL produced by scripts/eval_grounding_proxy.py --dump-details
Output: JSON summary with dominant conflict drivers, per-action stats, and top examples.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return float(ordered[idx])


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "source_inconsistency_mean": mean([r["source_inconsistency"] for r in rows]),
        "detector_conflict_mean": mean([r["detector_conflict"] for r in rows]),
        "retrieval_ambiguity_mean": mean([r["retrieval_ambiguity"] for r in rows]),
        "low_confidence_mean": mean([r["low_confidence"] for r in rows]),
        "combined_conflict_mean": mean([r["combined_conflict"] for r in rows]),
        "contradiction_rate_mean": mean([r["contradiction_rate"] for r in rows]),
        "contradiction_prob_mean": mean([r["contradiction_prob_mean"] for r in rows]),
        "u_ale_mean": mean([r["u_ale"] for r in rows]),
        "u_epi_mean": mean([r["u_epi"] for r in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze evidence-conflict patterns from detail dumps")
    parser.add_argument("--input", required=True, help="Path to detail JSONL")
    parser.add_argument("--output", required=True, help="Path to output JSON summary")
    parser.add_argument("--top-k", type=int, default=10, help="Top examples to keep")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    enriched: list[dict[str, Any]] = []
    action_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    abstain_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dominant_counts = Counter()
    dominant_counts_by_action: dict[str, Counter[str]] = defaultdict(Counter)
    all_signal_dominant_counts = Counter()

    for row in rows:
        stats = row.get("stats") or {}
        source_consistency = _clamp01(_to_float(stats.get("source_consistency"), 1.0))
        source_inconsistency = _clamp01(1.0 - source_consistency)
        conflict_mass = _clamp01(_to_float(stats.get("conflict_mass_mean"), 0.0))
        label_disagreement = _clamp01(_to_float(stats.get("label_disagreement"), 0.0))
        retrieval_ambiguity = _clamp01(_to_float(row.get("retrieval_spread"), 0.0))
        low_confidence = _clamp01(_to_float(row.get("low_confidence"), 0.0))
        contradiction_rate = _clamp01(_to_float(stats.get("contradiction_rate"), 0.0))
        contradiction_prob_mean = _clamp01(_to_float(stats.get("contradiction_prob_mean"), 0.0))
        u_ale = _clamp01(_to_float(row.get("u_ale"), 0.0))
        u_epi = _clamp01(_to_float(row.get("u_epi_stochastic", row.get("u_epi", 0.0)), 0.0))

        detector_conflict = _clamp01(
            0.5 * conflict_mass
            + 0.3 * label_disagreement
            + 0.2 * contradiction_prob_mean
        )
        combined_conflict = _clamp01(
            0.4 * source_inconsistency
            + 0.3 * detector_conflict
            + 0.15 * retrieval_ambiguity
            + 0.15 * low_confidence
        )

        evidence_drivers = {
            "source_inconsistency": source_inconsistency,
            "detector_conflict": detector_conflict,
            "retrieval_ambiguity": retrieval_ambiguity,
        }
        all_drivers = {
            **evidence_drivers,
            "low_confidence": low_confidence,
        }
        dominant_driver = max(evidence_drivers.items(), key=lambda item: item[1])[0]
        all_signal_driver = max(all_drivers.items(), key=lambda item: item[1])[0]
        dominant_counts[dominant_driver] += 1
        dominant_counts_by_action[row.get("action", "unknown")][dominant_driver] += 1
        all_signal_dominant_counts[all_signal_driver] += 1

        enriched_row = {
            "query": row.get("query", ""),
            "action": row.get("action", "unknown"),
            "is_abstain": bool(row.get("is_abstain", False)),
            "attempts": int(row.get("attempts", 0) or 0),
            "k_used": row.get("k_used"),
            "u_ale": u_ale,
            "u_epi": u_epi,
            "source_consistency": source_consistency,
            "source_inconsistency": source_inconsistency,
            "detector_conflict": detector_conflict,
            "retrieval_ambiguity": retrieval_ambiguity,
            "low_confidence": low_confidence,
            "combined_conflict": combined_conflict,
            "conflict_mass_mean": conflict_mass,
            "label_disagreement": label_disagreement,
            "contradiction_rate": contradiction_rate,
            "contradiction_prob_mean": contradiction_prob_mean,
            "dominant_driver": dominant_driver,
            "all_signal_driver": all_signal_driver,
        }
        enriched.append(enriched_row)
        action_groups[enriched_row["action"]].append(enriched_row)
        abstain_groups["abstain" if enriched_row["is_abstain"] else "answered"].append(enriched_row)

    source_vals = [r["source_inconsistency"] for r in enriched]
    detector_vals = [r["detector_conflict"] for r in enriched]
    retrieval_vals = [r["retrieval_ambiguity"] for r in enriched]
    combined_vals = [r["combined_conflict"] for r in enriched]

    bucket_counts = Counter()
    for row in enriched:
        source_high = row["source_inconsistency"] >= median(source_vals)
        detector_high = row["detector_conflict"] >= median(detector_vals)
        if source_high and detector_high:
            bucket = "high_source_high_detector"
        elif source_high:
            bucket = "high_source_low_detector"
        elif detector_high:
            bucket = "low_source_high_detector"
        else:
            bucket = "low_source_low_detector"
        row["conflict_bucket"] = bucket
        bucket_counts[bucket] += 1

    top_conflict = sorted(
        enriched,
        key=lambda row: (row["combined_conflict"], row["contradiction_rate"], row["u_ale"]),
        reverse=True,
    )[: args.top_k]

    summary = {
        "input": args.input,
        "total": len(enriched),
        "action_counts": {k: len(v) for k, v in action_groups.items()},
        "abstain_counts": {k: len(v) for k, v in abstain_groups.items()},
        "global_means": summarize_group(enriched),
        "quantiles": {
            "source_inconsistency_median": median(source_vals),
            "detector_conflict_median": median(detector_vals),
            "retrieval_ambiguity_median": median(retrieval_vals),
            "combined_conflict_p75": pct(combined_vals, 0.75),
            "combined_conflict_p90": pct(combined_vals, 0.90),
        },
        "dominant_driver_counts": dict(dominant_counts),
        "all_signal_driver_counts": dict(all_signal_dominant_counts),
        "dominant_driver_by_action": {
            action: dict(counter) for action, counter in dominant_counts_by_action.items()
        },
        "conflict_bucket_counts": dict(bucket_counts),
        "by_action": {
            action: summarize_group(group) for action, group in action_groups.items()
        },
        "by_abstain": {
            name: summarize_group(group) for name, group in abstain_groups.items()
        },
        "top_conflict_examples": top_conflict,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote summary to {out_path}")


if __name__ == "__main__":
    main()
