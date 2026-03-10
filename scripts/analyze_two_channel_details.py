#!/usr/bin/env python3
"""Analyze per-query two-channel uncertainty details dumped by eval_grounding_proxy.

Goal:
- quantify whether risky answered examples are driven more by epistemic or aleatoric factors
- summarize quadrant occupancy under fixed thresholds
- give a compact JSON report for baseline diagnosis
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def avg(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return mean(clean) if clean else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stats = defaultdict(list)
    for row in rows:
        stats["u_epi"].append(row.get("u_epi"))
        stats["u_ale"].append(row.get("u_ale"))
        stats["contradiction_rate"].append(row.get("contradiction_rate"))
        stats["contradiction_prob_mean"].append(row.get("contradiction_prob_mean"))
        stats["source_consistency"].append(row.get("source_consistency"))
        stats["retrieval_spread"].append(row.get("retrieval_spread"))
        stats["low_confidence"].append(row.get("low_confidence"))
    return {
        "count": len(rows),
        "u_epi_mean": avg(stats["u_epi"]),
        "u_ale_mean": avg(stats["u_ale"]),
        "contradiction_rate_mean": avg(stats["contradiction_rate"]),
        "contradiction_prob_mean": avg(stats["contradiction_prob_mean"]),
        "source_consistency_mean": avg(stats["source_consistency"]),
        "retrieval_spread_mean": avg(stats["retrieval_spread"]),
        "low_confidence_mean": avg(stats["low_confidence"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze two-channel per-query dump.")
    parser.add_argument("--details", required=True, help="JSONL from --dump-details")
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    parser.add_argument("--epi-threshold", type=float, required=True)
    parser.add_argument("--ale-threshold", type=float, required=True)
    parser.add_argument(
        "--risky-contradiction-threshold",
        type=float,
        default=0.40,
        help="Per-query contradiction-rate threshold for a risky answered example.",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.details))

    enriched: list[dict[str, Any]] = []
    quadrants = Counter()
    driver_counts = Counter()
    for row in rows:
        stats = row.get("stats") or {}
        contradiction_rate = float(stats.get("contradiction_rate", 0.0) or 0.0)
        contradiction_prob_mean = float(stats.get("contradiction_prob_mean", 0.0) or 0.0)
        source_consistency = float(stats.get("source_consistency", 0.0) or 0.0)
        u_epi = float(row.get("u_epi", 0.0) or 0.0)
        u_ale = float(row.get("u_ale", 0.0) or 0.0)
        is_abstain = bool(row.get("is_abstain", False))
        epi_high = u_epi >= args.epi_threshold
        ale_high = u_ale >= args.ale_threshold

        if epi_high and ale_high:
            quadrant = "high_epi_high_ale"
        elif epi_high and not ale_high:
            quadrant = "high_epi_low_ale"
        elif (not epi_high) and ale_high:
            quadrant = "low_epi_high_ale"
        else:
            quadrant = "low_epi_low_ale"
        quadrants[quadrant] += 1

        risky_answered = (not is_abstain) and contradiction_rate >= args.risky_contradiction_threshold
        safe_answered = (not is_abstain) and not risky_answered

        if risky_answered:
            if u_epi > u_ale + 1e-9:
                driver = "epi_dominant"
            elif u_ale > u_epi + 1e-9:
                driver = "ale_dominant"
            else:
                driver = "balanced"
            driver_counts[driver] += 1
        else:
            driver = None

        enriched.append(
            {
                "query": row.get("query"),
                "action": row.get("action"),
                "is_abstain": is_abstain,
                "u_epi": u_epi,
                "u_ale": u_ale,
                "epi_high": epi_high,
                "ale_high": ale_high,
                "quadrant": quadrant,
                "contradiction_rate": contradiction_rate,
                "contradiction_prob_mean": contradiction_prob_mean,
                "source_consistency": source_consistency,
                "retrieval_spread": float(row.get("retrieval_spread", 0.0) or 0.0),
                "low_confidence": float(row.get("low_confidence", 0.0) or 0.0),
                "risky_answered": risky_answered,
                "safe_answered": safe_answered,
                "dominant_driver": driver,
            }
        )

    risky_answered_rows = [r for r in enriched if r["risky_answered"]]
    safe_answered_rows = [r for r in enriched if r["safe_answered"]]
    abstained_rows = [r for r in enriched if r["is_abstain"]]

    quadrant_breakdown = defaultdict(lambda: {"total": 0, "risky_answered": 0, "safe_answered": 0, "abstain": 0})
    for row in enriched:
        q = quadrant_breakdown[row["quadrant"]]
        q["total"] += 1
        if row["risky_answered"]:
            q["risky_answered"] += 1
        elif row["safe_answered"]:
            q["safe_answered"] += 1
        elif row["is_abstain"]:
            q["abstain"] += 1

    report = {
        "total": len(enriched),
        "epi_threshold": args.epi_threshold,
        "ale_threshold": args.ale_threshold,
        "risky_contradiction_threshold": args.risky_contradiction_threshold,
        "counts": {
            "abstain": len(abstained_rows),
            "safe_answered": len(safe_answered_rows),
            "risky_answered": len(risky_answered_rows),
        },
        "risky_answered_driver_counts": dict(driver_counts),
        "quadrant_counts": dict(quadrants),
        "quadrant_breakdown": dict(quadrant_breakdown),
        "groups": {
            "all": summarize_rows(enriched),
            "abstain": summarize_rows(abstained_rows),
            "safe_answered": summarize_rows(safe_answered_rows),
            "risky_answered": summarize_rows(risky_answered_rows),
        },
        "top_risky_examples": sorted(
            risky_answered_rows,
            key=lambda r: (r["contradiction_rate"], r["u_ale"], r["u_epi"]),
            reverse=True,
        )[:10],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote analysis to {out_path}")


if __name__ == "__main__":
    main()
