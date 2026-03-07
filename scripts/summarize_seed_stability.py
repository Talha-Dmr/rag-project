#!/usr/bin/env python3
"""Summarize multi-seed stability from high-stakes eval JSON outputs."""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


FILE_RE = re.compile(
    r"(?P<domain>health|finreg|disaster)_logit_mi_(?P<set_size>\d+)_seed(?P<seed>\d+)(?:_(?P<tag>[^.]+))?\.json$"
)


@dataclass
class RunMetrics:
    domain: str
    seed: int
    set_size: int
    tag: str
    total: int
    abstain_rate: float
    retrieve_more_rate: float
    contradiction_rate: float
    contradiction_prob_mean: float
    uncertainty_mean: float
    source_consistency: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default="evaluation_results/auto_eval/*_logit_mi_50_seed*.json",
        help="Glob for per-run JSON outputs.",
    )
    parser.add_argument(
        "--json-out",
        default="evaluation_results/auto_eval/seed_stability_summary.json",
        help="Output path for machine-readable summary.",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/stability_report.md",
        help="Output path for markdown summary.",
    )
    parser.add_argument(
        "--include-tagged",
        action="store_true",
        help="Include files with suffix tags like _crt095/_thr101.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Only include a specific tag (requires --include-tagged), e.g. crt095.",
    )
    return parser.parse_args()


def load_runs(input_glob: str, include_tagged: bool, tag_filter: str) -> List[RunMetrics]:
    runs: List[RunMetrics] = []
    for path in sorted(glob.glob(input_glob)):
        match = FILE_RE.search(Path(path).name)
        if not match:
            continue
        tag = match.group("tag") or ""
        if tag and not include_tagged:
            continue
        if tag_filter and tag != tag_filter:
            continue

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        total = int(payload.get("total", 0) or 0)
        actions = payload.get("actions", {}) or {}
        stats_all = payload.get("stats_all", {}) or {}
        retrieve_more = int(actions.get("retrieve_more", 0) or 0)
        runs.append(
            RunMetrics(
                domain=match.group("domain"),
                seed=int(match.group("seed")),
                set_size=int(match.group("set_size")),
                tag=tag,
                total=total,
                abstain_rate=float(payload.get("abstain_rate", 0.0) or 0.0),
                retrieve_more_rate=(retrieve_more / total) if total > 0 else 0.0,
                contradiction_rate=float(
                    stats_all.get("contradiction_rate", 0.0) or 0.0
                ),
                contradiction_prob_mean=float(
                    stats_all.get("contradiction_prob_mean", 0.0) or 0.0
                ),
                uncertainty_mean=float(stats_all.get("uncertainty_mean", 0.0) or 0.0),
                source_consistency=float(
                    stats_all.get("source_consistency", 0.0) or 0.0
                ),
            )
        )
    return runs


def summarize_domain(runs: List[RunMetrics], domain: str) -> Dict[str, float]:
    subset = [r for r in runs if r.domain == domain]
    if not subset:
        return {}

    def metric(name: str) -> Dict[str, float]:
        values = [getattr(r, name) for r in subset]
        return {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "count": len(subset),
        "abstain_rate": metric("abstain_rate"),
        "retrieve_more_rate": metric("retrieve_more_rate"),
        "contradiction_rate": metric("contradiction_rate"),
        "contradiction_prob_mean": metric("contradiction_prob_mean"),
        "uncertainty_mean": metric("uncertainty_mean"),
        "source_consistency": metric("source_consistency"),
    }


def build_quick_read(domain_summary: Dict[str, Dict]) -> List[str]:
    lines: List[str] = []
    for domain in ("health", "finreg", "disaster"):
        summary = domain_summary.get(domain, {})
        if not summary:
            continue

        abstain_mean = summary["abstain_rate"]["mean"]
        abstain_std = summary["abstain_rate"]["std"]
        contradiction_mean = summary["contradiction_rate"]["mean"]
        contradiction_std = summary["contradiction_rate"]["std"]

        if abstain_std <= 0.03 and contradiction_std <= 0.02:
            stability = "seed-stable"
        elif abstain_std <= 0.06 and contradiction_std <= 0.05:
            stability = "moderately stable"
        else:
            stability = "seed-sensitive"

        if abstain_mean <= 0.08:
            coverage = "high coverage (low abstain)"
        elif abstain_mean <= 0.20:
            coverage = "balanced coverage"
        else:
            coverage = "conservative coverage (high abstain)"

        if contradiction_mean <= 0.05:
            risk = "low contradiction risk"
        elif contradiction_mean <= 0.15:
            risk = "moderate contradiction risk"
        else:
            risk = "elevated contradiction risk"

        lines.append(f"- `{domain}`: {stability}; {coverage}; {risk}.")

    if lines:
        max_contradiction = max(
            (
                domain_summary[d]["contradiction_rate"]["max"]
                for d in ("health", "finreg", "disaster")
                if domain_summary.get(d)
            ),
            default=0.0,
        )
        if max_contradiction <= 0.05:
            lines.append("- Cross-domain: no contradiction spike observed across seeds.")
        elif max_contradiction <= 0.15:
            lines.append("- Cross-domain: mild contradiction variation across seeds.")
        else:
            lines.append("- Cross-domain: contradiction spikes exist; keep calibration active.")

    return lines


def build_markdown(runs: List[RunMetrics], domain_summary: Dict[str, Dict]) -> str:
    lines: List[str] = []
    lines.append("# Seed Stability Report (High-Stakes 3-Domain)")
    lines.append("")
    set_sizes = sorted({r.set_size for r in runs})
    lines.append(
        f"- Question set size(s): {', '.join(str(s) for s in set_sizes)}"
    )
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Runs found: {len(runs)}")
    lines.append("")
    lines.append("## Per-Run Results")
    lines.append("")
    lines.append(
        "| Domain | Seed | Set | Tag | Abstain | Retrieve-More | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |"
    )
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run in sorted(runs, key=lambda r: (r.domain, r.tag, r.seed, r.set_size)):
        tag_display = run.tag or "-"
        lines.append(
            f"| {run.domain} | {run.seed} | {run.set_size} | {tag_display} | {run.abstain_rate:.3f} | "
            f"{run.retrieve_more_rate:.3f} | {run.contradiction_rate:.3f} | "
            f"{run.contradiction_prob_mean:.3f} | {run.uncertainty_mean:.4f} | "
            f"{run.source_consistency:.3f} |"
        )
    lines.append("")
    lines.append("## Domain Summary (mean ± std, with min/max)")
    lines.append("")
    lines.append(
        "| Domain | Metric | Mean | Std | Min | Max |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for domain in ("health", "finreg", "disaster"):
        summary = domain_summary.get(domain, {})
        if not summary:
            continue
        for metric in (
            "abstain_rate",
            "retrieve_more_rate",
            "contradiction_rate",
            "contradiction_prob_mean",
            "uncertainty_mean",
            "source_consistency",
        ):
            stats = summary[metric]
            lines.append(
                f"| {domain} | {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                f"{stats['min']:.4f} | {stats['max']:.4f} |"
            )
    lines.append("")
    lines.append("## Quick Read")
    lines.append("")
    lines.extend(build_quick_read(domain_summary))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    tag_filter = args.tag.strip()
    if tag_filter and not args.include_tagged:
        raise SystemExit("--tag requires --include-tagged.")

    runs = load_runs(args.input_glob, args.include_tagged, tag_filter)
    if not runs:
        raise SystemExit(f"No matching run files found for: {args.input_glob}")

    domain_summary = {
        domain: summarize_domain(runs, domain)
        for domain in ("health", "finreg", "disaster")
    }

    json_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_glob": args.input_glob,
        "include_tagged": args.include_tagged,
        "tag_filter": tag_filter,
        "runs": [
            r.__dict__
            for r in sorted(runs, key=lambda r: (r.domain, r.tag, r.seed, r.set_size))
        ],
        "domain_summary": domain_summary,
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    md_out = Path(args.markdown_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(build_markdown(runs, domain_summary), encoding="utf-8")

    print(f"Wrote JSON summary: {json_out}")
    print(f"Wrote markdown report: {md_out}")


if __name__ == "__main__":
    main()
