#!/usr/bin/env python3
"""Summarize balanced-vs-focal detector ablation over multi-seed runs."""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


FILE_RE = re.compile(
    r"(?P<domain>health|finreg|disaster)_(?P<variant>balanced|focal)_(?P<set_size>\d+)_seed(?P<seed>\d+)\.json$"
)

METRICS = (
    "abstain_rate",
    "retrieve_more_rate",
    "contradiction_rate",
    "contradiction_prob_mean",
    "uncertainty_mean",
    "source_consistency",
)


@dataclass
class RunMetrics:
    domain: str
    variant: str
    seed: int
    set_size: int
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
        "--set-size",
        type=int,
        default=20,
        help="Question set size encoded in filenames (default: 20).",
    )
    parser.add_argument(
        "--json-out",
        default="evaluation_results/auto_eval/detector_ablation_summary_20.json",
        help="Output path for machine-readable summary.",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/detector_ablation_report.md",
        help="Output path for markdown summary.",
    )
    return parser.parse_args()


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def safe_std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def load_runs(set_size: int) -> List[RunMetrics]:
    files = sorted(
        glob.glob(f"evaluation_results/auto_eval/*_balanced_{set_size}_seed*.json")
        + glob.glob(f"evaluation_results/auto_eval/*_focal_{set_size}_seed*.json")
    )
    runs: List[RunMetrics] = []
    for path in files:
        name = Path(path).name
        match = FILE_RE.search(name)
        if not match:
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        total = int(payload.get("total", 0) or 0)
        actions = payload.get("actions", {}) or {}
        stats = payload.get("stats_all", {}) or {}
        retrieve_more = int(actions.get("retrieve_more", 0) or 0)

        runs.append(
            RunMetrics(
                domain=match.group("domain"),
                variant=match.group("variant"),
                seed=int(match.group("seed")),
                set_size=int(match.group("set_size")),
                total=total,
                abstain_rate=float(payload.get("abstain_rate", 0.0) or 0.0),
                retrieve_more_rate=(retrieve_more / total) if total else 0.0,
                contradiction_rate=float(stats.get("contradiction_rate", 0.0) or 0.0),
                contradiction_prob_mean=float(
                    stats.get("contradiction_prob_mean", 0.0) or 0.0
                ),
                uncertainty_mean=float(stats.get("uncertainty_mean", 0.0) or 0.0),
                source_consistency=float(stats.get("source_consistency", 0.0) or 0.0),
            )
        )
    return runs


def summarize_subset(runs: List[RunMetrics]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for metric in METRICS:
        values = [float(getattr(r, metric)) for r in runs]
        summary[metric] = {
            "mean": safe_mean(values),
            "std": safe_std(values),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
        }
    return summary


def build_group_summaries(
    runs: List[RunMetrics],
) -> Tuple[Dict[str, Dict[str, Dict]], Dict[str, Dict]]:
    domain_variant: Dict[str, Dict[str, Dict]] = {}
    overall_variant: Dict[str, Dict] = {}

    for domain in ("health", "finreg", "disaster"):
        domain_variant[domain] = {}
        for variant in ("balanced", "focal"):
            subset = [r for r in runs if r.domain == domain and r.variant == variant]
            domain_variant[domain][variant] = {
                "count": len(subset),
                "metrics": summarize_subset(subset),
            }

    for variant in ("balanced", "focal"):
        subset = [r for r in runs if r.variant == variant]
        overall_variant[variant] = {
            "count": len(subset),
            "metrics": summarize_subset(subset),
        }

    return domain_variant, overall_variant


def paired_deltas(runs: List[RunMetrics]) -> Dict[str, Dict[str, float]]:
    index = {(r.domain, r.seed, r.variant): r for r in runs}
    deltas: Dict[str, List[float]] = {m: [] for m in METRICS}

    for domain in ("health", "finreg", "disaster"):
        for seed in sorted({r.seed for r in runs}):
            b = index.get((domain, seed, "balanced"))
            f = index.get((domain, seed, "focal"))
            if not b or not f:
                continue
            for metric in METRICS:
                deltas[metric].append(getattr(f, metric) - getattr(b, metric))

    return {
        metric: {
            "mean": safe_mean(values),
            "std": safe_std(values),
            "count": len(values),
        }
        for metric, values in deltas.items()
    }


def _fmt(val: float, digits: int = 3) -> str:
    return f"{val:.{digits}f}"


def _direction(delta: float, eps: float = 1e-6) -> str:
    if delta > eps:
        return "increases"
    if delta < -eps:
        return "decreases"
    return "keeps"


def build_quick_read(
    domain_variant: Dict[str, Dict[str, Dict]],
    deltas: Dict[str, Dict[str, float]],
) -> List[str]:
    abstain_delta = deltas["abstain_rate"]["mean"]
    contradiction_delta = deltas["contradiction_rate"]["mean"]
    cprob_delta = deltas["contradiction_prob_mean"]["mean"]

    lines: List[str] = []
    lines.append(
        f"- Overall, `focal` {_direction(abstain_delta)} abstain "
        f"({abstain_delta:+.3f}) and {_direction(contradiction_delta)} contradiction "
        f"({contradiction_delta:+.3f}) vs `balanced`."
    )
    lines.append(
        f"- Contradiction probability mean shifts by {cprob_delta:+.3f} "
        f"(focal - balanced)."
    )

    risky_domains: List[str] = []
    for domain in ("health", "finreg", "disaster"):
        b = domain_variant[domain]["balanced"]["metrics"]["contradiction_rate"]["mean"]
        f = domain_variant[domain]["focal"]["metrics"]["contradiction_rate"]["mean"]
        if (f - b) >= 0.20:
            risky_domains.append(domain)

    if risky_domains:
        lines.append(
            "- Risk concentration: focal contradiction rises sharply in "
            + ", ".join(f"`{d}`" for d in risky_domains)
            + "."
        )
    else:
        lines.append("- No domain shows a sharp focal contradiction spike (>= +0.20).")

    safer_default = "balanced" if contradiction_delta >= 0 else "focal"
    lines.append(
        f"- Default recommendation for risk control: `{safer_default}`."
    )
    return lines


def build_markdown(
    runs: List[RunMetrics],
    domain_variant: Dict[str, Dict[str, Dict]],
    overall_variant: Dict[str, Dict],
    deltas: Dict[str, Dict[str, float]],
) -> str:
    lines: List[str] = []
    lines.append("# Detector Ablation Report (Balanced vs Focal)")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Runs found: {len(runs)}")
    lines.append("- Domains: `health`, `finreg`, `disaster`")
    lines.append("- Variants: `balanced`, `focal`")
    lines.append("")

    lines.append("## Per-Run Results")
    lines.append("")
    lines.append(
        "| Domain | Variant | Seed | Abstain | Retrieve-More | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run in sorted(runs, key=lambda r: (r.domain, r.variant, r.seed)):
        lines.append(
            f"| {run.domain} | {run.variant} | {run.seed} | {_fmt(run.abstain_rate)} | "
            f"{_fmt(run.retrieve_more_rate)} | {_fmt(run.contradiction_rate)} | "
            f"{_fmt(run.contradiction_prob_mean)} | {_fmt(run.uncertainty_mean, 4)} | "
            f"{_fmt(run.source_consistency)} |"
        )
    lines.append("")

    lines.append("## Domain Summary (mean ± std)")
    lines.append("")
    lines.append(
        "| Domain | Variant | Abstain | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for domain in ("health", "finreg", "disaster"):
        for variant in ("balanced", "focal"):
            metrics = domain_variant[domain][variant]["metrics"]
            lines.append(
                f"| {domain} | {variant} | "
                f"{_fmt(metrics['abstain_rate']['mean'])} ± {_fmt(metrics['abstain_rate']['std'])} | "
                f"{_fmt(metrics['contradiction_rate']['mean'])} ± {_fmt(metrics['contradiction_rate']['std'])} | "
                f"{_fmt(metrics['contradiction_prob_mean']['mean'])} ± {_fmt(metrics['contradiction_prob_mean']['std'])} | "
                f"{_fmt(metrics['uncertainty_mean']['mean'], 4)} ± {_fmt(metrics['uncertainty_mean']['std'], 4)} | "
                f"{_fmt(metrics['source_consistency']['mean'])} ± {_fmt(metrics['source_consistency']['std'])} |"
            )
    lines.append("")

    lines.append("## Overall Summary (all domains, all seeds)")
    lines.append("")
    lines.append(
        "| Variant | Abstain | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for variant in ("balanced", "focal"):
        metrics = overall_variant[variant]["metrics"]
        lines.append(
            f"| {variant} | {_fmt(metrics['abstain_rate']['mean'])} | "
            f"{_fmt(metrics['contradiction_rate']['mean'])} | "
            f"{_fmt(metrics['contradiction_prob_mean']['mean'])} | "
            f"{_fmt(metrics['uncertainty_mean']['mean'], 4)} | "
            f"{_fmt(metrics['source_consistency']['mean'])} |"
        )
    lines.append("")

    lines.append("## Paired Delta (focal - balanced)")
    lines.append("")
    lines.append(
        "| Metric | Mean Delta | Std Delta | Pairs |"
    )
    lines.append("| --- | ---: | ---: | ---: |")
    for metric in (
        "abstain_rate",
        "contradiction_rate",
        "contradiction_prob_mean",
        "uncertainty_mean",
        "source_consistency",
    ):
        lines.append(
            f"| {metric} | {_fmt(deltas[metric]['mean'], 4)} | {_fmt(deltas[metric]['std'], 4)} | {int(deltas[metric]['count'])} |"
        )
    lines.append("")

    lines.append("## Quick Read")
    lines.append("")
    lines.extend(build_quick_read(domain_variant, deltas))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = load_runs(args.set_size)
    if not runs:
        raise SystemExit("No matching balanced/focal run files found.")

    domain_variant, overall_variant = build_group_summaries(runs)
    deltas = paired_deltas(runs)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "set_size": args.set_size,
        "runs": [r.__dict__ for r in sorted(runs, key=lambda r: (r.domain, r.variant, r.seed))],
        "domain_variant_summary": domain_variant,
        "overall_variant_summary": overall_variant,
        "paired_deltas_focal_minus_balanced": deltas,
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = build_markdown(runs, domain_variant, overall_variant, deltas)
    md_out = Path(args.markdown_out)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md, encoding="utf-8")

    print(f"Wrote JSON summary: {json_out}")
    print(f"Wrote markdown report: {md_out}")


if __name__ == "__main__":
    main()
