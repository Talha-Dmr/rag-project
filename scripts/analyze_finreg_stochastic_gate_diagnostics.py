#!/usr/bin/env python3
"""Diagnose FinReg stochastic gate calibration and action overlap.

This script is intentionally analysis-only. It helps distinguish real stochastic
signal from scale mismatch by reporting score distributions, calibrated operating
points, and action agreement against a baseline source.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.eval_grounding_proxy import (
    compute_shadow_epistemic,
    compute_shadow_uncertainty,
    decide_shadow_action_2d,
)


DEFAULT_EPI_SOURCES = [
    "logit_mi",
    "stochastic_ou",
    "stochastic_langevin",
    "stochastic_mirror_langevin",
    "stochastic_wright_fisher",
    "stochastic_sghmc",
    "stochastic_sgbd",
    "stochastic_prox_langevin",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def quantile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    return {
        "min": min(values),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def load_labels(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    return {
        str(row.get("id")): row
        for row in read_jsonl(Path(path))
        if row.get("id")
    }


def load_question_ids(path: str) -> dict[str, str]:
    if not path:
        return {}
    mapping: dict[str, str] = {}
    for row in read_jsonl(Path(path)):
        query = str(row.get("query") or "").strip()
        qid = str(row.get("id") or "").strip()
        if query and qid:
            mapping[query] = qid
    return mapping


def expected_action_family(label: dict[str, Any]) -> str:
    task_family = str(label.get("task_family") or "")
    support_level = str(label.get("support_level") or "")
    ambiguity_family = str(label.get("ambiguity_family") or "")

    if task_family == "sanity_anchor" and ambiguity_family in ("none_or_low", ""):
        return "answer"
    if support_level.endswith("_hard"):
        return "retrieve_more"
    if task_family in ("comparison_conflict", "supervisory_escalation"):
        return "retrieve_more"
    return "answer"


def best_runs_by_source(replay_path: str) -> dict[str, dict[str, Any]]:
    if not replay_path:
        return {}
    data = json.loads(Path(replay_path).read_text(encoding="utf-8"))
    runs = data.get("runs") or data.get("results") or []
    best: dict[str, dict[str, Any]] = {}
    for run in runs:
        source = str(run.get("epi_source") or run.get("src") or "")
        if not source:
            continue
        score = run.get("operating_score")
        if score is None:
            score = run.get("balanced_label_aware_utility")
        if score is None:
            score = run.get("label_aware_utility")
        if score is None:
            score = -1.0
        current = best.get(source)
        current_score = None
        if current is not None:
            current_score = current.get("operating_score")
            if current_score is None:
                current_score = current.get("balanced_label_aware_utility")
            if current_score is None:
                current_score = current.get("label_aware_utility")
            if current_score is None:
                current_score = -1.0
        if current is None or float(score) > float(current_score):
            best[source] = run
    return best


def action_for(
    row: dict[str, Any],
    source: str,
    epi_threshold: float,
    ale_threshold: float,
    policy: str,
    uncertainty_formula: str,
) -> tuple[str, float, float]:
    stats = row.get("stats") or {}
    metrics = compute_shadow_uncertainty(stats, formula=uncertainty_formula)
    u_epi_base = float(metrics["u_epi"])
    u_epi = compute_shadow_epistemic(
        stats=stats,
        u_epi_baseline=u_epi_base,
        epi_source=source,
    )
    u_ale = float(metrics["u_ale"])
    action = decide_shadow_action_2d(
        u_epi=u_epi,
        u_ale=u_ale,
        epi_threshold=epi_threshold,
        ale_threshold=ale_threshold,
        policy=policy,
    )
    return action, u_epi, u_ale


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze FinReg stochastic gate calibration and action overlap."
    )
    parser.add_argument("--details", required=True, help="JSONL from eval_grounding_proxy --dump-details.")
    parser.add_argument("--replay", default="", help="Optional replay JSON with threshold sweeps.")
    parser.add_argument("--questions", default="", help="Optional question JSONL for query-to-id mapping.")
    parser.add_argument("--labels", default="", help="Optional FinReg label JSONL.")
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    parser.add_argument("--baseline-source", default="logit_mi")
    parser.add_argument("--baseline-epi-threshold", type=float, default=0.05)
    parser.add_argument("--baseline-ale-threshold", type=float, default=0.40)
    parser.add_argument("--policy", default="epi_coupled_v2")
    parser.add_argument("--uncertainty-formula", default="v2_conflict_aware")
    parser.add_argument(
        "--epi-sources",
        default=",".join(DEFAULT_EPI_SOURCES),
        help="Comma-separated source list for score distribution reporting.",
    )
    args = parser.parse_args()

    rows = read_jsonl(Path(args.details))
    labels_by_id = load_labels(args.labels)
    question_ids = load_question_ids(args.questions)
    sources = [item.strip() for item in args.epi_sources.split(",") if item.strip()]
    calibrated = best_runs_by_source(args.replay)

    score_distributions: dict[str, dict[str, float | None]] = {}
    for source in sources:
        values: list[float] = []
        for row in rows:
            stats = row.get("stats") or {}
            metrics = compute_shadow_uncertainty(stats, formula=args.uncertainty_formula)
            values.append(
                compute_shadow_epistemic(
                    stats=stats,
                    u_epi_baseline=float(metrics["u_epi"]),
                    epi_source=source,
                )
            )
        score_distributions[source] = summarize_values(values)

    baseline_actions: list[str] = []
    question_meta: list[tuple[str, str]] = []
    for row in rows:
        action, _, _ = action_for(
            row=row,
            source=args.baseline_source,
            epi_threshold=args.baseline_epi_threshold,
            ale_threshold=args.baseline_ale_threshold,
            policy=args.policy,
            uncertainty_formula=args.uncertainty_formula,
        )
        baseline_actions.append(action)

        qid = str(row.get("id") or "").strip()
        if not qid:
            qid = question_ids.get(str(row.get("query") or "").strip(), "")
        label = labels_by_id.get(qid, {})
        expected = expected_action_family(label) if label else "unknown"
        question_meta.append((qid, expected))

    calibrated_diagnostics: dict[str, Any] = {}
    for source, run in calibrated.items():
        epi_threshold = float(run.get("epi_threshold", run.get("epi", args.baseline_epi_threshold)))
        ale_threshold = float(run.get("ale_threshold", run.get("ale", args.baseline_ale_threshold)))
        actions: list[str] = []
        by_expected: dict[str, Counter[str]] = defaultdict(Counter)
        disagreements: list[dict[str, Any]] = []

        for idx, row in enumerate(rows):
            action, u_epi, u_ale = action_for(
                row=row,
                source=source,
                epi_threshold=epi_threshold,
                ale_threshold=ale_threshold,
                policy=str(run.get("policy") or args.policy),
                uncertainty_formula=str(run.get("uncertainty_formula") or args.uncertainty_formula),
            )
            actions.append(action)
            qid, expected = question_meta[idx]
            by_expected[expected][action] += 1
            if action != baseline_actions[idx]:
                disagreements.append(
                    {
                        "id": qid,
                        "expected": expected,
                        "baseline_action": baseline_actions[idx],
                        "source_action": action,
                        "u_epi": u_epi,
                        "u_ale": u_ale,
                        "query": row.get("query"),
                    }
                )

        counts = Counter(actions)
        same = sum(1 for left, right in zip(actions, baseline_actions) if left == right)
        calibrated_diagnostics[source] = {
            "epi_threshold": epi_threshold,
            "ale_threshold": ale_threshold,
            "operating_score": run.get("operating_score"),
            "action_counts": dict(counts),
            "answer_rate": counts["answer"] / len(rows) if rows else 0.0,
            "retrieve_more_rate": counts["retrieve_more"] / len(rows) if rows else 0.0,
            "abstain_rate": counts["abstain"] / len(rows) if rows else 0.0,
            "non_answer_rate": (counts["retrieve_more"] + counts["abstain"]) / len(rows) if rows else 0.0,
            "baseline_action_agreement": same / len(rows) if rows else 0.0,
            "baseline_action_agreement_count": same,
            "by_expected": {key: dict(value) for key, value in by_expected.items()},
            "disagreements": disagreements,
        }

    report = {
        "total": len(rows),
        "baseline": {
            "source": args.baseline_source,
            "epi_threshold": args.baseline_epi_threshold,
            "ale_threshold": args.baseline_ale_threshold,
            "policy": args.policy,
            "uncertainty_formula": args.uncertainty_formula,
            "action_counts": dict(Counter(baseline_actions)),
        },
        "score_distributions": score_distributions,
        "calibrated_diagnostics": calibrated_diagnostics,
    }

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
