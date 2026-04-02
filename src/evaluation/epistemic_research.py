"""Helpers for epistemic/shadow evaluation and reporting."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Tuple


DEFAULT_ABSTAIN_BOUNDS: Tuple[float, float] = (0.05, 0.25)
DEFAULT_CONTRADICTION_GUARD = 0.15
DEFAULT_RUNTIME_RATIO_LIMIT = 2.5


def classify_answer_outcome(
    action: str,
    is_abstain: bool,
    contradiction_rate: float,
    risky_contradiction_threshold: float = 0.40,
) -> str:
    """Bucket one query outcome for retrieve-more utility analysis."""
    normalized_action = (action or "none").strip().lower()

    if is_abstain or normalized_action == "abstain":
        return "abstain"

    is_risky = float(contradiction_rate) >= float(risky_contradiction_threshold)
    if normalized_action == "retrieve_more":
        return "retrieve_more_still_risky" if is_risky else "retrieve_more_resolved"
    return "answer_risky" if is_risky else "answer_safe"


def aggregate_question_type_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Count outcome buckets by question type."""
    grouped: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        question_type = str(row.get("question_type") or "unknown")
        outcome = str(row.get("outcome_bucket") or "unknown")
        grouped[question_type][outcome] += 1

    return {
        question_type: dict(sorted(counter.items()))
        for question_type, counter in sorted(grouped.items())
    }


def extract_shadow_run_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    """Normalize the metrics we compare across shadow epistemic runs."""
    shadow = payload.get("shadow_two_channel", {}) or {}
    stats_answer = shadow.get("stats_answer", {}) or {}
    stats_non_abstain = shadow.get("stats_non_abstain", {}) or {}
    stats_all = payload.get("stats_all", {}) or {}

    return {
        "shadow_answer_rate": float(shadow.get("answer_rate", 0.0) or 0.0),
        "shadow_retrieve_more_rate": float(shadow.get("retrieve_more_rate", 0.0) or 0.0),
        "shadow_abstain_rate": float(shadow.get("abstain_rate", 0.0) or 0.0),
        "shadow_answered_contradiction_rate": float(
            stats_non_abstain.get("contradiction_rate", 0.0) or 0.0
        ),
        "shadow_answer_contradiction_rate": float(
            stats_answer.get("contradiction_rate", 0.0) or 0.0
        ),
        "stats_all_contradiction_rate": float(
            stats_all.get("contradiction_rate", 0.0) or 0.0
        ),
        "runtime_mean_seconds": float(payload.get("runtime_mean_seconds", 0.0) or 0.0),
        "runtime_total_seconds": float(payload.get("runtime_total_seconds", 0.0) or 0.0),
        "u_epi_stochastic_mean": float(payload.get("u_epi_stochastic_mean", 0.0) or 0.0),
        "u_ale_mean": float(payload.get("u_ale_mean", 0.0) or 0.0),
    }


def evaluate_shadow_candidate(
    candidate_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    abstain_bounds: Tuple[float, float] = DEFAULT_ABSTAIN_BOUNDS,
    contradiction_guard: float = DEFAULT_CONTRADICTION_GUARD,
    runtime_ratio_limit: float = DEFAULT_RUNTIME_RATIO_LIMIT,
) -> Dict[str, Any]:
    """Apply stage-1/2 PASS gates for a shadow epistemic candidate."""
    result: Dict[str, Any] = {
        "passed_stage_0": True,
        "passed_stage_1": False,
        "passed_stage_2": False,
        "candidate_pass_stage": "stage0_feasible",
        "reasons": [],
    }

    answered_contra = float(candidate_metrics.get("shadow_answered_contradiction_rate", 1.0))
    baseline_answered_contra = float(
        baseline_metrics.get("shadow_answered_contradiction_rate", 1.0)
    )
    abstain_rate = float(candidate_metrics.get("shadow_abstain_rate", 1.0))
    all_contradiction = float(candidate_metrics.get("stats_all_contradiction_rate", 1.0))
    min_abstain, max_abstain = abstain_bounds

    if answered_contra > baseline_answered_contra + 1e-12:
        result["reasons"].append("answered_contradiction_worse_than_baseline")
    if abstain_rate < min_abstain or abstain_rate > max_abstain:
        result["reasons"].append("abstain_rate_out_of_band")
    if all_contradiction > contradiction_guard:
        result["reasons"].append("contradiction_guard_failed")

    if not result["reasons"]:
        result["passed_stage_1"] = True
        result["candidate_pass_stage"] = "stage1_fast_pass"
    else:
        return result

    baseline_runtime = float(baseline_metrics.get("runtime_mean_seconds", 0.0))
    candidate_runtime = float(candidate_metrics.get("runtime_mean_seconds", 0.0))
    if baseline_runtime > 0.0:
        runtime_ratio = candidate_runtime / baseline_runtime
    else:
        runtime_ratio = 1.0 if candidate_runtime <= 0.0 else float("inf")
    result["runtime_ratio"] = runtime_ratio

    if runtime_ratio > runtime_ratio_limit:
        result["reasons"].append("runtime_ratio_failed")
        return result

    result["passed_stage_2"] = True
    result["candidate_pass_stage"] = "stage2_cost_pass"
    return result

