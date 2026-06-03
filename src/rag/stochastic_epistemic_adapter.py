"""Shared stochastic epistemic adapter scores for RAG gating.

The functions in this module are side-effect free. Replay scripts can use them
for calibration sweeps, while runtime code can use the same formulas when a
config explicitly enables an adapter.
"""

from __future__ import annotations

import math
from typing import Any


ADAPTER_SOURCES = [
    "logit_mi",
    "stochastic_ou",
    "stochastic_langevin",
    "stochastic_mirror_langevin",
    "stochastic_wright_fisher",
    "stochastic_sghmc",
    "stochastic_sgbd",
    "stochastic_prox_langevin",
    "stochastic_sgld",
    "stochastic_adaptive_sgld",
    "dirichlet_simplex",
    "stein_particle",
    "conformal_margin",
    "risk_budgeted_bayesian",
    "retrieval_evaluator_crag",
    "active_retrieval_multicriteria",
    "semantic_entropy_proxy",
    "evidence_instability",
    "energy_margin",
]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def augment_stats_with_evidence_gate(
    stats: dict[str, Any],
    evidence_sampling_gate: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a shallow stats copy enriched with evidence-subset gate fields."""
    if not evidence_sampling_gate:
        return stats
    augmented = dict(stats)
    for key in (
        "subset_answer_rate",
        "subset_retrieve_more_rate",
        "subset_abstain_rate",
        "subset_non_answer_rate",
        "subset_action_instability",
        "baseline_action_stability",
        "answer_include_risk_mean_across_subsets",
        "answer_include_risk_max_across_subsets",
        "support_score_mean_across_subsets",
        "entailment_prob_mean_across_subsets",
        "neutral_prob_mean_across_subsets",
        "contradiction_prob_mean_across_subsets",
        "uncertainty_mean_across_subsets",
        "top2_margin_mean_across_subsets",
        "contradiction_prob_std_mean_across_subsets",
        "label_disagreement_mean_across_subsets",
        "label_entropy_mean_across_subsets",
        "conflict_mass_mean_across_subsets",
        "confidence_mean_across_subsets",
        "answer_completeness_score_mean_across_subsets",
        "context_coverage_mean_across_subsets",
        "prob_vector_instability_across_subsets",
        "source_consistency_mean_across_subsets",
        "retrieval_max_score_mean_across_subsets",
        "retrieval_mean_score_mean_across_subsets",
    ):
        if key in evidence_sampling_gate:
            augmented[key] = evidence_sampling_gate[key]
    return augmented


def compute_epistemic_adapter(
    stats: dict[str, Any],
    u_epi_baseline: float,
    source: str,
) -> float:
    """Compute a replay/runtime-compatible epistemic adapter score."""
    source_key = (source or "logit_mi").strip().lower()
    base = max(0.0, float(u_epi_baseline))
    if source_key in ("logit_mi", "baseline"):
        return base

    contradiction_prob = clamp01(
        to_float(
            stats.get("contradiction_prob_mean"),
            to_float(stats.get("contradiction_prob_mean_across_subsets"), 0.0),
        )
    )
    contradiction_prob_std = clamp01(
        to_float(
            stats.get("contradiction_prob_std"),
            to_float(stats.get("contradiction_prob_std_mean_across_subsets"), 0.0),
        )
    )
    label_disagreement = clamp01(
        to_float(
            stats.get("label_disagreement"),
            to_float(stats.get("label_disagreement_mean_across_subsets"), 0.0),
        )
    )
    neutral_prob_mean = clamp01(
        to_float(
            stats.get("neutral_prob_mean"),
            to_float(stats.get("neutral_prob_mean_across_subsets"), 0.0),
        )
    )
    entailment_prob_mean = clamp01(
        to_float(
            stats.get("entailment_prob_mean"),
            to_float(stats.get("entailment_prob_mean_across_subsets"), 0.0),
        )
    )
    low_confidence = clamp01(
        to_float(
            stats.get("low_confidence"),
            1.0 - to_float(stats.get("confidence_mean_across_subsets"), 1.0),
        )
    )
    conflict_mass = clamp01(
        to_float(
            stats.get("conflict_mass_mean"),
            to_float(stats.get("conflict_mass_mean_across_subsets"), 0.0),
        )
    )
    top2_margin = clamp01(
        to_float(
            stats.get("top2_margin_mean"),
            to_float(stats.get("top2_margin_mean_across_subsets"), 0.0),
        )
    )
    support_score = clamp01(
        to_float(
            stats.get("support_score"),
            to_float(stats.get("support_score_mean_across_subsets"), 1.0),
        )
    )
    answer_completeness = clamp01(
        to_float(
            stats.get("answer_completeness_score"),
            to_float(stats.get("answer_completeness_score_mean_across_subsets"), 1.0),
        )
    )
    context_coverage = clamp01(
        to_float(
            stats.get("context_coverage"),
            to_float(stats.get("context_coverage_mean_across_subsets"), 1.0),
        )
    )
    source_consistency = clamp01(
        to_float(
            stats.get("source_consistency"),
            to_float(stats.get("source_consistency_mean_across_subsets"), 1.0),
        )
    )
    retrieval_max_score = clamp01(
        to_float(
            stats.get("retrieval_max_score"),
            to_float(stats.get("retrieval_max_score_mean_across_subsets"), 0.0),
        )
    )
    retrieval_mean_score = clamp01(
        to_float(
            stats.get("retrieval_mean_score"),
            to_float(stats.get("retrieval_mean_score_mean_across_subsets"), 0.0),
        )
    )
    contradiction_neutral_gap = clamp01(
        max(0.0, to_float(stats.get("contradiction_neutral_gap_mean"), 0.0))
    )
    retrieval_spread = clamp01(max(0.0, retrieval_max_score - retrieval_mean_score))

    total_prob = max(1e-8, entailment_prob_mean + neutral_prob_mean + contradiction_prob)
    p_e = entailment_prob_mean / total_prob
    p_n = neutral_prob_mean / total_prob
    p_c = contradiction_prob / total_prob
    simplex_entropy = 0.0
    for prob in (p_e, p_n, p_c):
        simplex_entropy -= prob * math.log(max(1e-8, prob))
    entropy_norm = clamp01(simplex_entropy / math.log(3.0))
    label_entropy = clamp01(
        to_float(
            stats.get("label_entropy"),
            to_float(stats.get("label_entropy_mean_across_subsets"), entropy_norm),
        )
    )
    simplex_variance = clamp01(
        (p_e * (1.0 - p_e) + p_n * (1.0 - p_n) + p_c * (1.0 - p_c)) / 0.75
    )

    if source_key in ("stochastic_ou", "ou"):
        return max(
            0.0,
            0.80 * base
            + 0.10 * contradiction_prob_std
            + 0.07 * label_disagreement
            + 0.03 * retrieval_spread,
        )

    if source_key in ("stochastic_langevin", "langevin"):
        return max(
            0.0,
            0.60 * base
            + 0.20 * contradiction_prob
            + 0.12 * label_disagreement
            + 0.08 * neutral_prob_mean,
        )

    if source_key in ("stochastic_mirror_langevin", "mirror_langevin", "mla"):
        mirror_tension = 1.0 - entropy_norm
        return max(
            0.0,
            0.58 * base
            + 0.16 * contradiction_prob_std
            + 0.14 * label_disagreement
            + 0.12 * mirror_tension,
        )

    if source_key in ("stochastic_wright_fisher", "wright_fisher", "wf"):
        wf_strength = (p_e * (1.0 - p_e) + p_n * (1.0 - p_n) + p_c * (1.0 - p_c)) / 3.0
        wf_strength = clamp01(wf_strength / 0.25)
        return max(
            0.0,
            0.52 * base
            + 0.26 * wf_strength
            + 0.14 * label_disagreement
            + 0.08 * contradiction_prob_std,
        )

    if source_key in ("stochastic_sghmc", "sghmc"):
        return max(
            0.0,
            0.68 * base
            + 0.12 * contradiction_prob
            + 0.10 * contradiction_prob_std
            + 0.07 * label_disagreement
            + 0.03 * retrieval_spread,
        )

    if source_key in ("stochastic_sgbd", "sgbd", "barker"):
        return max(
            0.0,
            0.58 * base
            + 0.14 * low_confidence
            + 0.12 * contradiction_prob_std
            + 0.10 * label_disagreement
            + 0.06 * neutral_prob_mean,
        )

    if source_key in (
        "stochastic_prox_langevin",
        "prox_langevin",
        "proximal_langevin",
        "reflected_langevin",
    ):
        bounded_tension = clamp01(0.5 * conflict_mass + 0.5 * low_confidence)
        return max(
            0.0,
            0.62 * base
            + 0.12 * bounded_tension
            + 0.10 * contradiction_prob_std
            + 0.08 * label_disagreement
            + 0.08 * retrieval_spread,
        )

    if source_key in ("stochastic_sgld", "sgld"):
        return max(
            0.0,
            0.54 * base
            + 0.18 * contradiction_prob
            + 0.12 * low_confidence
            + 0.10 * contradiction_prob_std
            + 0.06 * retrieval_spread,
        )

    if source_key in ("adaptive_sgld", "stochastic_adaptive_sgld", "asgld"):
        adaptive_noise = clamp01(0.5 * low_confidence + 0.5 * conflict_mass)
        coupled_conflict = clamp01(contradiction_prob * (0.5 + 0.5 * low_confidence))
        return max(
            0.0,
            0.50 * base
            + 0.18 * coupled_conflict
            + 0.14 * adaptive_noise
            + 0.10 * contradiction_prob_std
            + 0.08 * label_disagreement,
        )

    if source_key in ("dirichlet_simplex", "stochastic_dirichlet_simplex", "dirichlet"):
        not_entailment_mass = clamp01(p_n + p_c)
        return max(
            0.0,
            0.44 * base
            + 0.18 * entropy_norm
            + 0.16 * simplex_variance
            + 0.14 * not_entailment_mass
            + 0.08 * label_disagreement,
        )

    if source_key in ("stein_particle", "stochastic_stein_particle", "svgd"):
        particle_repulsion = clamp01(
            0.40 * label_disagreement
            + 0.30 * contradiction_prob_std
            + 0.20 * retrieval_spread
            + 0.10 * (1.0 - top2_margin)
        )
        return max(
            0.0,
            0.48 * base
            + 0.22 * particle_repulsion
            + 0.16 * contradiction_prob
            + 0.08 * neutral_prob_mean
            + 0.06 * conflict_mass,
        )

    if source_key in ("conformal_margin", "stochastic_conformal_margin", "conformal"):
        margin_risk = clamp01(1.0 - top2_margin)
        not_included_mass = clamp01(neutral_prob_mean + contradiction_prob)
        return max(
            0.0,
            0.42 * base
            + 0.22 * margin_risk
            + 0.16 * (1.0 - support_score)
            + 0.12 * not_included_mass
            + 0.08 * contradiction_prob_std,
        )

    if source_key in ("risk_budgeted_bayesian", "stochastic_risk_budgeted_bayes", "rbb"):
        posterior_unsupported = clamp01(
            0.45 * contradiction_prob
            + 0.35 * neutral_prob_mean
            + 0.20 * (1.0 - support_score)
        )
        dispersion = clamp01(
            0.45 * contradiction_prob_std
            + 0.35 * label_disagreement
            + 0.20 * retrieval_spread
        )
        return max(
            0.0,
            0.46 * base
            + 0.24 * posterior_unsupported
            + 0.18 * dispersion
            + 0.12 * low_confidence,
        )

    if source_key in ("retrieval_evaluator_crag", "crag_retrieval", "crag"):
        evidence_quality_risk = clamp01(
            0.22 * (1.0 - retrieval_max_score)
            + 0.18 * (1.0 - retrieval_mean_score)
            + 0.16 * retrieval_spread
            + 0.18 * (1.0 - source_consistency)
            + 0.14 * (1.0 - support_score)
            + 0.06 * (1.0 - context_coverage)
            + 0.06 * (1.0 - answer_completeness)
        )
        return max(
            0.0,
            0.42 * base
            + 0.30 * evidence_quality_risk
            + 0.16 * contradiction_prob
            + 0.12 * label_disagreement,
        )

    if source_key in ("active_retrieval_multicriteria", "uar_multicriteria", "active_retrieval"):
        knowledge_gap = clamp01(0.55 * (1.0 - support_score) + 0.45 * low_confidence)
        evidence_conflict = clamp01(
            0.45 * contradiction_prob
            + 0.25 * label_disagreement
            + 0.20 * conflict_mass
            + 0.10 * contradiction_prob_std
        )
        source_gap = clamp01(
            0.55 * (1.0 - source_consistency)
            + 0.25 * retrieval_spread
            + 0.20 * (1.0 - retrieval_mean_score)
        )
        incompleteness = clamp01(
            0.45 * (1.0 - answer_completeness)
            + 0.35 * (1.0 - context_coverage)
            + 0.20 * (1.0 - support_score)
        )
        criteria_peak = max(knowledge_gap, evidence_conflict, source_gap, incompleteness)
        criteria_mean = clamp01(
            (knowledge_gap + evidence_conflict + source_gap + incompleteness) / 4.0
        )
        return max(
            0.0,
            0.36 * base
            + 0.34 * criteria_peak
            + 0.20 * criteria_mean
            + 0.10 * contradiction_prob_std,
        )

    if source_key in ("semantic_entropy_proxy", "semantic_entropy", "sep_proxy"):
        margin_risk = clamp01(1.0 - top2_margin)
        semantic_dispersion = clamp01(
            0.35 * label_entropy
            + 0.25 * entropy_norm
            + 0.18 * margin_risk
            + 0.12 * label_disagreement
            + 0.10 * (1.0 - support_score)
        )
        return max(
            0.0,
            0.44 * base
            + 0.32 * semantic_dispersion
            + 0.14 * neutral_prob_mean
            + 0.10 * contradiction_prob_std,
        )

    if source_key in ("evidence_instability", "subset_instability", "vector_instability"):
        subset_instability = clamp01(to_float(stats.get("subset_action_instability"), 0.0))
        subset_non_answer = clamp01(to_float(stats.get("subset_non_answer_rate"), 0.0))
        subset_retrieve = clamp01(to_float(stats.get("subset_retrieve_more_rate"), 0.0))
        subset_abstain = clamp01(to_float(stats.get("subset_abstain_rate"), 0.0))
        subset_risk_mean = clamp01(
            to_float(
                stats.get("answer_include_risk_mean_across_subsets"),
                to_float(stats.get("answer_include_risk"), 0.0),
            )
        )
        subset_risk_max = clamp01(
            to_float(
                stats.get("answer_include_risk_max_across_subsets"),
                subset_risk_mean,
            )
        )
        return max(
            0.0,
            0.34 * base
            + 0.24 * subset_instability
            + 0.18 * subset_risk_max
            + 0.12 * subset_risk_mean
            + 0.08 * subset_non_answer
            + 0.04 * max(subset_retrieve, subset_abstain),
        )

    if source_key in ("energy_margin", "semantic_energy_margin", "energy"):
        margin_energy = clamp01(
            0.34 * (1.0 - top2_margin)
            + 0.22 * entropy_norm
            + 0.18 * label_entropy
            + 0.12 * (1.0 - support_score)
            + 0.08 * contradiction_neutral_gap
            + 0.06 * contradiction_prob_std
        )
        return max(
            0.0,
            0.40 * base
            + 0.36 * margin_energy
            + 0.14 * contradiction_prob
            + 0.10 * conflict_mass,
        )

    return base
