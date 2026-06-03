"""Evidence-subset sampling policies for RAG gating.

These policies are intentionally small and side-effect free so shadow eval
scripts and the runtime pipeline can use the same decision rules.
"""

from __future__ import annotations

from typing import Any

from src.rag.stochastic_epistemic_adapter import compute_epistemic_adapter


def canonical_action(action: str) -> str:
    return "answer" if action in ("none", "answer") else action


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_closed_probe(text: str) -> bool:
    first = (text or "").strip().lower().split(" ", 1)[0]
    return first in {"can", "could", "does", "do", "is", "are", "should", "will", "would"}


def _is_low_evidence_probe(text: str) -> bool:
    query = f" {(text or '').strip().lower()} "
    return (
        query.strip().startswith("if ")
        and " not " in query
        and any(marker in query for marker in (" how should ", " what should ", " should the system "))
    )


def _has_direct_refutation(text: str) -> bool:
    answer = f" {(text or '').strip().lower()} "
    markers = (
        " no,",
        " no ",
        " not ",
        " does not ",
        " do not ",
        " is not ",
        " are not ",
        " cannot ",
        " can't ",
        " should not ",
        " without ",
        " no evidence ",
        " not stated ",
        " not establish ",
        " does not establish ",
        " not supported ",
    )
    return any(marker in answer for marker in markers)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _vector_answer_risk(row: dict[str, Any]) -> float:
    """Continuous detector-vector risk, independent of hard predicted labels."""
    risk_mean = _float(row, "answer_include_risk_mean_across_subsets")
    support_mean = _float(row, "support_score_mean_across_subsets", 1.0)
    entailment_mean = _float(row, "entailment_prob_mean_across_subsets")
    neutral_mean = _float(row, "neutral_prob_mean_across_subsets")
    contradiction_mean = _float(row, "contradiction_prob_mean_across_subsets")
    uncertainty_mean = _float(row, "uncertainty_mean_across_subsets")
    vector_instability = _float(row, "prob_vector_instability_across_subsets")
    top2_margin = _float(row, "top2_margin_mean_across_subsets")
    not_included_mass = _clamp01(neutral_mean + contradiction_mean)

    return _clamp01(
        0.34 * risk_mean
        + 0.22 * not_included_mass
        + 0.16 * (1.0 - support_mean)
        + 0.12 * (1.0 - entailment_mean)
        + 0.08 * uncertainty_mean
        + 0.08 * vector_instability
        - 0.04 * top2_margin
    )


def _adapter_baseline_uncertainty(row: dict[str, Any]) -> float:
    return _float(row, "uncertainty_mean_across_subsets", _float(row, "uncertainty_mean", 0.0))


def _adapter_score(row: dict[str, Any], source: str) -> float:
    return compute_epistemic_adapter(
        stats=row,
        u_epi_baseline=_adapter_baseline_uncertainty(row),
        source=source,
    )


def decide_policy(row: dict[str, Any], policy: str) -> tuple[str, str]:
    qtype = str(row.get("type") or row.get("question_type") or "")
    baseline = str(row.get("baseline_action") or "none")

    if row.get("candidate_unavailable"):
        return baseline, "candidate_unavailable"

    answer_rate = _float(row, "subset_answer_rate")
    non_answer_rate = _float(row, "subset_non_answer_rate")
    instability = _float(row, "subset_action_instability")
    risk_mean = _float(row, "answer_include_risk_mean_across_subsets")
    risk_max = _float(row, "answer_include_risk_max_across_subsets")
    support_mean = _float(row, "support_score_mean_across_subsets", 1.0)
    entailment_mean = _float(row, "entailment_prob_mean_across_subsets")
    neutral_mean = _float(row, "neutral_prob_mean_across_subsets")
    contradiction_mean = _float(row, "contradiction_prob_mean_across_subsets")
    vector_instability = _float(row, "prob_vector_instability_across_subsets")
    query = str(row.get("query") or row.get("question") or "")
    answer = str(row.get("answer") or row.get("candidate_answer") or "")

    if policy == "baseline":
        return baseline, "baseline"

    if policy == "subset_majority":
        if non_answer_rate > answer_rate:
            return "retrieve_more", "subset_majority_non_answer"
        return "none", "subset_majority_answer"

    if policy == "guarded_v1":
        if qtype == "sanity":
            if non_answer_rate >= 0.75 and risk_max >= 0.66:
                return "retrieve_more", "sanity_stable_risky"
            return "none", "sanity_guarded_answer"

        if non_answer_rate >= 0.75 and risk_mean >= 0.66:
            return "abstain", "conflict_stable_high_risk"
        if non_answer_rate >= 0.50 or instability >= 0.50:
            return "retrieve_more", "conflict_unstable_or_non_answer"
        if answer_rate >= 0.75 and risk_max < 0.66:
            return "none", "conflict_stable_safe"
        return baseline, "conflict_fallback_baseline"

    if policy == "guarded_v2":
        if qtype == "sanity":
            if non_answer_rate >= 0.75 and risk_max >= 0.66:
                return "retrieve_more", "sanity_stable_risky"
            if non_answer_rate >= 0.50 and risk_max >= 0.68:
                return "retrieve_more", "sanity_high_risk_tie"
            return "none", "sanity_guarded_answer"

        if non_answer_rate >= 1.0 and risk_mean >= 0.66:
            return "abstain", "conflict_all_subsets_risky"
        if non_answer_rate >= 0.50:
            return "retrieve_more", "conflict_non_answer_majority_or_tie"
        if instability >= 0.50 and risk_max >= 0.64:
            return "retrieve_more", "conflict_high_instability"
        if answer_rate >= 0.75 and risk_max < 0.66:
            return "none", "conflict_stable_safe"
        return baseline, "conflict_fallback_baseline"

    if policy == "guarded_v3":
        # First-pass sequential policy:
        # - subset instability/high risk can request more evidence
        # - it should not abstain immediately because risk may still be reducible
        # - final abstain should be handled by a later evidence-budget stage
        if qtype == "sanity":
            if non_answer_rate >= 0.75 and risk_max >= 0.66:
                return "retrieve_more", "sanity_stable_risky"
            return "none", "sanity_guarded_answer"

        if non_answer_rate >= 0.50:
            return "retrieve_more", "conflict_subset_non_answer"
        if instability >= 0.50 and risk_max >= 0.64:
            return "retrieve_more", "conflict_high_instability"
        if answer_rate >= 0.75 and risk_max < 0.66:
            return "none", "conflict_stable_safe"
        return baseline, "conflict_fallback_baseline"

    if policy == "guarded_v4":
        # FinReg-tuned first-pass policy:
        # - sanity anchors should remain answerable even when the detector is noisy
        # - conflict questions should request more evidence when subset risk is
        #   materially high, even if a majority of subsets still answer
        if qtype == "sanity":
            return "none", "sanity_guarded_answer"

        if non_answer_rate >= 0.50:
            return "retrieve_more", "conflict_subset_non_answer"
        if risk_max >= 0.64:
            return "retrieve_more", "conflict_high_answer_risk"
        if instability >= 0.50 and risk_max >= 0.62:
            return "retrieve_more", "conflict_high_instability"
        if answer_rate >= 0.75 and risk_max < 0.64:
            return "none", "conflict_stable_safe"
        return baseline, "conflict_fallback_baseline"

    if policy == "guarded_v4_lite":
        # Less conservative variant of guarded_v4. This keeps the strong sanity
        # guard but requires slightly higher conflict risk before overriding an
        # otherwise stable answer.
        if qtype == "sanity":
            return "none", "sanity_guarded_answer"

        if non_answer_rate >= 0.50:
            return "retrieve_more", "conflict_subset_non_answer"
        if risk_max >= 0.65:
            return "retrieve_more", "conflict_high_answer_risk"
        if instability >= 0.50 and risk_max >= 0.63:
            return "retrieve_more", "conflict_high_instability"
        if answer_rate >= 0.75 and risk_max < 0.65:
            return "none", "conflict_stable_safe"
        return baseline, "conflict_fallback_baseline"

    if policy == "guarded_v5":
        # Conservative production candidate:
        # - preserve an existing non-answer when subset checks agree with it
        # - request more evidence only when subsets show non-answer pressure or
        #   a combined low-support/high-neutral/high-risk pattern
        # - avoid risk-only gating, which over-triggers on correct FinReg answers
        if canonical_action(baseline) != "answer":
            if non_answer_rate >= 0.50:
                return baseline, "baseline_non_answer_confirmed"
            return baseline, "baseline_non_answer_preserved"

        if non_answer_rate >= 0.50:
            return "retrieve_more", "subset_non_answer_majority"
        if non_answer_rate >= 0.20 and risk_mean >= 0.85:
            return "retrieve_more", "partial_subset_non_answer_high_risk"
        if risk_mean >= 0.85 and neutral_mean >= 0.93 and support_mean <= 0.30:
            return "retrieve_more", "low_support_high_neutral_risk"
        if instability >= 0.25 and risk_max >= 0.85:
            return "retrieve_more", "unstable_high_risk"
        return baseline, "stable_or_insufficient_signal"

    if policy == "guarded_v6":
        # guarded_v5 plus a closed-question refutation guard. This catches
        # yes/no regulatory probes where the answer discusses related evidence
        # but never directly rejects the unsupported premise.
        if _is_closed_probe(query) and not _has_direct_refutation(answer) and risk_mean >= 0.75:
            return "abstain", "closed_probe_without_refutation"

        if canonical_action(baseline) != "answer":
            if non_answer_rate >= 0.50:
                return baseline, "baseline_non_answer_confirmed"
            return baseline, "baseline_non_answer_preserved"

        if non_answer_rate >= 0.50:
            return "retrieve_more", "subset_non_answer_majority"
        if non_answer_rate >= 0.20 and risk_mean >= 0.85:
            return "retrieve_more", "partial_subset_non_answer_high_risk"
        if risk_mean >= 0.85 and neutral_mean >= 0.93 and support_mean <= 0.30:
            return "retrieve_more", "low_support_high_neutral_risk"
        if instability >= 0.25 and risk_max >= 0.85:
            return "retrieve_more", "unstable_high_risk"
        return baseline, "stable_or_insufficient_signal"

    if policy == "vector_v3":
        # guarded_v6 with a continuous detector-vector layer. This uses the
        # entailment/neutral/contradiction probability distribution across
        # sampled evidence subsets, not only the final hard detector action.
        closed_without_refutation = _is_closed_probe(query) and not _has_direct_refutation(answer)
        not_included_mass = _clamp01(neutral_mean + contradiction_mean)
        vector_risk = _vector_answer_risk(row)

        if closed_without_refutation and vector_risk >= 0.64 and not_included_mass >= 0.76:
            return "abstain", "closed_probe_vector_not_included"
        if (
            _is_low_evidence_probe(query)
            and vector_risk >= 0.68
            and risk_mean >= 0.80
            and support_mean <= 0.20
            and not_included_mass >= 0.82
        ):
            return "abstain", "low_evidence_vector_not_included"

        if canonical_action(baseline) != "answer":
            if non_answer_rate >= 0.50:
                return baseline, "baseline_non_answer_confirmed"
            if vector_risk >= 0.78 and support_mean <= 0.22:
                return baseline, "baseline_non_answer_vector_confirmed"
            return baseline, "baseline_non_answer_preserved"

        if non_answer_rate >= 0.50:
            return "retrieve_more", "subset_non_answer_majority"
        if non_answer_rate >= 0.20 and vector_risk >= 0.76:
            return "retrieve_more", "partial_subset_non_answer_vector_risk"
        if vector_risk >= 0.73 and support_mean <= 0.16 and neutral_mean >= 0.86:
            return "retrieve_more", "low_support_high_neutral_vector"
        if vector_risk >= 0.84 and entailment_mean <= 0.10 and not_included_mass >= 0.88:
            return "retrieve_more", "low_entailment_not_included_vector"
        if contradiction_mean >= 0.16 and vector_risk >= 0.66 and support_mean <= 0.35:
            return "retrieve_more", "soft_contradiction_vector"
        if vector_instability >= 0.05 and risk_max >= 0.82:
            return "retrieve_more", "unstable_probability_vector"
        return baseline, "vector_stable_or_insufficient_signal"

    if policy in {"adapter_evidence_instability", "evidence_instability_v1"}:
        score = _adapter_score(row, "evidence_instability")
        threshold = _float(row, "adapter_threshold", 0.42)

        if canonical_action(baseline) != "answer":
            if score >= threshold or non_answer_rate >= 0.50:
                return baseline, "baseline_non_answer_adapter_confirmed"
            return baseline, "baseline_non_answer_preserved"
        if score >= threshold:
            return "retrieve_more", "evidence_instability_high"
        return baseline, "evidence_instability_stable"

    if policy in {"adapter_active_retrieval", "active_retrieval_multicriteria_v1"}:
        score = _adapter_score(row, "active_retrieval_multicriteria")
        threshold = _float(row, "adapter_threshold", 0.25)

        if canonical_action(baseline) != "answer":
            if score >= threshold or non_answer_rate >= 0.50:
                return baseline, "baseline_non_answer_adapter_confirmed"
            return baseline, "baseline_non_answer_preserved"
        if score >= threshold:
            return "retrieve_more", "active_retrieval_multicriteria_high"
        return baseline, "active_retrieval_multicriteria_stable"

    raise ValueError(f"Unknown evidence sampling policy: {policy}")
