"""Evidence-subset sampling policies for RAG gating.

These policies are intentionally small and side-effect free so shadow eval
scripts and the runtime pipeline can use the same decision rules.
"""

from __future__ import annotations

import re
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


def _is_specific_detail_probe(text: str) -> bool:
    query = f" {(text or '').strip().lower()} "
    return any(
        marker in query
        for marker in (
            "audit sampling",
            "automatic penalty",
            "board paper",
            "calendar-day",
            "color code",
            "column layout",
            "committee meeting frequency",
            "control owners",
            "email address",
            "does not specify",
            "exact ",
            "fixed ",
            "form identifier",
            "mandatory ",
            "meeting frequency",
            "minimum number",
            "named ",
            "notice wording",
            "numeric ",
            "percentage",
            "precise ",
            "universal ",
            "specific ",
            "requirement ",
            "requires ",
            "required ",
            "required audit",
            "required numeric",
            "required spreadsheet",
            "single escalation",
            "supervisor's email",
            "needed for every",
            "for every ",
            "means that ",
            "portal",
            "deadline",
            "threshold",
            "template",
            "certification",
            "certificate",
            "approval",
            "approval workflow",
            "software product",
            "public disclosure",
            "is mandatory",
            "are mandatory",
            "mandatory for",
            "include the requirement",
            "source rationale",
            "implied by",
            "checklist item",
            "converted into",
            "treated as evidence",
        )
    )


def _is_cross_authority_transfer_probe(text: str) -> bool:
    query = f" {(text or '').strip().lower()} "
    authority_mentions = sum(
        1
        for marker in (
            " eba",
            " ecb",
            " pra",
            " bcbs",
            " basel",
            " federal reserve",
            " occ",
        )
        if marker in query
    )
    transfer_markers = (
        " as supporting context",
        " cross-authority",
        " cross authority",
        " evidence establishes",
        " evidence into",
        " evidence supports that transfer",
        " inherit this claim",
        " inherits this claim",
        " source-transfer",
        " source transfer",
        " transfer",
        " transfers",
        " proves that",
        " can be used to conclude",
        " used to conclude",
        " used to create",
        " justify this sentence under",
        " obligation that",
        " requirement saying that",
    )
    strong_transfer_markers = (
        " inherit this claim",
        " inherits this claim",
        " evidence establishes",
        " evidence into",
        " source-transfer",
        " source transfer",
        " transfer",
        " transfers",
        " proves that",
        " can be used to conclude",
        " used to conclude",
        " used to create",
    )
    requirement_markers = (
        " obligation ",
        " requires ",
        " requirement ",
        " mandatory ",
        " must ",
        " allowed only",
        " can ignore",
        " may apply",
        " is supported by the two passages",
    )
    return (
        authority_mentions >= 2
        and any(marker in query for marker in transfer_markers)
        and (
            any(marker in query for marker in requirement_markers)
            or any(marker in query for marker in strong_transfer_markers)
        )
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
        " does not support ",
        " do not support ",
        " cannot be converted into ",
        " cannot be directly converted ",
        " cannot be treated as ",
        " cannot be verified ",
        " does not discuss ",
        " do not discuss ",
        " not verifiable ",
        " i don't see ",
        " i do not see ",
        " not supported ",
    )
    return any(marker in answer for marker in markers)


def _has_specific_detail_limitation(query: str, answer: str) -> bool:
    answer_lower = f" {(answer or '').strip().lower()} "
    if not _has_direct_refutation(answer_lower):
        return False

    strict_limitation_markers = (
        " not established",
        " does not establish",
        " not specified",
        " not specify",
        " does not specify",
        " do not specify",
        " doesn't specify",
        " not stated",
        " not mentioned",
        " not mandated",
        " not required",
        " no explicit",
        " not explicit",
        " not explicitly",
        " no evidence",
        " not supported",
        " does not support",
        " do not support",
        " cannot be converted into",
        " cannot be directly converted",
        " cannot be treated as",
        " cannot be verified",
        " does not discuss",
        " do not discuss",
        " cannot determine",
        " cannot conclude",
        " not verifiable",
        " i don't see",
        " i do not see",
        " don't know based on the provided context",
        " do not know based on the provided context",
    )
    if any(marker in answer_lower for marker in strict_limitation_markers):
        return True

    query_lower = f" {(query or '').strip().lower()} "
    detail_terms = (
        "portal",
        "deadline",
        "threshold",
        "template",
        "certification",
        "certificate",
        "approval",
        "audit sampling",
        "automatic penalty",
        "board paper",
        "calendar-day",
        "color code",
        "column layout",
        "committee",
        "control owner",
        "email address",
        "form identifier",
        "meeting frequency",
        "minimum number",
        "named",
        "notice wording",
        "numeric",
        "percentage",
        "precise",
        "software product",
        "vendor",
        "public disclosure",
        "cloud",
        "global",
    )
    query_terms = [term for term in detail_terms if term in query_lower]
    return bool(query_terms) and any(term in answer_lower for term in query_terms)


def _answer_affirms_risk_probe(query: str, answer: str) -> bool:
    if not (_is_specific_detail_probe(query) or _is_cross_authority_transfer_probe(query)):
        return False

    answer_lower = f" {(answer or '').strip().lower()} "
    if not answer_lower.strip():
        return False
    early = re.split(r"\b(?:however|but|although|nevertheless)\b", answer_lower, maxsplit=1)[0]
    if any(
        marker in early
        for marker in (
            " does not establish",
            " does not explicitly establish",
            " does not support",
            " no evidence",
            " not established",
            " not supported",
            " cannot conclude",
            " cannot be verified",
        )
    ):
        return False

    affirmation_markers = (
        " is supported",
        " this is supported",
        " supports the transfer",
        " establish a requirement",
        " establishes a requirement",
        " establishes that",
        " mandate",
        " mandates",
        " must use",
        " must be selected",
        " can apply",
        " may apply",
        " without requiring local",
        " without local governance",
        " requirement mandates",
        " requirements stating",
        " derived from",
        " transferred into",
        " transfers",
        " can be converted",
        " is converted",
    )
    return any(marker in early for marker in affirmation_markers)


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

        open_synthesis_query = not _is_closed_probe(query) and not _is_low_evidence_probe(query)
        if (
            open_synthesis_query
            and contradiction_mean <= 0.08
            and vector_instability <= 0.03
        ):
            return baseline, "open_synthesis_neutral_without_contradiction"

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

    if policy == "vector_v4":
        # Evidence-consensus policy:
        # open supported questions can override an overly conservative detector
        # when sampled evidence is stable, while specific-detail probes require
        # an explicit limitation or abstention.
        closed_probe = _is_closed_probe(query)
        cross_authority_transfer_probe = _is_cross_authority_transfer_probe(query)
        low_evidence_probe = (
            _is_low_evidence_probe(query)
            or _is_specific_detail_probe(query)
            or cross_authority_transfer_probe
        )
        closed_without_refutation = closed_probe and not _has_direct_refutation(answer)
        not_included_mass = _clamp01(neutral_mean + contradiction_mean)
        vector_risk = _vector_answer_risk(row)
        answer_completeness = _float(row, "answer_completeness_score_mean_across_subsets", 1.0)
        context_coverage = _float(row, "context_coverage_mean_across_subsets", 1.0)
        open_supported_probe = not closed_probe and not low_evidence_probe
        stable_supported_evidence = (
            answer_rate >= 0.60
            and contradiction_mean <= 0.08
            and vector_instability <= 0.06
            and answer_completeness >= 0.40
            and context_coverage >= 0.40
        )
        strong_supported_evidence = (
            answer_rate >= 0.75
            and support_mean >= 0.18
            and contradiction_mean <= 0.10
            and risk_max <= 0.92
        )
        risk_probe_affirmed = (
            (_is_specific_detail_probe(query) or cross_authority_transfer_probe)
            and _answer_affirms_risk_probe(query, answer)
        )

        if canonical_action(baseline) != "answer":
            if risk_probe_affirmed:
                return baseline, "risk_probe_affirmation_preserves_non_answer"
            if open_supported_probe and (stable_supported_evidence or strong_supported_evidence):
                return "none", "stable_evidence_overrides_conservative_detector"
            if (
                low_evidence_probe
                and _has_specific_detail_limitation(query, answer)
                and contradiction_mean <= 0.12
            ):
                return "none", "explicit_limitation_overrides_conservative_detector"
            if (
                not low_evidence_probe
                and _has_direct_refutation(answer)
                and contradiction_mean <= 0.08
                and not_included_mass <= 0.92
            ):
                return "none", "safe_refutation_overrides_conservative_detector"
            if non_answer_rate >= 0.50 or vector_risk >= 0.74:
                return baseline, "baseline_non_answer_evidence_confirmed"
            return baseline, "baseline_non_answer_preserved"

        if low_evidence_probe:
            if cross_authority_transfer_probe and not _has_specific_detail_limitation(query, answer):
                return "retrieve_more", "cross_authority_transfer_without_limitation"
            if _has_specific_detail_limitation(query, answer):
                return baseline, "specific_detail_limited_answer"
            if vector_risk >= 0.58 or not_included_mass >= 0.70 or support_mean <= 0.25:
                return "retrieve_more", "specific_detail_without_limitation"

        if risk_probe_affirmed:
            return "retrieve_more", "risk_probe_affirmed_claim"

        if closed_without_refutation and (
            contradiction_mean >= 0.05
            or not_included_mass >= 0.68
            or vector_risk >= 0.58
        ):
            return "abstain", "closed_probe_without_refutation_vector_v4"

        if non_answer_rate >= 0.60:
            return "retrieve_more", "subset_non_answer_supermajority"
        if non_answer_rate >= 0.30 and vector_risk >= 0.70:
            return "retrieve_more", "partial_subset_non_answer_vector_v4"
        if vector_risk >= 0.82 and entailment_mean <= 0.14 and not_included_mass >= 0.84:
            return "retrieve_more", "low_entailment_not_included_vector_v4"
        if contradiction_mean >= 0.14 and support_mean <= 0.35:
            return "retrieve_more", "soft_contradiction_vector_v4"
        if open_supported_probe and contradiction_mean <= 0.08 and vector_instability <= 0.06:
            return baseline, "open_supported_stable_vector_v4"
        return baseline, "vector_v4_stable_or_insufficient_signal"

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
