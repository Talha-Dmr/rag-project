"""Evidence-subset sampling policies for RAG gating.

These policies are intentionally small and side-effect free so shadow eval
scripts and the runtime pipeline can use the same decision rules.
"""

from __future__ import annotations

from typing import Any


def canonical_action(action: str) -> str:
    return "answer" if action in ("none", "answer") else action


def decide_policy(row: dict[str, Any], policy: str) -> tuple[str, str]:
    qtype = str(row.get("type") or row.get("question_type") or "")
    baseline = str(row.get("baseline_action") or "none")

    if row.get("candidate_unavailable"):
        return baseline, "candidate_unavailable"

    answer_rate = float(row.get("subset_answer_rate") or 0.0)
    non_answer_rate = float(row.get("subset_non_answer_rate") or 0.0)
    instability = float(row.get("subset_action_instability") or 0.0)
    risk_mean = float(row.get("answer_include_risk_mean_across_subsets") or 0.0)
    risk_max = float(row.get("answer_include_risk_max_across_subsets") or 0.0)

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

    raise ValueError(f"Unknown evidence sampling policy: {policy}")
