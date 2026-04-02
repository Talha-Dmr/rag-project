from src.evaluation.epistemic_research import (
    aggregate_question_type_counts,
    classify_answer_outcome,
    evaluate_shadow_candidate,
    extract_shadow_run_metrics,
)


def test_classify_answer_outcome_buckets_retrieve_more_resolution():
    assert classify_answer_outcome("none", False, 0.05) == "answer_safe"
    assert classify_answer_outcome("none", False, 0.55) == "answer_risky"
    assert classify_answer_outcome("retrieve_more", False, 0.10) == "retrieve_more_resolved"
    assert classify_answer_outcome("retrieve_more", False, 0.60) == "retrieve_more_still_risky"
    assert classify_answer_outcome("retrieve_more", True, 0.05) == "abstain"


def test_aggregate_question_type_counts_groups_outcomes():
    rows = [
        {"question_type": "sanity", "outcome_bucket": "answer_safe"},
        {"question_type": "sanity", "outcome_bucket": "abstain"},
        {"question_type": "conflict", "outcome_bucket": "retrieve_more_resolved"},
    ]

    counts = aggregate_question_type_counts(rows)

    assert counts["sanity"]["answer_safe"] == 1
    assert counts["sanity"]["abstain"] == 1
    assert counts["conflict"]["retrieve_more_resolved"] == 1


def test_extract_shadow_run_metrics_reads_shadow_summary():
    payload = {
        "runtime_mean_seconds": 1.2,
        "runtime_total_seconds": 24.0,
        "u_epi_stochastic_mean": 0.03,
        "u_ale_mean": 0.15,
        "stats_all": {"contradiction_rate": 0.02},
        "shadow_two_channel": {
            "answer_rate": 0.6,
            "retrieve_more_rate": 0.25,
            "abstain_rate": 0.15,
            "stats_answer": {"contradiction_rate": 0.01},
            "stats_non_abstain": {"contradiction_rate": 0.02},
        },
    }

    metrics = extract_shadow_run_metrics(payload)

    assert metrics["shadow_answer_rate"] == 0.6
    assert metrics["shadow_retrieve_more_rate"] == 0.25
    assert metrics["shadow_abstain_rate"] == 0.15
    assert metrics["shadow_answered_contradiction_rate"] == 0.02
    assert metrics["runtime_mean_seconds"] == 1.2


def test_evaluate_shadow_candidate_requires_quality_and_cost_pass():
    baseline = {
        "shadow_answered_contradiction_rate": 0.05,
        "runtime_mean_seconds": 1.0,
    }
    candidate = {
        "shadow_answered_contradiction_rate": 0.04,
        "shadow_abstain_rate": 0.10,
        "stats_all_contradiction_rate": 0.02,
        "runtime_mean_seconds": 1.5,
    }

    result = evaluate_shadow_candidate(candidate, baseline)

    assert result["passed_stage_1"] is True
    assert result["passed_stage_2"] is True
    assert result["candidate_pass_stage"] == "stage2_cost_pass"


def test_evaluate_shadow_candidate_fails_when_abstain_band_or_risk_breaks():
    baseline = {
        "shadow_answered_contradiction_rate": 0.05,
        "runtime_mean_seconds": 1.0,
    }
    candidate = {
        "shadow_answered_contradiction_rate": 0.08,
        "shadow_abstain_rate": 0.40,
        "stats_all_contradiction_rate": 0.20,
        "runtime_mean_seconds": 1.0,
    }

    result = evaluate_shadow_candidate(candidate, baseline)

    assert result["passed_stage_1"] is False
    assert "answered_contradiction_worse_than_baseline" in result["reasons"]
    assert "abstain_rate_out_of_band" in result["reasons"]
    assert "contradiction_guard_failed" in result["reasons"]
