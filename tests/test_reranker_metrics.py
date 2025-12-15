import math
import pytest

from src.reranking.metrics import (
    aggregate_metrics,
    compute_query_metrics,
    evaluate_rankings,
)

def test_empty_relevance_labels_returns_zero_metrics():
    ranked = ["doc_a", "doc_b"]
    labels = {}

    metrics = compute_query_metrics(ranked, labels, k_values=(5,))
    assert metrics["ndcg@5"] == 0.0
    assert metrics["recall@5"] == 0.0
    assert metrics["hit_rate@5"] == 0.0
    assert metrics["mrr"] == 0.0


def test_compute_query_metrics_handles_graded_relevance():
    ranked_doc_ids = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
    relevance_labels = {
        "doc_a": 2.0,
        "doc_c": 1.0,
        "doc_f": 0.5,  # relevant item that the reranker missed
    }

    metrics = compute_query_metrics(ranked_doc_ids, relevance_labels, k_values=(5, 10))

    # Recall considers all three relevant interpretations
    assert metrics["recall@5"] == pytest.approx(2 / 3)
    assert metrics["recall@10"] == pytest.approx(2 / 3)

    # Precision only looks at retrieved items
    assert metrics["precision@5"] == pytest.approx(2 / 5)
    assert metrics["precision@10"] == pytest.approx(2 / 5)

    # Hit rate signals whether we retrieved anything relevant
    assert metrics["hit_rate@5"] == 1.0
    assert metrics["hit_rate@10"] == 1.0

    # First relevant document is already at rank 1
    assert metrics["mrr"] == 1.0

    # nDCG uses graded contributions with logarithmic discount
    dcg = (2 ** 2 - 1) / math.log2(2) + (2 ** 1 - 1) / math.log2(4)
    ideal = (
        (2 ** 2 - 1) / math.log2(2)
        + (2 ** 1 - 1) / math.log2(3)
        + (2 ** 0.5 - 1) / math.log2(4)
    )
    assert metrics["ndcg@5"] == pytest.approx(dcg / ideal)
    assert metrics["ndcg@10"] == pytest.approx(dcg / ideal)


def test_evaluate_rankings_aggregates_per_query_scores():
    ranked = [
        ["doc_a", "doc_b", "doc_c"],
        ["doc_c", "doc_a", "doc_b"],
    ]
    labels = [
        {"doc_b": 2.0, "doc_c": 1.0},
        {"doc_a": 1.0},
    ]

    summary = evaluate_rankings(ranked, labels, k_values=(3,))

    assert len(summary["per_query"]) == 2
    averaged = summary["average"]

    manual_avg = aggregate_metrics(list(summary["per_query"]) if isinstance(summary["per_query"], list) else [summary["per_query"]])
    assert averaged.keys() == manual_avg.keys()
    for key, value in averaged.items():
        assert value == pytest.approx(manual_avg[key])

    # Quality checks: at least one relevant doc is retrieved everywhere
    assert averaged["hit_rate@3"] == 1.0
    assert averaged["recall@3"] == 1.0
