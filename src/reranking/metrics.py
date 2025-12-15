"""
Metrics for evaluating rerankers on ambiguous multi-answer datasets such as AmbigQA.

All metrics operate per-query and can then be averaged across the dataset.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

RelevanceDict = Dict[str, float]
Ranking = List[str]


def _validate_k_values(k_values: Sequence[int]) -> List[int]:
    if not k_values:
        raise ValueError("k_values must contain at least one cutoff.")
    cleaned = sorted(set(int(k) for k in k_values))
    if cleaned[0] <= 0:
        raise ValueError("k_values must all be positive integers.")
    return cleaned


def _positive_doc_ids(relevance_labels: RelevanceDict, threshold: float) -> List[str]:
    return [doc_id for doc_id, rel in relevance_labels.items() if rel > threshold]


def _dcg_at_k(ranked_doc_ids: Ranking, relevance_labels: RelevanceDict, k: int) -> float:
    dcg = 0.0
    for idx, doc_id in enumerate(ranked_doc_ids[:k]):
        rel = relevance_labels.get(doc_id, 0.0)
        if rel <= 0:
            continue
        discount = math.log2(idx + 2)  # ranks are 1-indexed here
        dcg += (2 ** rel - 1) / discount
    return dcg


def ndcg_at_k(
    ranked_doc_ids: Ranking,
    relevance_labels: RelevanceDict,
    k: int,
) -> float:
    """Compute nDCG@k for a single query."""
    ideal_relevances = sorted((rel for rel in relevance_labels.values() if rel > 0), reverse=True)
    ideal_dcg = 0.0
    for idx, rel in enumerate(ideal_relevances[:k]):
        discount = math.log2(idx + 2)
        ideal_dcg += (2 ** rel - 1) / discount
    if ideal_dcg == 0.0:
        return 0.0
    actual_dcg = _dcg_at_k(ranked_doc_ids, relevance_labels, k)
    return actual_dcg / ideal_dcg
    """Notes:
    - Supports graded relevance.
    - Ideal DCG is computed over all known relevant passages,
      including those not retrieved by the reranker.
    - Operates at passage-level relevance (not interpretation-level).
    """

def recall_at_k(
    ranked_doc_ids: Ranking,
    relevance_labels: RelevanceDict,
    k: int,
    positive_relevance_threshold: float = 0.0,
) -> float:
    """Recall@k using the provided threshold to decide relevance."""
    relevant_doc_ids = _positive_doc_ids(relevance_labels, positive_relevance_threshold)
    if not relevant_doc_ids:
        return 0.0
    retrieved = sum(
        1
        for doc_id in ranked_doc_ids[:k]
        if relevance_labels.get(doc_id, 0.0) > positive_relevance_threshold
    )
    return retrieved / len(relevant_doc_ids)
    """
    Recall@k at passage level.

    Measures how many relevant passages are retrieved in top-k.
    Does not explicitly model interpretation-level coverage.
    """

def precision_at_k(
    ranked_doc_ids: Ranking,
    relevance_labels: RelevanceDict,
    k: int,
    positive_relevance_threshold: float = 0.0,
) -> float:
    """Precision@k (order inside top-k does not matter)."""
    if k <= 0:
        raise ValueError("k must be positive for precision@k")
    effective_k = min(k, len(ranked_doc_ids))
    if effective_k == 0:
        return 0.0
    relevant_in_top_k = sum(
        1
        for doc_id in ranked_doc_ids[:effective_k]
        if relevance_labels.get(doc_id, 0.0) > positive_relevance_threshold
    )
    return relevant_in_top_k / effective_k


def hit_rate_at_k(
    ranked_doc_ids: Ranking,
    relevance_labels: RelevanceDict,
    k: int,
    positive_relevance_threshold: float = 0.0,
) -> float:
    """Binary indicator: did any relevant passage appear in top-k?"""
    for doc_id in ranked_doc_ids[:k]:
        if relevance_labels.get(doc_id, 0.0) > positive_relevance_threshold:
            return 1.0
    return 0.0


def reciprocal_rank(
    ranked_doc_ids: Ranking,
    relevance_labels: RelevanceDict,
    positive_relevance_threshold: float = 0.0,
) -> float:
    """Reciprocal rank of the first relevant document (MRR component)."""
    for idx, doc_id in enumerate(ranked_doc_ids):
        if relevance_labels.get(doc_id, 0.0) > positive_relevance_threshold:
            return 1.0 / (idx + 1)
    return 0.0


def compute_query_metrics(
    ranked_doc_ids: Ranking,
    relevance_labels: RelevanceDict,
    k_values: Sequence[int] = (5, 10),
    positive_relevance_threshold: float = 0.0,
) -> Dict[str, float]:
    """Compute all reranker metrics for a single query."""
    cleaned_k = _validate_k_values(k_values)
    metrics: Dict[str, float] = {}
    for k in cleaned_k:
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_doc_ids, relevance_labels, k)
        metrics[f"recall@{k}"] = recall_at_k(
            ranked_doc_ids,
            relevance_labels,
            k,
            positive_relevance_threshold,
        )
        metrics[f"precision@{k}"] = precision_at_k(
            ranked_doc_ids,
            relevance_labels,
            k,
            positive_relevance_threshold,
        )
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(
            ranked_doc_ids,
            relevance_labels,
            k,
            positive_relevance_threshold,
        )
    metrics["mrr"] = reciprocal_rank(ranked_doc_ids, relevance_labels, positive_relevance_threshold)
    return metrics


def aggregate_metrics(per_query_metrics: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Average per-query metrics to obtain corpus-level scores."""
    if not per_query_metrics:
        raise ValueError("per_query_metrics must contain at least one entry")
    totals: MutableMapping[str, float] = {}
    for metrics in per_query_metrics:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    count = len(per_query_metrics)
    return {key: total / count for key, total in totals.items()}


def evaluate_rankings(
    rankings: Sequence[Ranking],
    relevance_labels_list: Sequence[RelevanceDict],
    k_values: Sequence[int] = (5, 10),
    positive_relevance_threshold: float = 0.0,
) -> Dict[str, Dict[str, float] | List[Dict[str, float]]]:
    """
    Convenience wrapper to compute per-query and averaged metrics.
    Returns a dict with keys 'per_query' and 'average'.
    """
    if len(rankings) != len(relevance_labels_list):
        raise ValueError("rankings and relevance_labels_list must have the same length")
    per_query = [
        compute_query_metrics(ranked, labels, k_values, positive_relevance_threshold)
        for ranked, labels in zip(rankings, relevance_labels_list)
    ]
    averaged = aggregate_metrics(per_query)
    return {"per_query": per_query, "average": averaged}
