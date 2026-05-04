"""Lightweight lexical verifier for answer-context support checks."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "without",
    "unless",
    "neither",
    "nor",
    "cannot",
    "can't",
    "mustn't",
    "prohibit",
    "prohibited",
    "prohibits",
    "ban",
    "banned",
    "except",
    "exempt",
    "exemption",
}
HALLUCINATION_MARKERS = {
    "email",
    "address",
    "xml",
    "template",
    "vendor",
    "private",
    "provider",
    "watermark",
    "printed",
    "print",
    "14",
    "point",
    "font",
    "notices",
    "approved",
    "named",
}
CONTRADICTION_MARKERS = {
    "must",
    "mustn't",
    "shall",
    "should",
    "cannot",
    "required",
    "requires",
    "prohibited",
    "unless",
    "except",
    "only",
    "all",
    "never",
    "no",
    "not",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}
SUPPORT_TO_NLI = {
    "supported": "entailment",
    "unsupported": "neutral",
    "contradicted": "contradiction",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9%.,' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str, *, remove_stopwords: bool = True) -> list[str]:
    found = [match.group(0).lower() for match in WORD_RE.finditer(text)]
    if remove_stopwords:
        return [token for token in found if token not in STOPWORDS]
    return found


def token_containment(candidate_tokens: list[str], evidence_tokens: list[str]) -> float:
    if not candidate_tokens:
        return 0.0
    evidence_counts = Counter(evidence_tokens)
    overlap = 0
    for token, count in Counter(candidate_tokens).items():
        overlap += min(count, evidence_counts.get(token, 0))
    return overlap / len(candidate_tokens)


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def numbers(text: str) -> set[str]:
    return {match.group(0).replace(",", ".") for match in NUMBER_RE.finditer(text)}


def negation_count(text: str) -> int:
    text_tokens = tokens(text, remove_stopwords=False)
    return sum(1 for token in text_tokens if token in NEGATION_WORDS)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def softmax(scores: dict[str, float]) -> dict[str, float]:
    max_score = max(scores.values())
    exps = {key: math.exp(value - max_score) for key, value in scores.items()}
    total = sum(exps.values())
    return {key: value / total for key, value in exps.items()}


def predict_support_status(query: str, candidate_answer: str, evidence_span: str) -> dict[str, Any]:
    candidate_norm = normalize_text(candidate_answer)
    evidence_norm = normalize_text(evidence_span)
    query_norm = normalize_text(query)

    candidate_tokens = tokens(candidate_norm)
    evidence_tokens = tokens(evidence_norm)
    query_tokens = tokens(query_norm)
    candidate_token_set = set(candidate_tokens)
    evidence_token_set = set(evidence_tokens)

    containment = token_containment(candidate_tokens, evidence_tokens)
    overlap = jaccard(candidate_tokens, evidence_tokens)
    query_evidence_overlap = jaccard(query_tokens, evidence_tokens)
    exact_substring = bool(candidate_norm and candidate_norm in evidence_norm)

    candidate_numbers = numbers(candidate_norm)
    evidence_numbers = numbers(evidence_norm)
    number_mismatch = bool(candidate_numbers and evidence_numbers and not candidate_numbers <= evidence_numbers)

    candidate_negations = negation_count(candidate_norm)
    evidence_negations = negation_count(evidence_norm)
    negation_mismatch = abs(candidate_negations - evidence_negations) > 0
    unsupported_marker_count = len((candidate_token_set - evidence_token_set) & HALLUCINATION_MARKERS)
    contradiction_marker_count = len(candidate_token_set & CONTRADICTION_MARKERS)
    unsupported_marker_hit = unsupported_marker_count > 0 and containment < 0.55
    contradiction_marker_hit = contradiction_marker_count > 0 and containment >= 0.30

    supported_score = 0.15 + 0.65 * containment + 0.20 * overlap
    unsupported_score = 0.55 - 0.35 * containment + 0.10 * (1.0 - query_evidence_overlap)
    contradicted_score = 0.12 + 0.15 * overlap

    if exact_substring:
        supported_score += 0.65
        unsupported_score -= 0.25
        contradicted_score -= 0.10

    if number_mismatch and containment >= 0.35:
        contradicted_score += 0.45
        supported_score -= 0.25
        unsupported_score -= 0.05

    if negation_mismatch and containment >= 0.35:
        contradicted_score += 0.35
        supported_score -= 0.20
        unsupported_score -= 0.05

    if unsupported_marker_hit:
        unsupported_score += 0.35
        supported_score -= 0.25
        contradicted_score -= 0.05

    if contradiction_marker_hit and not exact_substring:
        contradicted_score += 0.18

    if containment >= 0.55 and not exact_substring:
        supported_score -= 0.22
        contradicted_score += 0.18

    if containment < 0.45:
        unsupported_score += 0.25
        supported_score -= 0.20

    raw_scores = {
        "supported": clamp(supported_score),
        "unsupported": clamp(unsupported_score),
        "contradicted": clamp(contradicted_score),
    }
    probabilities = softmax(raw_scores)
    support_status = max(probabilities.items(), key=lambda item: item[1])[0]

    return {
        "support_status": support_status,
        "support_status_scores": probabilities,
        "features": {
            "candidate_token_containment": round(containment, 4),
            "candidate_evidence_jaccard": round(overlap, 4),
            "query_evidence_jaccard": round(query_evidence_overlap, 4),
            "exact_substring": exact_substring,
            "number_mismatch": number_mismatch,
            "negation_mismatch": negation_mismatch,
            "unsupported_marker_count": unsupported_marker_count,
            "contradiction_marker_count": contradiction_marker_count,
        },
    }


def to_detector_result(prediction: dict[str, Any]) -> dict[str, Any]:
    support_status = prediction["support_status"]
    support_scores = prediction["support_status_scores"]
    label = SUPPORT_TO_NLI[support_status]
    return {
        "is_hallucination": support_status == "contradicted",
        "label": label,
        "confidence": max(float(value) for value in support_scores.values()),
        "scores": {
            "entailment": float(support_scores.get("supported", 0.0)),
            "neutral": float(support_scores.get("unsupported", 0.0)),
            "contradiction": float(support_scores.get("contradicted", 0.0)),
        },
        "artifact_verifier": {
            "support_status": support_status,
            "support_status_scores": support_scores,
            "features": prediction["features"],
        },
        "prediction_source": "artifact_verifier",
    }
