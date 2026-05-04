#!/usr/bin/env python3
"""Produce FinRegBench detector predictions with a lightweight local baseline.

This is not intended to be the final detector.  It gives us a deterministic
baseline and a compatible prediction file while the real detector pipeline is
being wired into the FinRegBench adapter.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


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


def build_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        prediction = predict_support_status(
            query=str(row.get("query") or ""),
            candidate_answer=str(row.get("candidate_answer") or ""),
            evidence_span=str(row.get("evidence_span") or ""),
        )
        predictions.append(
            {
                "id": row.get("id"),
                "prediction_source": "lexical_finregbench_baseline_v1",
                **prediction,
            }
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_inputs.jsonl"),
        help="Input JSONL exported by eval_finregbench_detector_adapter.py --export-inputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("FinRegBench/data/finreg_3000_detector_predictions_baseline.jsonl"),
        help="Prediction JSONL output.",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.inputs)
    predictions = build_predictions(rows)
    write_jsonl(args.output, predictions)


if __name__ == "__main__":
    main()
