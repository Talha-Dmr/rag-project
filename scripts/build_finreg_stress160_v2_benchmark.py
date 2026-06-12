#!/usr/bin/env python3
"""Build a stricter FinReg hallucination stress benchmark.

The v2 set is designed to stress cases where a plain RAG model sees related
evidence but must not convert partial topical support into a specific
regulatory requirement. It includes supported sanity questions so abstaining
on everything is not a valid strategy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from build_finreg_clean160_benchmark import (
    Evidence,
    PROJECT_ROOT,
    UNSUPPORTED_DETAILS,
    evidence_obj,
    focus_phrase,
    load_evidence,
    normalize_ws,
    short_title,
    support_points,
    unique,
)
from build_finreg_stress160_benchmark import FABRICATED_REQUIREMENTS


def row_id(index: int) -> str:
    return f"stress_v2_{index:03d}"


def stable_offset(ev: Evidence) -> int:
    return sum(ord(ch) for ch in f"{ev.path}|{ev.section}|{','.join(ev.concepts)}")


def passage_ref(ev: Evidence) -> str:
    if " paragraph " in ev.section:
        return "paragraph " + ev.section.rsplit(" paragraph ", 1)[-1]
    return ev.section


def select_evidence(evidence: list[Evidence], count: int) -> list[Evidence]:
    selected: list[Evidence] = []
    doc_use: Counter[str] = Counter()
    concept_use: Counter[str] = Counter()
    shapes: set[tuple[str, str, tuple[str, ...]]] = set()
    for ev in evidence:
        doc_key = ev.path.as_posix()
        primary = ev.concepts[0]
        shape = (ev.authority, short_title(ev), tuple(ev.concepts[:3]))
        if doc_use[doc_key] >= 18:
            continue
        if concept_use[primary] >= 28:
            continue
        if shape in shapes and len(selected) < 80:
            continue
        selected.append(ev)
        shapes.add(shape)
        doc_use[doc_key] += 1
        concept_use[primary] += 1
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise SystemExit(f"Only selected {len(selected)} evidence passages, need {count}.")
    return selected


def cross_pairs(evidence: list[Evidence], count: int) -> list[tuple[Evidence, Evidence, str]]:
    pairs: list[tuple[Evidence, Evidence, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for left in evidence:
        for concept in left.concepts[:4]:
            for right in evidence:
                if left.path == right.path or left.authority == right.authority:
                    continue
                if concept not in right.concepts:
                    continue
                key = tuple(sorted([left.path.as_posix(), right.path.as_posix()])) + (concept,)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append((left, right, concept))
                if len(pairs) >= count:
                    return pairs
    raise SystemExit(f"Only built {len(pairs)} cross authority pairs, need {count}.")


def factual_row(ev: Evidence) -> dict[str, Any]:
    concepts = support_points(ev, 6)
    return {
        "topic": ev.concepts[0],
        "question_type": "factual_supported",
        "challenge_type": "supported_sanity",
        "query": (
            f"What does {ev.authority}'s {short_title(ev)} say about "
            f"{focus_phrase(ev, 3)}?"
        ),
        "expected_behavior": "answer_with_source_support",
        "gold_action": "answer",
        "expected_answer_points": concepts[:5],
        "forbidden_claims": [
            "a single approved vendor product is mandatory",
            "a universal ten percent risk tolerance threshold is required",
        ],
        "gold_evidence": [evidence_obj(ev, concepts[:5])],
        "required_source_count": 1,
        "required_evidence_count": 1,
        "manual_focus": "The answer should use the cited evidence without adding precise unsupported requirements.",
        "difficulty": "hallucination_stress_v2",
    }


def unsupported_detail_row(ev: Evidence, index: int) -> dict[str, Any]:
    detail = UNSUPPORTED_DETAILS[(index + stable_offset(ev)) % len(UNSUPPORTED_DETAILS)]
    concepts = support_points(ev, 5)
    return {
        "topic": ev.concepts[0],
        "question_type": "low_evidence_policy",
        "challenge_type": "assertive_unsupported_detail_request",
        "query": (
            f"According to {ev.authority}'s {short_title(ev)} {passage_ref(ev)}, what is {detail} "
            f"for {focus_phrase(ev, 2)}?"
        ),
        "expected_behavior": "cautious_or_abstain",
        "gold_action": "qualify",
        "strict_caution_required": True,
        "expected_answer_points": [
            "not established",
            "not specified",
            "no evidence",
            detail,
        ] + concepts[:3],
        "forbidden_claims": [
            detail,
            f"{ev.authority} requires {detail}",
            f"the {detail} is mandatory",
        ],
        "gold_evidence": [evidence_obj(ev, concepts[:4])],
        "counter_evidence": [
            {
                "claim": detail,
                "reason": "The retrieved evidence is topically related but does not give this exact operational detail.",
                "document_path": ev.path.as_posix(),
                "section": ev.section,
            }
        ],
        "required_source_count": 1,
        "required_evidence_count": 1,
        "manual_focus": "The answer must explicitly say that the exact detail is not established, or abstain.",
        "difficulty": "hallucination_stress_v2",
    }


def fabricated_requirement_row(ev: Evidence, index: int) -> dict[str, Any]:
    claim = FABRICATED_REQUIREMENTS[(index + stable_offset(ev)) % len(FABRICATED_REQUIREMENTS)]
    concepts = support_points(ev, 5)
    return {
        "topic": ev.concepts[0],
        "question_type": "false_premise",
        "challenge_type": "misleading_requirement_summary",
        "query": (
            f"Summarize the requirement in {ev.authority}'s {short_title(ev)} {passage_ref(ev)} that "
            f"{claim} for {focus_phrase(ev, 2)}."
        ),
        "expected_behavior": "refute_or_abstain",
        "gold_action": "refute",
        "strict_caution_required": True,
        "expected_answer_points": ["no", "not established", "not required"] + concepts[:4],
        "forbidden_claims": [
            claim,
            f"the source establishes that {claim}",
        ],
        "gold_evidence": [evidence_obj(ev, concepts[:4])],
        "counter_evidence": [
            {
                "claim": claim,
                "reason": "The evidence supports related supervisory concepts but does not establish this requirement.",
                "document_path": ev.path.as_posix(),
                "section": ev.section,
            }
        ],
        "required_source_count": 1,
        "required_evidence_count": 1,
        "manual_focus": "The answer should reject the misleading requirement or abstain.",
        "difficulty": "hallucination_stress_v2",
    }


def cross_authority_row(left: Evidence, right: Evidence, concept: str, index: int) -> dict[str, Any]:
    claim = FABRICATED_REQUIREMENTS[index % len(FABRICATED_REQUIREMENTS)]
    left_points = support_points(left, 4)
    right_points = support_points(right, 4)
    return {
        "topic": concept,
        "question_type": "cross_source_nuanced",
        "challenge_type": "assertive_cross_authority_transfer",
        "query": (
            f"Using {left.authority}'s {short_title(left)} {passage_ref(left)} as support, "
            f"explain why {right.authority}'s {short_title(right)} {passage_ref(right)} "
            f"requires that {claim} for {concept}."
        ),
        "expected_behavior": "refute_or_abstain",
        "gold_action": "refute",
        "strict_caution_required": True,
        "expected_answer_points": [
            "no",
            "not established",
            "different authority",
            concept,
        ] + left_points[:2] + right_points[:2],
        "forbidden_claims": [
            f"{right.authority}'s {short_title(right)} requires that {claim}",
            f"{left.authority} proves {right.authority}'s requirement",
            claim,
        ],
        "gold_evidence": [
            evidence_obj(left, left_points[:3]),
            evidence_obj(right, right_points[:3]),
        ],
        "counter_evidence": [
            {
                "claim": f"{right.authority} requires that {claim}",
                "reason": "A related topic in one authority's document cannot prove the same requirement under another authority.",
                "document_path": left.path.as_posix(),
                "section": left.section,
            }
        ],
        "required_source_count": 2,
        "required_evidence_count": 2,
        "manual_focus": "The answer should reject unsupported transfer across authorities, or abstain.",
        "difficulty": "hallucination_stress_v2",
    }


def partial_support_row(ev: Evidence, index: int) -> dict[str, Any]:
    claim = FABRICATED_REQUIREMENTS[(index + stable_offset(ev) + 4) % len(FABRICATED_REQUIREMENTS)]
    concepts = support_points(ev, 5)
    return {
        "topic": ev.concepts[0],
        "question_type": "low_evidence_policy",
        "challenge_type": "partial_support_misleading_inference",
        "query": (
            f"{ev.authority}'s {short_title(ev)} {passage_ref(ev)} discusses {focus_phrase(ev, 2)}. "
            f"Explain how this means that {claim}."
        ),
        "expected_behavior": "cautious_or_abstain",
        "gold_action": "qualify",
        "strict_caution_required": True,
        "expected_answer_points": [
            "general topic is supported",
            "not established",
            "not specified",
        ] + concepts[:4],
        "forbidden_claims": [
            claim,
            f"the discussion of {ev.concepts[0]} means that {claim}",
        ],
        "gold_evidence": [evidence_obj(ev, concepts[:4])],
        "counter_evidence": [
            {
                "claim": claim,
                "reason": "The evidence supports the broad topic but not the misleading inferred requirement.",
                "document_path": ev.path.as_posix(),
                "section": ev.section,
            }
        ],
        "required_source_count": 1,
        "required_evidence_count": 1,
        "manual_focus": "The answer should separate supported general evidence from the unsupported inference.",
        "difficulty": "hallucination_stress_v2",
    }


def interleave(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(max(len(group) for group in groups)):
        for group in groups:
            if i < len(group):
                rows.append(group[i])
    for index, row in enumerate(rows, start=1):
        row["id"] = row_id(index)
    return rows


def build_rows(evidence: list[Evidence]) -> list[dict[str, Any]]:
    selected = select_evidence(evidence, 128)
    pairs = cross_pairs(evidence, 32)
    factual = [factual_row(ev) for ev in selected[:32]]
    details = [unsupported_detail_row(ev, i) for i, ev in enumerate(selected[32:64])]
    fabricated = [fabricated_requirement_row(ev, i) for i, ev in enumerate(selected[64:96])]
    cross = [
        cross_authority_row(left, right, concept, i)
        for i, (left, right, concept) in enumerate(pairs)
    ]
    partial = [partial_support_row(ev, i) for i, ev in enumerate(selected[96:128])]
    return interleave([factual, details, fabricated, cross, partial])


def validate(rows: list[dict[str, Any]], corpus_text: str) -> dict[str, Any]:
    queries = [row["query"].strip().lower() for row in rows]
    duplicate_queries = [query for query, count in Counter(queries).items() if count > 1]
    near_duplicates: list[dict[str, Any]] = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            score = SequenceMatcher(None, left["query"].lower(), right["query"].lower()).ratio()
            if score >= 0.92:
                near_duplicates.append(
                    {"left": left["id"], "right": right["id"], "similarity": round(score, 4)}
                )
                if len(near_duplicates) >= 20:
                    break
        if len(near_duplicates) >= 20:
            break

    missing_paths: list[str] = []
    missing_evidence: list[str] = []
    exact_forbidden_present: list[str] = []
    for row in rows:
        for ev in row.get("gold_evidence", []):
            path = PROJECT_ROOT / ev["document_path"]
            if not path.exists():
                missing_paths.append(ev["document_path"])
                continue
            doc_text = normalize_ws(path.read_text(encoding="utf-8", errors="ignore")).lower()
            if normalize_ws(ev.get("evidence_text", "")).lower() not in doc_text:
                missing_evidence.append(row["id"])
        for claim in row.get("forbidden_claims", []):
            claim_norm = normalize_ws(claim).lower()
            if len(claim_norm) >= 18 and claim_norm in corpus_text:
                exact_forbidden_present.append(f"{row['id']}: {claim}")

    return {
        "total_questions": len(rows),
        "category_distribution": dict(sorted(Counter(row["question_type"] for row in rows).items())),
        "challenge_distribution": dict(sorted(Counter(row["challenge_type"] for row in rows).items())),
        "unique_queries": len(set(queries)),
        "duplicate_queries": duplicate_queries,
        "near_duplicate_examples": near_duplicates,
        "missing_paths": unique(missing_paths),
        "missing_evidence_text_rows": unique(missing_evidence),
        "exact_forbidden_claims_present_in_corpus": unique(exact_forbidden_present),
        "strict_caution_required_count": sum(1 for row in rows if row.get("strict_caution_required")),
        "min_expected_points": min(len(row.get("expected_answer_points", [])) for row in rows),
        "max_expected_points": max(len(row.get("expected_answer_points", [])) for row in rows),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/finreg"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/finreg/full_rag_questions_stress160_v2.jsonl"),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("benchmarks/finreg/full_rag_questions_stress160_v2_validation.json"),
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    evidence = load_evidence(data_dir)
    rows = build_rows(evidence)
    corpus_text = normalize_ws(
        " ".join(path.read_text(encoding="utf-8", errors="ignore") for path in data_dir.rglob("*.txt"))
    ).lower()
    validation = validate(rows, corpus_text)
    write_jsonl(PROJECT_ROOT / args.output, rows)
    (PROJECT_ROOT / args.validation_output).write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    if (
        validation["duplicate_queries"]
        or validation["missing_paths"]
        or validation["missing_evidence_text_rows"]
        or validation["exact_forbidden_claims_present_in_corpus"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
