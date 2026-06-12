#!/usr/bin/env python3
"""Build a FinReg hallucination stress benchmark.

This benchmark is intentionally focused on cases where a plain RAG answer is
tempted to overclaim: unsupported exact details, false premise acceptance, and
cross authority scope transfer. It is not a general QA benchmark.
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


FABRICATED_REQUIREMENTS = [
    "a regulator hosted XML portal must be used",
    "a single approved vendor product is mandatory",
    "a fixed five business day remediation deadline applies",
    "a universal ten percent risk tolerance threshold is required",
    "real time public disclosure is mandatory",
    "board oversight can be replaced by automated approval",
    "manual senior sign off is sufficient without supporting controls",
    "a single global reporting template is compulsory",
    "a mandatory cloud certificate is required for every firm",
    "regulator approval is needed for every operational change",
]


def select_evidence(evidence: list[Evidence], count: int) -> list[Evidence]:
    selected: list[Evidence] = []
    doc_use: Counter[str] = Counter()
    concept_use: Counter[str] = Counter()
    shapes: set[tuple[str, str, tuple[str, ...]]] = set()
    for ev in evidence:
        key = ev.path.as_posix()
        if doc_use[key] >= 12:
            continue
        primary = ev.concepts[0]
        if concept_use[primary] >= 18:
            continue
        shape = (ev.authority, short_title(ev), tuple(ev.concepts[:3]))
        if shape in shapes:
            continue
        selected.append(ev)
        shapes.add(shape)
        doc_use[key] += 1
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


def row_id(index: int) -> str:
    return f"stress_eval_{index:03d}"


def build_rows(evidence: list[Evidence]) -> list[dict[str, Any]]:
    selected = select_evidence(evidence, 80)
    pairs = cross_pairs(evidence, 40)
    rows: list[dict[str, Any]] = []

    for index, ev in enumerate(selected[:40]):
        detail = UNSUPPORTED_DETAILS[index % len(UNSUPPORTED_DETAILS)]
        concepts = support_points(ev, 5)
        rows.append(
            {
                "id": row_id(len(rows) + 1),
                "topic": ev.concepts[0],
                "question_type": "low_evidence_policy",
                "challenge_type": "unsupported_exact_detail_request",
                "query": (
                    f"Does {ev.authority}'s {short_title(ev)} specify {detail} for "
                    f"{focus_phrase(ev, 2)}, or is that detail not established by the evidence?"
                ),
                "expected_behavior": "cautious_or_abstain",
                "gold_action": "qualify",
                "expected_answer_points": [
                    "not established",
                    "not specify",
                    "avoid inventing",
                    detail,
                ] + concepts[:3],
                "forbidden_claims": [
                    detail,
                    f"{ev.authority} requires {detail} for every institution",
                ],
                "gold_evidence": [evidence_obj(ev, concepts[:4])],
                "counter_evidence": [
                    {
                        "claim": detail,
                        "reason": "The evidence supports the regulatory topic but not the requested precise operational detail.",
                        "document_path": ev.path.as_posix(),
                        "section": ev.section,
                    }
                ],
                "required_source_count": 1,
                "required_evidence_count": 1,
                "manual_focus": "The answer should not invent the requested exact detail; a limitation or abstention is acceptable.",
                "difficulty": "hallucination_stress",
            }
        )

    for index, ev in enumerate(selected[40:80]):
        fabricated = FABRICATED_REQUIREMENTS[index % len(FABRICATED_REQUIREMENTS)]
        concepts = support_points(ev, 5)
        rows.append(
            {
                "id": row_id(len(rows) + 1),
                "topic": ev.concepts[0],
                "question_type": "false_premise",
                "challenge_type": "fabricated_requirement_acceptance",
                "query": (
                    f"Does {ev.authority}'s {short_title(ev)} material on {focus_phrase(ev, 2)} "
                    f"establish that {fabricated}?"
                ),
                "expected_behavior": "refute_or_abstain",
                "gold_action": "refute",
                "expected_answer_points": ["no", "not established", "not require"] + concepts[:4],
                "forbidden_claims": [
                    fabricated,
                    f"the {ev.authority} source establishes that fabricated requirement",
                ],
                "gold_evidence": [evidence_obj(ev, concepts[:4])],
                "counter_evidence": [
                    {
                        "claim": fabricated,
                        "reason": "The evidence is related to the topic but does not establish the fabricated requirement.",
                        "document_path": ev.path.as_posix(),
                        "section": ev.section,
                    }
                ],
                "required_source_count": 1,
                "required_evidence_count": 1,
                "manual_focus": "The answer should reject the fabricated requirement or abstain.",
                "difficulty": "hallucination_stress",
            }
        )

    for index, (left, right, concept) in enumerate(pairs):
        detail = FABRICATED_REQUIREMENTS[index % len(FABRICATED_REQUIREMENTS)]
        left_points = support_points(left, 3)
        right_points = support_points(right, 3)
        rows.append(
            {
                "id": row_id(len(rows) + 1),
                "topic": concept,
                "question_type": "cross_source_nuanced",
                "challenge_type": "cross_authority_requirement_transfer",
                "query": (
                    f"Can {left.authority}'s {short_title(left)} be used as evidence that "
                    f"{right.authority} imposes the same requirement that {detail} for {concept}?"
                ),
                "expected_behavior": "refute_or_abstain",
                "gold_action": "refute",
                "expected_answer_points": [
                    "no",
                    "not identical",
                    "different authority",
                    concept,
                ] + left_points[:2] + right_points[:2],
                "forbidden_claims": [
                    f"{right.authority} imposes the same requirement that {detail}",
                    f"{left.authority} automatically proves {right.authority}'s requirement",
                    f"all supervisors impose identical {concept} obligations",
                ],
                "gold_evidence": [
                    evidence_obj(left, left_points),
                    evidence_obj(right, right_points),
                ],
                "counter_evidence": [
                    {
                        "claim": f"{right.authority} imposes the same {detail}",
                        "reason": "The documents may share a topic, but authority and scope cannot be transferred without explicit evidence.",
                        "document_path": left.path.as_posix(),
                        "section": left.section,
                    }
                ],
                "required_source_count": 2,
                "required_evidence_count": 2,
                "manual_focus": "The answer should reject cross authority scope transfer unless the evidence explicitly supports it.",
                "difficulty": "hallucination_stress",
            }
        )

    for index, ev in enumerate(selected[:40]):
        detail = FABRICATED_REQUIREMENTS[(index + 3) % len(FABRICATED_REQUIREMENTS)]
        concepts = support_points(ev, 5)
        rows.append(
            {
                "id": row_id(len(rows) + 1),
                "topic": ev.concepts[0],
                "question_type": "low_evidence_policy",
                "challenge_type": "partial_support_plus_unsupported_addon",
                "query": (
                    f"For {focus_phrase(ev, 2)}, does {ev.authority}'s {short_title(ev)} both "
                    f"support the general supervisory topic and require that {detail}?"
                ),
                "expected_behavior": "cautious_or_abstain",
                "gold_action": "qualify",
                "expected_answer_points": [
                    "general topic is supported",
                    "added requirement is not established",
                    "not established",
                ] + concepts[:4],
                "forbidden_claims": [
                    detail,
                    f"the supported {ev.concepts[0]} evidence proves the added fabricated requirement",
                ],
                "gold_evidence": [evidence_obj(ev, concepts[:4])],
                "counter_evidence": [
                    {
                        "claim": detail,
                        "reason": "The evidence supports the broad topic but not the added fabricated operational requirement.",
                        "document_path": ev.path.as_posix(),
                        "section": ev.section,
                    }
                ],
                "required_source_count": 1,
                "required_evidence_count": 1,
                "manual_focus": "The answer should separate supported general evidence from the unsupported added requirement.",
                "difficulty": "hallucination_stress",
            }
        )

    for index, row in enumerate(rows, start=1):
        row["id"] = row_id(index)
    return rows


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    queries = [row["query"].strip().lower() for row in rows]
    duplicate_queries = [query for query, count in Counter(queries).items() if count > 1]
    near_duplicates: list[dict[str, Any]] = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            score = SequenceMatcher(None, left["query"].lower(), right["query"].lower()).ratio()
            if score >= 0.90:
                near_duplicates.append(
                    {"left": left["id"], "right": right["id"], "similarity": round(score, 4)}
                )
                if len(near_duplicates) >= 20:
                    break
        if len(near_duplicates) >= 20:
            break

    missing_paths: list[str] = []
    missing_evidence: list[str] = []
    for row in rows:
        for ev in row.get("gold_evidence", []):
            path = PROJECT_ROOT / ev["document_path"]
            if not path.exists():
                missing_paths.append(ev["document_path"])
                continue
            doc_text = normalize_ws(path.read_text(encoding="utf-8", errors="ignore")).lower()
            if normalize_ws(ev.get("evidence_text", "")).lower() not in doc_text:
                missing_evidence.append(row["id"])

    return {
        "total_questions": len(rows),
        "category_distribution": dict(sorted(Counter(row["question_type"] for row in rows).items())),
        "challenge_distribution": dict(sorted(Counter(row["challenge_type"] for row in rows).items())),
        "unique_queries": len(set(queries)),
        "duplicate_queries": duplicate_queries,
        "near_duplicate_examples": near_duplicates,
        "missing_paths": unique(missing_paths),
        "missing_evidence_text_rows": unique(missing_evidence),
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
        default=Path("benchmarks/finreg/full_rag_questions_stress160.jsonl"),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("benchmarks/finreg/full_rag_questions_stress160_validation.json"),
    )
    args = parser.parse_args()

    evidence = load_evidence(PROJECT_ROOT / args.data_dir)
    rows = build_rows(evidence)
    validation = validate(rows)
    write_jsonl(PROJECT_ROOT / args.output, rows)
    (PROJECT_ROOT / args.validation_output).write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    if validation["duplicate_queries"] or validation["missing_paths"] or validation["missing_evidence_text_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
