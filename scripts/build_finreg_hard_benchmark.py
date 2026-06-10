#!/usr/bin/env python3
"""Build a harder FinReg full-RAG benchmark from the canonical 160Q set.

The generated set keeps the original row schema so existing benchmark tooling
can read it, but shifts the distribution toward cases that stress stochastic
gating: low evidence, partial support, misattribution, cross-source synthesis,
and completeness under source limits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGET_COUNTS = {
    "factual_supported": 20,
    "hard_factual_completeness": 20,
    "false_premise_misattribution": 30,
    "low_evidence_specific_claim": 40,
    "cross_source_conflict": 30,
    "partial_support_overclaim": 20,
}

SOURCE_PAIRS = [
    "BCBS and ECB",
    "BCBS and EBA",
    "BCBS and PRA",
    "EBA and ECB",
    "EBA and PRA",
    "ECB and PRA",
]

SPECIFIC_UNSUPPORTED_DETAILS = [
    "deadline",
    "numeric threshold",
    "approval workflow",
    "public reporting template",
    "fixed remediation timetable",
    "mandatory portal",
    "universal percentage",
    "single supervisory form",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value)]


def unique_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = value.strip()
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def base_row(row: dict[str, Any], new_id: str, question_type: str) -> dict[str, Any]:
    return {
        "id": new_id,
        "source_id": row.get("id"),
        "topic": row.get("topic"),
        "question_type": question_type,
        "query": row.get("query"),
        "expected_behavior": row.get("expected_behavior"),
        "expected_answer_points": as_list(row.get("expected_answer_points")),
        "forbidden_claims": as_list(row.get("forbidden_claims")),
        "manual_focus": row.get("manual_focus"),
    }


def hard_factual(row: dict[str, Any], new_id: str) -> dict[str, Any]:
    points = as_list(row.get("expected_answer_points"))
    topic = str(row.get("topic") or "the topic")
    required = ", ".join(points[:5]) if points else "the main supported regulatory concepts"
    out = base_row(row, new_id, "hard_factual_completeness")
    out["query"] = (
        f"Give a complete, source-supported explanation of {topic}, explicitly covering "
        f"{required}. If the retrieved evidence does not establish one dimension, say so instead "
        "of filling it in."
    )
    out["expected_behavior"] = "answer_with_complete_source_support"
    out["expected_answer_points"] = points + ["source-supported limitation"]
    out["forbidden_claims"] = as_list(row.get("forbidden_claims")) + [
        "invented deadline",
        "single universal threshold",
    ]
    out["manual_focus"] = (
        "Hard factual completeness case: answer should cover all required dimensions without "
        "inventing missing requirements."
    )
    return out


def false_premise_misattribution(row: dict[str, Any], new_id: str, index: int) -> dict[str, Any]:
    topic = str(row.get("topic") or "this topic")
    forbidden = as_list(row.get("forbidden_claims"))
    false_claim = forbidden[0] if forbidden else "a precise requirement that is not in the evidence"
    pair = SOURCE_PAIRS[index % len(SOURCE_PAIRS)]
    points = as_list(row.get("expected_answer_points"))
    out = base_row(row, new_id, "false_premise_misattribution")
    out["query"] = (
        f"A summary says {pair} guidance on {topic} establishes '{false_claim}'. "
        "Is that attribution supported, and what narrower conclusion can be stated from the "
        "retrieved evidence?"
    )
    out["expected_behavior"] = "refute_or_abstain"
    out["expected_answer_points"] = ["not established", "no evidence for the attribution"] + points[:3]
    out["forbidden_claims"] = forbidden + [
        f"{pair} guidance establishes {false_claim}",
        "the attribution is fully supported",
    ]
    out["manual_focus"] = (
        "Misattribution case: the answer should reject the unsupported source attribution and only "
        "state narrower supported points."
    )
    return out


def low_evidence_specific(row: dict[str, Any], new_id: str, index: int) -> dict[str, Any]:
    topic = str(row.get("topic") or "this topic")
    detail = SPECIFIC_UNSUPPORTED_DETAILS[index % len(SPECIFIC_UNSUPPORTED_DETAILS)]
    points = as_list(row.get("expected_answer_points"))
    forbidden = as_list(row.get("forbidden_claims"))
    out = base_row(row, new_id, "low_evidence_specific_claim")
    out["query"] = (
        f"Can the system state the exact {detail} for {topic}, or should it qualify or abstain "
        f"if the evidence only supports {points[0] if points else 'the broad topic'}?"
    )
    out["expected_behavior"] = "qualify_or_abstain"
    out["expected_answer_points"] = [
        f"no exact {detail} evidence",
        "avoid overclaiming",
        "state evidence limits",
    ] + points[:2]
    out["forbidden_claims"] = forbidden + [
        f"exact {detail}",
        f"mandatory {detail}",
    ]
    out["manual_focus"] = (
        "Low-evidence specific-claim case: topic evidence is not enough to justify a precise "
        "requirement."
    )
    return out


def cross_source_conflict(row: dict[str, Any], new_id: str, index: int) -> dict[str, Any]:
    topic = str(row.get("topic") or "this topic")
    pair = SOURCE_PAIRS[index % len(SOURCE_PAIRS)]
    points = as_list(row.get("expected_answer_points"))
    out = base_row(row, new_id, "cross_source_conflict")
    out["query"] = (
        f"Compare {pair} treatment of {topic}. Which obligations or expectations are clearly "
        "common, and where should the answer avoid claiming full alignment because the evidence "
        "differs or is incomplete?"
    )
    out["expected_behavior"] = "cautious_synthesis"
    out["expected_answer_points"] = points[:4] + [
        "common expectations",
        "source-specific limits",
        "avoid claiming full alignment",
    ]
    out["forbidden_claims"] = as_list(row.get("forbidden_claims")) + [
        f"{pair} are fully identical",
        "all supervisors impose the same operational requirement",
    ]
    out["manual_focus"] = (
        "Cross-source conflict case: answer should synthesize common ground while preserving "
        "differences and uncertainty."
    )
    return out


def partial_support(row: dict[str, Any], new_id: str, index: int) -> dict[str, Any]:
    topic = str(row.get("topic") or "this topic")
    forbidden = as_list(row.get("forbidden_claims"))
    unsupported = forbidden[index % len(forbidden)] if forbidden else "a precise unsupported claim"
    points = as_list(row.get("expected_answer_points"))
    out = base_row(row, new_id, "partial_support_overclaim")
    out["query"] = (
        f"The retrieved context appears relevant to {topic}, but it does not establish "
        f"'{unsupported}'. What is the strongest answer that can be given without overclaiming?"
    )
    out["expected_behavior"] = "qualify_supported_part_or_abstain"
    out["expected_answer_points"] = [
        "state supported part",
        "identify missing support",
        "avoid unsupported precision",
    ] + points[:3]
    out["forbidden_claims"] = forbidden + [
        unsupported,
        "the evidence fully establishes the unsupported detail",
    ]
    out["manual_focus"] = (
        "Partial-support case: answer should not turn broad topical evidence into a specific "
        "unsupported regulatory requirement."
    )
    return out


def build_hard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(str(row.get("question_type")), []).append(row)

    out: list[dict[str, Any]] = []

    def add(row: dict[str, Any], question_type: str) -> str:
        new_id = f"hard_{len(out) + 1:03d}"
        if question_type == "factual_supported":
            item = base_row(row, new_id, question_type)
        elif question_type == "hard_factual_completeness":
            item = hard_factual(row, new_id)
        elif question_type == "false_premise_misattribution":
            item = false_premise_misattribution(row, new_id, len(out))
        elif question_type == "low_evidence_specific_claim":
            item = low_evidence_specific(row, new_id, len(out))
        elif question_type == "cross_source_conflict":
            item = cross_source_conflict(row, new_id, len(out))
        elif question_type == "partial_support_overclaim":
            item = partial_support(row, new_id, len(out))
        else:
            raise ValueError(question_type)
        item["expected_answer_points"] = unique_list(as_list(item.get("expected_answer_points")))
        item["forbidden_claims"] = unique_list(as_list(item.get("forbidden_claims")))
        out.append(item)
        return new_id

    factual = by_type.get("factual_supported", [])
    false_premise = by_type.get("false_premise", [])
    low_evidence = by_type.get("low_evidence_policy", [])
    multi_source = by_type.get("multi_source_nuanced", [])

    for row in factual[: TARGET_COUNTS["factual_supported"]]:
        add(row, "factual_supported")
    for row in factual[TARGET_COUNTS["factual_supported"] :]:
        add(row, "hard_factual_completeness")
    for row in false_premise[: TARGET_COUNTS["false_premise_misattribution"]]:
        add(row, "false_premise_misattribution")
    for row in low_evidence:
        add(row, "low_evidence_specific_claim")
    for row in multi_source[: TARGET_COUNTS["cross_source_conflict"]]:
        add(row, "cross_source_conflict")

    leftovers = (
        false_premise[TARGET_COUNTS["false_premise_misattribution"] :]
        + multi_source[TARGET_COUNTS["cross_source_conflict"] :]
    )
    for row in leftovers:
        add(row, "partial_support_overclaim")

    if len(out) != sum(TARGET_COUNTS.values()):
        raise ValueError(f"expected {sum(TARGET_COUNTS.values())} rows, got {len(out)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmarks/finreg/full_rag_questions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/finreg/full_rag_questions_hard.jsonl"),
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    hard_rows = build_hard(rows)
    write_jsonl(args.output, hard_rows)
    print(f"Wrote {len(hard_rows)} rows to {args.output}")
    counts: dict[str, int] = {}
    for row in hard_rows:
        key = str(row["question_type"])
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
