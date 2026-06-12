#!/usr/bin/env python3
"""Build a duplicate checked FinReg full RAG benchmark from local evidence files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CONCEPT_PATTERNS: list[tuple[str, str]] = [
    ("board oversight", r"\bboard\b|\bboards\b"),
    ("senior management", r"senior management"),
    ("management body", r"management body|management bodies"),
    ("governance", r"\bgovernance\b"),
    ("risk culture", r"risk culture"),
    ("internal controls", r"internal controls?"),
    ("risk appetite", r"risk appetite"),
    ("data lineage", r"data lineage|traceability"),
    ("data quality", r"data quality|accurate|complete|timely"),
    ("risk data aggregation", r"risk data aggregation|RDA\b"),
    ("risk reporting", r"risk reporting|internal risk reporting|risk reports?"),
    ("stress testing", r"stress testing|stress scenarios?|\bstress\b"),
    ("crisis response", r"\bcrisis\b|crises|response"),
    ("third party risk", r"third[- ]party|third parties|outsourcing"),
    ("outsourcing", r"\boutsourcing\b|service provider"),
    ("cloud services", r"\bcloud\b"),
    ("ICT risk", r"\bICT\b|information and communication"),
    ("security risk", r"security risk|cyber|information security"),
    ("incident management", r"incident management|incident reporting|incidents"),
    ("business continuity", r"business continuity|continuity"),
    ("operational resilience", r"operational resilience|resilience"),
    ("climate risk", r"climate[- ]related|climate risk|climate"),
    ("physical risk", r"physical risk"),
    ("transition risk", r"transition risk"),
    ("ESG risk", r"\bESG\b|environmental, social and governance"),
    ("model risk", r"model risk|internal models?"),
    ("model validation", r"\bmodel validation\b|\bvalidation of models?\b|\bmodels?\b.{0,80}\bvalidation\b|\bvalidation\b.{0,80}\bmodels?\b"),
    ("validation", r"\bindependent validation\b|\bvalidation processes?\b|\bvalidation of\b|\bvalidation\b"),
    ("liquidity risk", r"liquidity risk|\bliquidity\b"),
    ("intraday liquidity", r"intraday liquidity|payment and settlement"),
    ("payment obligations", r"payment obligations|settlement obligations|payment behaviour"),
    ("capital adequacy", r"capital adequacy|capital requirements?|\bcapital\b"),
    ("SREP", r"\bSREP\b|supervisory review"),
    ("business model", r"business model|viability|sustainability"),
    ("ICAAP", r"\bICAAP\b|internal capital"),
    ("ILAAP", r"\bILAAP\b|internal liquidity"),
    ("remuneration", r"remuneration|bonus|variable remuneration"),
    ("suitability", r"suitability|fit and proper|knowledge, skills and experience"),
    ("conflicts of interest", r"conflicts? of interest"),
    ("branch supervision", r"branches|subsidiar(?:y|ies)|third country"),
    ("supervisory assessment", r"supervisory assessment|supervisors?|competent authorities"),
    ("risk management framework", r"risk management framework|risk management"),
]


AUTHORITY_BY_FOLDER = {
    "bcbs": "BCBS",
    "eba": "EBA",
    "ecb": "ECB",
    "fed_occ": "Federal Reserve",
    "pra_boe": "PRA",
}


FALSE_CLAIMS = [
    "a single regulator hosted XML portal is mandatory",
    "all firms must use the same vendor tool",
    "manual senior sign off replaces the control framework",
    "a fixed five business day remediation deadline applies in every case",
    "public disclosure is required in real time for every incident",
    "the source bans the activity entirely for all banks",
    "board oversight can be removed if automation is used",
    "a universal numerical threshold applies to every institution",
]


UNSUPPORTED_DETAILS = [
    "the exact portal name",
    "a fixed remediation deadline",
    "a mandatory vendor certification",
    "a universal quantitative threshold",
    "a public disclosure timetable",
    "a single global template",
    "a regulator approval workflow for every case",
    "a required software product",
]


@dataclass(frozen=True)
class Evidence:
    path: Path
    title: str
    authority: str
    section: str
    text: str
    concepts: tuple[str, ...]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def title_from_path(path: Path) -> str:
    first = ""
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            cleaned = normalize_ws(line)
            if cleaned:
                first = cleaned
                break
    except OSError:
        pass
    if first and len(first) <= 140 and not first.lower().startswith("[page "):
        return first
    return path.stem.replace("_", " ").title()


def authority_from_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    for folder, authority in AUTHORITY_BY_FOLDER.items():
        if folder in parts:
            return authority
    return path.parent.name.upper()


def detect_concepts(text: str) -> list[str]:
    found: list[str] = []
    for concept, pattern in CONCEPT_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(concept)
    return found


def split_paragraphs(text: str) -> list[str]:
    chunks = [normalize_ws(chunk) for chunk in re.split(r"\n\s*\n+", text)]
    paragraphs: list[str] = []
    for chunk in chunks:
        lower = chunk.lower()
        if any(
            marker in lower
            for marker in (
                "table of contents",
                "contents 1.",
                "legal information",
                "copyright and permissions",
                "privacy notice",
                "email scam warning",
            )
        ):
            continue
        if len(re.findall(r"\b\d+(?:\.\d+)*\b", chunk)) >= 18:
            continue
        if len(chunk) < 220:
            continue
        if len(chunk) > 1300:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            window: list[str] = []
            for sentence in sentences:
                if not sentence:
                    continue
                window.append(sentence)
                joined = normalize_ws(" ".join(window))
                if len(joined) >= 380:
                    paragraphs.append(joined[:1300])
                    window = []
            if window:
                joined = normalize_ws(" ".join(window))
                if len(joined) >= 220:
                    paragraphs.append(joined[:1300])
        else:
            paragraphs.append(chunk)
    return paragraphs


def load_evidence(data_dir: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    for path in sorted(data_dir.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = title_from_path(path)
        authority = authority_from_path(path)
        for index, paragraph in enumerate(split_paragraphs(text), start=1):
            concepts = detect_concepts(paragraph)
            if len(concepts) < 2:
                continue
            section = f"{title} paragraph {index}"
            evidence.append(
                Evidence(
                    path=path.relative_to(PROJECT_ROOT),
                    title=title,
                    authority=authority,
                    section=section,
                    text=paragraph,
                    concepts=tuple(concepts[:8]),
                )
            )
    evidence.sort(key=lambda item: (-len(item.concepts), item.authority, item.title, item.section))
    return evidence


def unique(values: list[str], limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_ws(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            out.append(cleaned)
        if limit is not None and len(out) >= limit:
            break
    return out


def evidence_obj(ev: Evidence, supports: list[str]) -> dict[str, Any]:
    return {
        "document_path": ev.path.as_posix(),
        "document_name": ev.title,
        "authority": ev.authority,
        "section": ev.section,
        "evidence_text": ev.text,
        "supports": supports,
    }


def support_points(ev: Evidence, limit: int = 5) -> list[str]:
    points = list(ev.concepts[:limit])
    if ev.authority not in points:
        points.append(ev.authority)
    return unique(points, limit)


def short_title(ev: Evidence) -> str:
    title = ev.title
    if len(title) > 80:
        title = ev.path.stem.replace("_", " ").title()
    return title


def focus_phrase(ev: Evidence, width: int = 3) -> str:
    concepts = list(ev.concepts[:width])
    return ", ".join(concepts[:-1]) + f", and {concepts[-1]}" if len(concepts) > 1 else concepts[0]


def choose_evidence(evidence: list[Evidence], count: int) -> list[Evidence]:
    selected: list[Evidence] = []
    doc_use: Counter[str] = Counter()
    concept_use: Counter[str] = Counter()
    query_shapes: set[tuple[str, str, tuple[str, ...]]] = set()
    for ev in evidence:
        key = ev.path.as_posix()
        if doc_use[key] >= 5:
            continue
        primary = ev.concepts[0]
        if concept_use[primary] >= 7:
            continue
        query_shape = (ev.authority, short_title(ev), tuple(ev.concepts[:3]))
        if query_shape in query_shapes:
            continue
        selected.append(ev)
        query_shapes.add(query_shape)
        doc_use[key] += 1
        concept_use[primary] += 1
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise SystemExit(f"Only selected {len(selected)} evidence passages, need {count}.")
    return selected


def build_cross_pairs(evidence: list[Evidence], count: int) -> list[tuple[Evidence, Evidence, str]]:
    by_concept: dict[str, list[Evidence]] = defaultdict(list)
    for ev in evidence:
        for concept in ev.concepts[:5]:
            by_concept[concept].append(ev)

    pairs: list[tuple[Evidence, Evidence, str]] = []
    used: set[tuple[str, str, str]] = set()
    for concept, items in sorted(by_concept.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        for left in items:
            for right in items:
                if left.path == right.path or left.authority == right.authority:
                    continue
                key = tuple(sorted([left.path.as_posix(), right.path.as_posix()])) + (concept,)
                if key in used:
                    continue
                used.add(key)
                pairs.append((left, right, concept))
                if len(pairs) >= count:
                    return pairs
    raise SystemExit(f"Only built {len(pairs)} cross source pairs, need {count}.")


def row_id(index: int) -> str:
    return f"clean_eval_{index:03d}"


def build_rows(evidence: list[Evidence]) -> list[dict[str, Any]]:
    selected = choose_evidence(evidence, 40)
    cross_pairs = build_cross_pairs(evidence, 40)
    rows: list[dict[str, Any]] = []

    for idx, ev in enumerate(selected, start=1):
        concepts = support_points(ev, 5)
        focus = focus_phrase(ev)
        supported_templates = [
            (
                f"According to {ev.authority}, what supervisory issue is addressed in "
                f"{short_title(ev)} around {focus}?"
            ),
            (
                f"Using only {ev.authority}'s {short_title(ev)} {ev.section.rsplit(' paragraph ', 1)[-1] if ' paragraph ' in ev.section else ev.section}, "
                f"summarize the source-backed point on {focus_phrase(ev, 2)}."
            ),
            (
                f"Prepare a brief evidence note from {short_title(ev)} for {focus_phrase(ev, 3)} "
                f"under {ev.authority}'s material."
            ),
            (
                f"What should a reviewer take from the cited {ev.authority} passage in "
                f"{short_title(ev)} about {focus_phrase(ev, 2)}?"
            ),
            (
                f"Identify the supported controls or expectations in {ev.authority}'s "
                f"{short_title(ev)} that relate to {focus_phrase(ev, 3)}."
            ),
            (
                f"From the cited passage, explain how {ev.authority} treats "
                f"{focus_phrase(ev, 2)} in {short_title(ev)}."
            ),
        ]
        rows.append(
            {
                "id": row_id(len(rows) + 1),
                "topic": ev.concepts[0],
                "question_type": "factual_supported",
                "challenge_type": "evidence_grounded_summary",
                "query": supported_templates[(idx - 1) % len(supported_templates)],
                "expected_behavior": "answer_with_source_support",
                "gold_action": "answer",
                "expected_answer_points": concepts,
                "forbidden_claims": [
                    FALSE_CLAIMS[idx % len(FALSE_CLAIMS)],
                    f"{ev.authority} replaces governance judgment with automatic approval for {ev.concepts[0]}",
                ],
                "gold_evidence": [evidence_obj(ev, concepts)],
                "counter_evidence": [],
                "required_source_count": 1,
                "required_evidence_count": 1,
                "manual_focus": "The answer should cover the supported concepts without adding precise operational requirements not stated in the source.",
                "difficulty": "clean_hard",
            }
        )

    for idx, ev in enumerate(selected, start=1):
        concepts = support_points(ev, 5)
        false_claim = FALSE_CLAIMS[idx % len(FALSE_CLAIMS)]
        false_templates = [
            (
                f"Does {ev.authority}'s {short_title(ev)} material on {focus_phrase(ev, 2)} "
                f"say that {false_claim}?"
            ),
            (
                f"A policy memo asserts that {false_claim}. Does the cited {ev.authority} "
                f"evidence in {short_title(ev)} establish that claim?"
            ),
            (
                f"Check this claim against {ev.authority}'s source on {focus_phrase(ev, 2)}: "
                f"{false_claim}."
            ),
            (
                f"Should the system accept or reject the statement that {false_claim}, "
                f"based only on {short_title(ev)}?"
            ),
            (
                f"For {focus_phrase(ev, 2)}, is there source support in {ev.authority}'s "
                f"{short_title(ev)} for the claim that {false_claim}?"
            ),
        ]
        rows.append(
            {
                "id": row_id(len(rows) + 1),
                "topic": ev.concepts[0],
                "question_type": "false_premise",
                "challenge_type": "plausible_but_unsupported_premise",
                "query": false_templates[(idx - 1) % len(false_templates)],
                "expected_behavior": "refute_or_abstain",
                "gold_action": "refute",
                "expected_answer_points": ["no", "not established"] + concepts[:4],
                "forbidden_claims": [
                    false_claim,
                    f"the {ev.authority} source establishes that fabricated requirement",
                ],
                "gold_evidence": [evidence_obj(ev, concepts[:4])],
                "counter_evidence": [
                    {
                        "claim": false_claim,
                        "reason": "The evidence supports the regulatory topic but does not establish the fabricated operational requirement.",
                        "document_path": ev.path.as_posix(),
                        "section": ev.section,
                    }
                ],
                "required_source_count": 1,
                "required_evidence_count": 1,
                "manual_focus": "The answer should reject the premise or abstain, not accept a nearby but unsupported regulatory detail.",
                "difficulty": "clean_hard",
            }
        )

    for idx, ev in enumerate(selected, start=1):
        concepts = support_points(ev, 5)
        detail = UNSUPPORTED_DETAILS[idx % len(UNSUPPORTED_DETAILS)]
        detail_templates = [
            (
                f"If {ev.authority}'s {short_title(ev)} evidence discusses {focus_phrase(ev, 2)} "
                f"but does not specify {detail}, how should the system answer?"
            ),
            (
                f"Before drafting a control, verify whether {short_title(ev)} gives {detail} "
                f"for {focus_phrase(ev, 2)}."
            ),
            (
                f"The source covers {focus_phrase(ev, 2)}. Can the answer include {detail}, "
                f"or should it qualify the evidence?"
            ),
            (
                f"How should a cautious answer handle a request for {detail} when the cited "
                f"{ev.authority} passage only supports {focus_phrase(ev, 2)}?"
            ),
            (
                f"Assess whether {ev.authority}'s cited material supplies {detail} for "
                f"{focus_phrase(ev, 2)}."
            ),
        ]
        rows.append(
            {
                "id": row_id(len(rows) + 1),
                "topic": ev.concepts[0],
                "question_type": "low_evidence_policy",
                "challenge_type": "specific_detail_not_in_evidence",
                "query": detail_templates[(idx - 1) % len(detail_templates)],
                "expected_behavior": "cautious_or_abstain",
                "gold_action": "qualify",
                "expected_answer_points": [
                    "not established",
                    "avoid inventing",
                    detail,
                ] + concepts[:3],
                "forbidden_claims": [
                    detail,
                    f"{ev.authority} requires that specific detail for every institution",
                ],
                "gold_evidence": [evidence_obj(ev, concepts[:4])],
                "counter_evidence": [
                    {
                        "claim": detail,
                        "reason": "The evidence supports the general topic but not the requested precise detail.",
                        "document_path": ev.path.as_posix(),
                        "section": ev.section,
                    }
                ],
                "required_source_count": 1,
                "required_evidence_count": 1,
                "manual_focus": "The answer should state the supported topic and clearly avoid inventing the missing detail.",
                "difficulty": "clean_hard",
            }
        )

    for idx, (left, right, concept) in enumerate(cross_pairs, start=1):
        left_points = support_points(left, 3)
        right_points = support_points(right, 3)
        points = unique(left_points + right_points + ["scope is not identical"], 7)
        cross_templates = [
            (
                f"How should {left.authority}'s {short_title(left)} and {right.authority}'s "
                f"{short_title(right)} be compared on {concept} without assuming their scope "
                "or supervisory emphasis is identical?"
            ),
            (
                f"Give a cautious comparison of {concept} across {left.authority}'s "
                f"{short_title(left)} and {right.authority}'s {short_title(right)}, using only "
                "the two cited passages."
            ),
            (
                f"What source-backed overlap and limits appear when comparing {left.authority}'s "
                f"{short_title(left)} and {right.authority}'s {short_title(right)} treatment of "
                f"{concept}?"
            ),
            (
                f"For {concept}, explain what {left.authority}'s {short_title(left)} and "
                f"{right.authority}'s {short_title(right)} share and why one source should not "
                "be treated as replacing the other."
            ),
            (
                f"Compare the cited {left.authority} evidence from {short_title(left)} and "
                f"{right.authority} evidence from {short_title(right)} on {concept}, making "
                "clear where the evidence should stay qualified."
            ),
        ]
        rows.append(
            {
                "id": row_id(len(rows) + 1),
                "topic": concept,
                "question_type": "cross_source_nuanced",
                "challenge_type": "cross_authority_scope_comparison",
                "query": cross_templates[(idx - 1) % len(cross_templates)],
                "expected_behavior": "cautious_synthesis",
                "gold_action": "qualify",
                "expected_answer_points": points,
                "forbidden_claims": [
                    f"{left.authority} and {right.authority} impose identical {concept} requirements",
                    f"one source fully replaces the other source on {concept}",
                    f"all supervisors use the same operational rule for {concept}",
                ],
                "gold_evidence": [
                    evidence_obj(left, left_points),
                    evidence_obj(right, right_points),
                ],
                "counter_evidence": [
                    {
                        "claim": f"{left.authority} and {right.authority} impose identical {concept} requirements",
                        "reason": "The sources share a theme but come from different authorities and should not be flattened into one identical obligation.",
                        "document_path": left.path.as_posix(),
                        "section": left.section,
                    }
                ],
                "required_source_count": 2,
                "required_evidence_count": 2,
                "manual_focus": "The answer should compare the sources carefully and avoid treating related expectations as identical.",
                "difficulty": "clean_hard",
            }
        )

    for index, row in enumerate(rows, start=1):
        row["id"] = row_id(index)
    return rows


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    queries = [row["query"].strip().lower() for row in rows]
    duplicates = [query for query, count in Counter(queries).items() if count > 1]
    near_duplicates: list[dict[str, Any]] = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            score = SequenceMatcher(
                None,
                left["query"].lower(),
                right["query"].lower(),
            ).ratio()
            if score >= 0.86:
                near_duplicates.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        "similarity": round(score, 4),
                    }
                )
                if len(near_duplicates) >= 20:
                    break
        if len(near_duplicates) >= 20:
            break

    missing_paths: list[str] = []
    missing_text: list[str] = []
    for row in rows:
        for ev in row.get("gold_evidence", []):
            path = PROJECT_ROOT / ev["document_path"]
            if not path.exists():
                missing_paths.append(ev["document_path"])
                continue
            doc_text = normalize_ws(path.read_text(encoding="utf-8", errors="ignore")).lower()
            if normalize_ws(ev.get("evidence_text", "")).lower() not in doc_text:
                missing_text.append(row["id"])

    return {
        "total_questions": len(rows),
        "category_distribution": dict(sorted(Counter(row["question_type"] for row in rows).items())),
        "challenge_distribution": dict(sorted(Counter(row["challenge_type"] for row in rows).items())),
        "unique_queries": len(set(queries)),
        "duplicate_queries": duplicates,
        "near_duplicate_examples": near_duplicates,
        "missing_paths": unique(missing_paths),
        "missing_evidence_text_rows": unique(missing_text),
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
    parser.add_argument("--output", type=Path, default=Path("benchmarks/finreg/full_rag_questions_clean160.jsonl"))
    parser.add_argument("--validation-output", type=Path, default=Path("benchmarks/finreg/full_rag_questions_clean160_validation.json"))
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
