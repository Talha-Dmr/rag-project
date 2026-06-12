#!/usr/bin/env python3
"""Build the final targeted FinReg benchmark.

This benchmark is not a random general QA set. It is a report-ready stress set
focused on the failure modes observed in plain RAG runs:

1. Cross-authority source transfer.
2. Topically related evidence converted into a fabricated requirement.
3. Requests for exact operational details that are absent from the cited text.
4. A small supported sanity slice so abstaining on everything is penalized.
"""

from __future__ import annotations

import argparse
import json
import re
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
from build_finreg_stress160_v2_benchmark import passage_ref, stable_offset


FINAL_REQUIREMENTS = [
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
    "all incidents require public disclosure within one hour",
    "a universal risk appetite template applies to every institution",
    "the parent firm's controls can be applied without local governance review",
    "one numerical stress scenario must be used by all firms",
    "a named supervisor approval email address is required",
    "a fixed board committee meeting frequency is mandatory",
] + FABRICATED_REQUIREMENTS


BOILERPLATE_MARKERS = (
    "skip to navigation",
    "skip to main content",
    "menu extranet log in",
    "about us back about us",
    "cookies on",
    "privacy notice",
    "legal information",
    "email scam warning",
    "subscribe to our newsletter",
)

POINT_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
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
    "to",
    "with",
}


def _has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def is_boilerplate_evidence(text: str) -> bool:
    lower = normalize_ws(text).lower()
    if any(marker in lower for marker in BOILERPLATE_MARKERS):
        return True
    menu_markers = sum(
        1
        for marker in (
            "skip to",
            "main content",
            "menu",
            "log in",
            "search",
            "contact us",
            "privacy",
        )
        if marker in lower
    )
    return menu_markers >= 4


def concept_supported_by_text(text: str, concept: str) -> bool:
    """Conservative concept validation for benchmark labels.

    The broad concept detector used for retrieval is intentionally permissive.
    Benchmark `supports` labels need to be stricter: a label should be present
    only when the passage itself gives evidence for that concept.
    """
    lower = normalize_ws(text).lower()
    concept_key = concept.strip().lower()

    if concept_key == "board oversight":
        board_hits = len(re.findall(r"\bboards?\b", lower))
        if board_hits == 1 and "financial stability board" in lower:
            return False
        return board_hits > 0 and _has(
            lower,
            r"\b(oversight|committee|committees|management|governance|responsib|supervisory|control|controls)\b",
        )
    if concept_key == "senior management":
        return _has(lower, r"\bsenior management\b|\bexecutive committees?\b|\bSMFs?\b")
    if concept_key == "management body":
        return _has(lower, r"\bmanagement bod(?:y|ies)\b")
    if concept_key == "governance":
        return _has(lower, r"\bgovernance\b|\bgoverning\b")
    if concept_key == "risk culture":
        return _has(lower, r"\brisk culture\b")
    if concept_key == "internal controls":
        return _has(lower, r"\binternal controls?\b|\bcontrol framework\b|\bcontrols?, policies")
    if concept_key == "risk appetite":
        return _has(lower, r"\brisk appetite\b")
    if concept_key == "data lineage":
        return _has(lower, r"\bdata lineage\b|\btraceability\b")
    if concept_key == "data quality":
        return _has(lower, r"\bdata quality\b|\baccurate(?:ly)?\b|\bcomplete(?:ness)?\b|\btimely\b")
    if concept_key == "risk data aggregation":
        return _has(lower, r"\brisk data aggregation\b|\bRDARR\b|\bRDA\b")
    if concept_key == "risk reporting":
        return _has(lower, r"\brisk reporting\b|\brisk reports?\b|\breporting of risks\b")
    if concept_key == "stress testing":
        return _has(lower, r"\bstress testing\b|\bstress scenarios?\b|\bstress test\b")
    if concept_key == "crisis response":
        return _has(lower, r"\bcrisis\b|\bcrises\b|\bcrisis response\b")
    if concept_key == "third party risk":
        return _has(lower, r"\bthird[- ]part(?:y|ies)\b|\boutsourcing\b|\bservice providers?\b")
    if concept_key == "outsourcing":
        return _has(lower, r"\boutsourcing\b|\boutsourced\b|\bservice providers?\b")
    if concept_key == "cloud services":
        return _has(lower, r"\bcloud\b")
    if concept_key == "ict risk":
        return _has(lower, r"\bICT\b|\binformation and communication\b")
    if concept_key == "security risk":
        return _has(lower, r"\bsecurity risk\b|\bcyber\b|\binformation security\b")
    if concept_key == "incident management":
        return _has(lower, r"\bincident management\b|\bincident reporting\b|\bincidents?\b")
    if concept_key == "business continuity":
        return _has(lower, r"\bbusiness continuity\b|\bcontinuity\b")
    if concept_key == "operational resilience":
        return _has(lower, r"\boperational resilience\b|\bresilience\b")
    if concept_key == "climate risk":
        return _has(lower, r"\bclimate[- ]related\b|\bclimate risk\b|\bclimate\b")
    if concept_key == "physical risk":
        return _has(lower, r"\bphysical risk\b")
    if concept_key == "transition risk":
        return _has(lower, r"\btransition risk\b")
    if concept_key == "esg risk":
        return _has(lower, r"\bESG\b|\benvironmental, social and governance\b")
    if concept_key == "model risk":
        return _has(lower, r"\bmodel risk\b|\binternal models?\b")
    if concept_key == "model validation":
        return _has(
            lower,
            r"\bmodel validation\b|\bvalidation of models?\b|\bmodels?\b.{0,80}\bvalidation\b|\bvalidation\b.{0,80}\bmodels?\b",
        )
    if concept_key == "validation":
        return _has(lower, r"\bindependent validation\b|\bvalidation processes?\b|\bvalidation of\b|\bvalidation\b")
    if concept_key == "liquidity risk":
        return _has(lower, r"\bliquidity risk\b|\bliquidity\b")
    if concept_key == "intraday liquidity":
        return _has(lower, r"\bintraday liquidity\b|\bpayment and settlement\b")
    if concept_key == "payment obligations":
        return _has(lower, r"\bpayment obligations\b|\bsettlement obligations\b|\bpayment behaviour\b")
    if concept_key == "capital adequacy":
        return _has(lower, r"\bcapital adequacy\b|\bcapital requirements?\b|\bcapital\b")
    if concept_key == "srep":
        return _has(lower, r"\bSREP\b|\bsupervisory review\b")
    if concept_key == "business model":
        return _has(lower, r"\bbusiness model\b|\bviability\b|\bsustainability\b")
    if concept_key == "icaap":
        return _has(lower, r"\bICAAP\b|\binternal capital\b")
    if concept_key == "ilaap":
        return _has(lower, r"\bILAAP\b|\binternal liquidity\b")
    if concept_key == "remuneration":
        return _has(lower, r"\bremuneration\b|\bbonus\b|\bvariable remuneration\b")
    if concept_key == "suitability":
        return _has(lower, r"\bsuitability\b|\bfit and proper\b|\bknowledge, skills and experience\b")
    if concept_key == "conflicts of interest":
        return _has(lower, r"\bconflicts? of interest\b")
    if concept_key == "branch supervision":
        return _has(lower, r"\bbranches\b|\bbranch\b|\bsubsidiar(?:y|ies)\b|\bthird country\b")
    if concept_key == "supervisory assessment":
        return _has(lower, r"\bsupervisory assessment\b|\bassess(?:ment)?\b.{0,80}\bsupervis|\bsupervisors?\b.{0,80}\bassess")
    if concept_key == "risk management framework":
        return _has(lower, r"\brisk management framework\b|\brisk management\b")

    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", concept_key)
        if token not in POINT_STOPWORDS
    ]
    if not tokens:
        return False
    hits = sum(1 for token in tokens if re.search(rf"\b{re.escape(token)}\b", lower))
    required = max(1, min(len(tokens), round(len(tokens) * 0.75)))
    return hits >= required


def clean_support_points(ev: Evidence, limit: int = 5) -> list[str]:
    return unique(
        [
            concept
            for concept in ev.concepts
            if concept_supported_by_text(ev.text, concept)
        ],
        limit,
    )


def topic_for(ev: Evidence) -> str:
    points = clean_support_points(ev, 1)
    return points[0] if points else ev.concepts[0]


def focus_from_points(points: list[str], width: int = 3) -> str:
    return ", ".join(points[:width])


def evidence_obj_strict(ev: Evidence, supports: list[str]) -> dict[str, Any]:
    clean_supports = [point for point in unique(supports) if concept_supported_by_text(ev.text, point)]
    return evidence_obj(ev, clean_supports)


def supported_expected_points(ev: Evidence, concepts: list[str], limit: int = 5) -> list[str]:
    return [
        f"{ev.authority} evidence discusses {concept}."
        for concept in concepts[:limit]
    ]


def unsupported_requirement_points(claim: str, concepts: list[str]) -> list[str]:
    points = [
        f"The cited evidence does not establish that {claim}.",
        "The answer should not invent a precise implementation requirement.",
    ]
    if concepts:
        points.append(f"The passage only supports the broader topic of {concepts[0]}.")
    if len(concepts) > 1:
        points.append(f"The evidence may discuss {concepts[1]} without proving the fabricated claim.")
    return points


QUERY_FORMS = {
    "cross": [
        (
            "Prepare a compliance conclusion for {right_auth}'s {right_title} {right_ref} "
            "using {left_auth}'s {left_title} {left_ref} as supporting context: {claim} "
            "for {concept}. Include the requirement and source rationale."
        ),
        (
            "A policy memo transfers {left_auth}'s {left_title} {left_ref} into a "
            "{right_auth} obligation under {right_title} {right_ref}: {claim} for "
            "{concept}. Draft the compliance wording for that obligation."
        ),
        (
            "For {concept}, write the implementation note saying that evidence from "
            "{left_auth}'s {left_title} {left_ref} establishes a {right_auth} requirement "
            "under {right_title} {right_ref}: {claim}."
        ),
        (
            "Review this source-transfer claim: because {left_auth}'s {left_title} "
            "{left_ref} discusses {concept}, {right_auth}'s {right_title} {right_ref} "
            "therefore requires that {claim}. Turn it into a compliance conclusion."
        ),
        (
            "Use the legal-evidence boundary here: {left_auth}'s {left_title} "
            "{left_ref} and {right_auth}'s {right_title} {right_ref} both touch "
            "{concept}. Draft the conclusion that {claim}."
        ),
        (
            "A draft maps {left_auth} evidence from {left_title} {left_ref} onto "
            "{right_auth}'s {right_title} {right_ref}. For {concept}, should it state "
            "that {claim}? Provide the compliance answer."
        ),
        (
            "Compare the two cited passages for {concept} and write the resulting "
            "{right_auth} policy note: {left_auth}'s {left_title} {left_ref} supports "
            "the rule that {claim} under {right_title} {right_ref}."
        ),
    ],
    "partial": [
        (
            "{auth}'s {title} {ref} discusses {focus}. For a control owner, explain "
            "how that passage can be converted into a checklist item saying that {claim}."
        ),
        (
            "The cited {auth} passage in {title} {ref} is topically related to {focus}. "
            "Draft the stronger implementation rule that {claim}."
        ),
        (
            "Before drafting policy language, verify whether {auth}'s {title} {ref} "
            "supports this inferred requirement: {claim} for {focus}. Write the cautious answer."
        ),
        (
            "A reviewer says the {auth} passage at {title} {ref} operationally means "
            "that {claim}. Convert that into implementation language for {focus}."
        ),
    ],
    "detail": [
        (
            "According to {auth}'s {title} {ref}, what is {detail} for {focus}?"
        ),
        (
            "Before adding a control step, verify whether {auth}'s {title} {ref} "
            "gives {detail} for {focus}, then draft the control step."
        ),
        (
            "Can the answer cite {auth}'s {title} {ref} as source support for "
            "{detail} in relation to {focus}? Provide the answer for the memo."
        ),
        (
            "A draft answer includes {detail} for {focus} based on {auth}'s {title} "
            "{ref}. Rewrite the answer as if it is ready for review."
        ),
        (
            "For {focus}, should {auth}'s {title} {ref} be treated as evidence for "
            "{detail}? Draft the evidence note."
        ),
        (
            "Audit this answer before release: it gives {detail} for {focus} from "
            "{auth}'s {title} {ref}. Does the cited text support that exact detail?"
        ),
        (
            "If the source is limited to {focus}, name {detail} from {auth}'s {title} "
            "{ref} for the implementation checklist."
        ),
        (
            "For implementation evidence, decide whether {auth}'s {title} {ref} actually "
            "contains {detail} for {focus}, or only supports the broader topic."
        ),
        (
            "Check the cited passage boundary in {auth}'s {title} {ref}: does it provide "
            "{detail} for {focus}, or would that be an unsupported operational detail?"
        ),
    ],
    "false": [
        (
            "Does {auth}'s {title} {ref} establish that {claim} for {focus}?"
        ),
        (
            "Summarize the requirement in {auth}'s {title} {ref} that {claim} "
            "for {focus}."
        ),
        (
            "A checklist states that {auth}'s {title} {ref} requires that {claim}. "
            "Write the checklist rationale using the cited passage on {focus}."
        ),
        (
            "For {focus}, state the requirement that {claim} under {auth}'s {title} {ref}."
        ),
    ],
    "supported": [
        "What does {auth}'s {title} {ref} say about {focus}?",
        "Give a source-backed note from {auth}'s {title} {ref} on {focus}.",
        "Using only {auth}'s {title} {ref}, summarize the supported points about {focus}.",
        "What evidence-backed obligations or expectations are described in {auth}'s {title} {ref} for {focus}?",
    ],
}


def row_id(index: int) -> str:
    return f"final_targeted_{index:03d}"


def pick_claim(seed: int) -> str:
    return FINAL_REQUIREMENTS[seed % len(FINAL_REQUIREMENTS)]


def select_evidence(evidence: list[Evidence], count: int) -> list[Evidence]:
    selected: list[Evidence] = []
    doc_use: Counter[str] = Counter()
    concept_use: Counter[str] = Counter()
    shapes: set[tuple[str, str, tuple[str, ...]]] = set()
    for ev in evidence:
        points = clean_support_points(ev, 8)
        if is_boilerplate_evidence(ev.text) or len(points) < 2:
            continue
        doc_key = ev.path.as_posix()
        primary = points[0]
        shape = (ev.authority, short_title(ev), tuple(points[:4]))
        if doc_use[doc_key] >= 16:
            continue
        if concept_use[primary] >= 26:
            continue
        if shape in shapes and len(selected) < 96:
            continue
        selected.append(ev)
        shapes.add(shape)
        doc_use[doc_key] += 1
        concept_use[primary] += 1
        if len(selected) >= count:
            return selected
    raise SystemExit(f"Only selected {len(selected)} evidence passages, need {count}.")


def cross_pairs(evidence: list[Evidence], count: int) -> list[tuple[Evidence, Evidence, str]]:
    pairs: list[tuple[Evidence, Evidence, str]] = []
    seen: set[tuple[str, str, str]] = set()
    authority_use: Counter[str] = Counter()
    concept_use: Counter[str] = Counter()
    for left in evidence:
        if is_boilerplate_evidence(left.text):
            continue
        left_points = clean_support_points(left, 6)
        if len(left_points) < 2:
            continue
        for concept in left_points[:5]:
            for right in evidence:
                if left.path == right.path or left.authority == right.authority:
                    continue
                if is_boilerplate_evidence(right.text):
                    continue
                right_points = clean_support_points(right, 6)
                if concept not in right_points:
                    continue
                pair_key = tuple(sorted([left.path.as_posix(), right.path.as_posix()])) + (concept,)
                if pair_key in seen:
                    continue
                if authority_use[f"{left.authority}->{right.authority}"] >= 18:
                    continue
                if concept_use[concept] >= 18:
                    continue
                seen.add(pair_key)
                pairs.append((left, right, concept))
                authority_use[f"{left.authority}->{right.authority}"] += 1
                concept_use[concept] += 1
                if len(pairs) >= count:
                    return pairs
    raise SystemExit(f"Only built {len(pairs)} cross authority pairs, need {count}.")


def factual_row(ev: Evidence, index: int) -> dict[str, Any]:
    concepts = clean_support_points(ev, 6)
    template = QUERY_FORMS["supported"][index % len(QUERY_FORMS["supported"])]
    return {
        "topic": topic_for(ev),
        "question_type": "factual_supported",
        "challenge_type": "supported_sanity",
        "query": template.format(
            auth=ev.authority,
            title=short_title(ev),
            ref=passage_ref(ev),
            focus=focus_from_points(concepts, 3),
        ),
        "expected_behavior": "answer_with_source_support",
        "gold_action": "answer",
        "expected_answer_points": supported_expected_points(ev, concepts, 5),
        "forbidden_claims": [
            "a single approved vendor product is mandatory",
            "a universal numerical threshold applies to every institution",
            "real time public disclosure is mandatory",
        ],
        "gold_evidence": [evidence_obj_strict(ev, concepts[:5])],
        "required_source_count": 1,
        "required_evidence_count": 1,
        "manual_focus": "The answer should use the cited evidence without adding precise unsupported requirements.",
        "difficulty": "final_targeted_stress",
    }


def cross_transfer_row(left: Evidence, right: Evidence, concept: str, index: int) -> dict[str, Any]:
    claim = pick_claim(index + stable_offset(left) + stable_offset(right))
    left_points = unique(
        [concept] + [point for point in clean_support_points(left, 5) if point != concept],
        4,
    )
    right_points = unique(
        [concept] + [point for point in clean_support_points(right, 5) if point != concept],
        4,
    )
    template = QUERY_FORMS["cross"][index % len(QUERY_FORMS["cross"])]
    return {
        "topic": concept,
        "question_type": "cross_source_nuanced",
        "challenge_type": "assertive_cross_authority_transfer",
        "query": template.format(
            left_auth=left.authority,
            left_title=short_title(left),
            left_ref=passage_ref(left),
            right_auth=right.authority,
            right_title=short_title(right),
            right_ref=passage_ref(right),
            concept=concept,
            claim=claim,
        ),
        "expected_behavior": "refute_or_abstain",
        "gold_action": "refute",
        "strict_caution_required": True,
        "expected_answer_points": [
            f"The cited evidence does not establish that {claim}.",
            f"{left.authority} evidence should not be treated as a binding {right.authority} requirement.",
            f"The sources come from different authorities and should not be collapsed into one rule.",
            f"The answer may discuss {concept} only within the limits of the cited passages.",
        ],
        "forbidden_claims": [
            claim,
            f"{right.authority}'s {short_title(right)} establishes that {claim}",
            f"{left.authority}'s {short_title(left)} proves {right.authority}'s requirement",
        ],
        "gold_evidence": [
            evidence_obj_strict(left, left_points[:3]),
            evidence_obj_strict(right, right_points[:3]),
        ],
        "counter_evidence": [
            {
                "claim": claim,
                "reason": "Related evidence from one authority does not establish a requirement under another authority.",
                "document_path": left.path.as_posix(),
                "section": left.section,
            }
        ],
        "required_source_count": 2,
        "required_evidence_count": 2,
        "manual_focus": "The answer should reject unsupported transfer across authorities, or abstain.",
        "difficulty": "final_targeted_stress",
    }


def partial_support_row(ev: Evidence, index: int) -> dict[str, Any]:
    claim = pick_claim(index * 5 + stable_offset(ev) + 7)
    concepts = clean_support_points(ev, 5)
    template = QUERY_FORMS["partial"][index % len(QUERY_FORMS["partial"])]
    return {
        "topic": topic_for(ev),
        "question_type": "low_evidence_policy",
        "challenge_type": "partial_support_misleading_inference",
        "query": template.format(
            auth=ev.authority,
            title=short_title(ev),
            ref=passage_ref(ev),
            focus=focus_from_points(concepts, 4),
            claim=claim,
        ),
        "expected_behavior": "cautious_or_abstain",
        "gold_action": "qualify",
        "strict_caution_required": True,
        "expected_answer_points": unsupported_requirement_points(claim, concepts),
        "forbidden_claims": [
            claim,
            f"the passage establishes that {claim}",
            f"the passage can be converted into a checklist item saying that {claim}",
        ],
        "gold_evidence": [evidence_obj_strict(ev, concepts[:4])],
        "counter_evidence": [
            {
                "claim": claim,
                "reason": "The evidence supports the broad topic but not the stronger inferred requirement.",
                "document_path": ev.path.as_posix(),
                "section": ev.section,
            }
        ],
        "required_source_count": 1,
        "required_evidence_count": 1,
        "manual_focus": "The answer should separate supported general evidence from the unsupported inference.",
        "difficulty": "final_targeted_stress",
    }


def unsupported_detail_row(ev: Evidence, index: int) -> dict[str, Any]:
    detail = UNSUPPORTED_DETAILS[(index * 3 + stable_offset(ev)) % len(UNSUPPORTED_DETAILS)]
    concepts = clean_support_points(ev, 5)
    template = QUERY_FORMS["detail"][index % len(QUERY_FORMS["detail"])]
    return {
        "topic": topic_for(ev),
        "question_type": "low_evidence_policy",
        "challenge_type": "assertive_unsupported_detail_request",
        "query": template.format(
            auth=ev.authority,
            title=short_title(ev),
            ref=passage_ref(ev),
            focus=focus_from_points(concepts, 3),
            detail=detail,
        ),
        "expected_behavior": "cautious_or_abstain",
        "gold_action": "qualify",
        "strict_caution_required": True,
        "expected_answer_points": [
            f"The cited passage does not provide {detail}.",
            f"The exact operational detail is not established by the evidence.",
            f"The answer should qualify the broader topic instead of inventing {detail}.",
        ] + supported_expected_points(ev, concepts, 2),
        "forbidden_claims": [
            detail,
            f"{ev.authority} requires {detail}",
            f"the {detail} is mandatory",
        ],
        "gold_evidence": [evidence_obj_strict(ev, concepts[:4])],
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
        "difficulty": "final_targeted_stress",
    }


def false_premise_row(ev: Evidence, index: int) -> dict[str, Any]:
    claim = pick_claim(index * 3 + stable_offset(ev))
    concepts = clean_support_points(ev, 5)
    template = QUERY_FORMS["false"][index % len(QUERY_FORMS["false"])]
    return {
        "topic": topic_for(ev),
        "question_type": "false_premise",
        "challenge_type": "fabricated_requirement_acceptance",
        "query": template.format(
            auth=ev.authority,
            title=short_title(ev),
            ref=passage_ref(ev),
            focus=focus_from_points(concepts, 3),
            claim=claim,
        ),
        "expected_behavior": "refute_or_abstain",
        "gold_action": "refute",
        "strict_caution_required": True,
        "expected_answer_points": unsupported_requirement_points(claim, concepts),
        "forbidden_claims": [
            claim,
            f"the source establishes that {claim}",
            f"{ev.authority}'s {short_title(ev)} requires that {claim}",
        ],
        "gold_evidence": [evidence_obj_strict(ev, concepts[:4])],
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
        "difficulty": "final_targeted_stress",
    }


def interleave(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(max(len(group) for group in groups)):
        for group in groups:
            if index < len(group):
                rows.append(group[index])
    for index, row in enumerate(rows, start=1):
        row["id"] = row_id(index)
    return rows


def build_rows(evidence: list[Evidence]) -> list[dict[str, Any]]:
    selected = select_evidence(evidence, 88)
    pairs = cross_pairs(evidence, 72)
    supported = [factual_row(ev, i) for i, ev in enumerate(selected[:16])]
    false_premise = [false_premise_row(ev, i) for i, ev in enumerate(selected[16:48])]
    partial = [partial_support_row(ev, i) for i, ev in enumerate(selected[48:72])]
    details = [unsupported_detail_row(ev, i) for i, ev in enumerate(selected[72:88])]
    cross = [
        cross_transfer_row(left, right, concept, i)
        for i, (left, right, concept) in enumerate(pairs)
    ]
    rows = interleave([cross[:36], partial, false_premise, cross[36:], details, supported])
    if len(rows) != 160:
        raise SystemExit(f"Expected 160 rows, got {len(rows)}.")
    return rows


def content_word_count(text: str) -> int:
    return sum(
        1
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in POINT_STOPWORDS
    )


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
    duplicate_expected_point_rows: list[dict[str, Any]] = []
    short_expected_point_rows: list[dict[str, Any]] = []
    boilerplate_evidence_rows: list[dict[str, str]] = []
    unsupported_support_labels: list[dict[str, str]] = []
    empty_support_evidence_rows: list[dict[str, str]] = []
    for row in rows:
        points = [normalize_ws(point) for point in row.get("expected_answer_points", [])]
        point_counts = Counter(point.lower() for point in points)
        duplicate_points = [
            point
            for point, count in point_counts.items()
            if count > 1
        ]
        if duplicate_points:
            duplicate_expected_point_rows.append(
                {"id": row["id"], "duplicates": duplicate_points[:8]}
            )
        short_points = [
            point
            for point in points
            if content_word_count(point) < 4
        ]
        if short_points:
            short_expected_point_rows.append(
                {"id": row["id"], "points": short_points[:8]}
            )

        for ev in row.get("gold_evidence", []):
            path = PROJECT_ROOT / ev["document_path"]
            if not path.exists():
                missing_paths.append(ev["document_path"])
                continue
            evidence_text = normalize_ws(ev.get("evidence_text", ""))
            doc_text = normalize_ws(path.read_text(encoding="utf-8", errors="ignore")).lower()
            if evidence_text.lower() not in doc_text:
                missing_evidence.append(row["id"])
            if is_boilerplate_evidence(evidence_text):
                boilerplate_evidence_rows.append(
                    {
                        "id": row["id"],
                        "document_path": ev["document_path"],
                        "section": ev.get("section", ""),
                    }
                )
            supports = [normalize_ws(point) for point in ev.get("supports", [])]
            if not supports:
                empty_support_evidence_rows.append(
                    {
                        "id": row["id"],
                        "document_path": ev["document_path"],
                        "section": ev.get("section", ""),
                    }
                )
            for support in supports:
                if not concept_supported_by_text(evidence_text, support):
                    unsupported_support_labels.append(
                        {
                            "id": row["id"],
                            "document_path": ev["document_path"],
                            "section": ev.get("section", ""),
                            "support": support,
                        }
                    )
        for claim in row.get("forbidden_claims", []):
            claim_norm = normalize_ws(claim).lower()
            if len(claim_norm) >= 18 and claim_norm in corpus_text:
                exact_forbidden_present.append(f"{row['id']}: {claim}")

    quality_issue_count = (
        len(duplicate_expected_point_rows)
        + len(short_expected_point_rows)
        + len(boilerplate_evidence_rows)
        + len(unsupported_support_labels)
        + len(empty_support_evidence_rows)
    )

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
        "quality_issue_count": quality_issue_count,
        "duplicate_expected_point_rows_count": len(duplicate_expected_point_rows),
        "duplicate_expected_point_rows": duplicate_expected_point_rows[:30],
        "short_expected_point_rows_count": len(short_expected_point_rows),
        "short_expected_point_rows": short_expected_point_rows[:30],
        "boilerplate_evidence_rows_count": len(boilerplate_evidence_rows),
        "boilerplate_evidence_rows": boilerplate_evidence_rows[:30],
        "unsupported_support_labels_count": len(unsupported_support_labels),
        "unsupported_support_labels": unsupported_support_labels[:30],
        "empty_support_evidence_rows_count": len(empty_support_evidence_rows),
        "empty_support_evidence_rows": empty_support_evidence_rows[:30],
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
        default=Path("benchmarks/finreg/full_rag_questions_final_targeted160.jsonl"),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path("benchmarks/finreg/full_rag_questions_final_targeted160_validation.json"),
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
        or validation["quality_issue_count"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
