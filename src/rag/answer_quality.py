"""Lightweight answer quality checks for RAG outputs.

These checks are deliberately deterministic. They do not replace manual review
or learned grading, but they give the pipeline a cheap signal for the failure
mode the answer-include detector does not cover: supported but incomplete
answers.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "answer", "before",
    "being", "between", "both", "could", "does", "from", "have", "into",
    "main", "more", "must", "only", "other", "over", "risk", "should",
    "than", "that", "their", "there", "these", "this", "through", "under",
    "what", "when", "where", "which", "while", "with", "within", "would",
}

BROAD_QUESTION_MARKERS = (
    "what",
    "how",
    "which",
    "main",
    "elements",
    "controls",
    "expect",
    "expected",
    "manage",
    "management",
    "framework",
    "responsibility",
    "responsibilities",
    "connect",
    "fit",
)

SPECIFIC_UNSUPPORTED_MARKERS = (
    "not establish",
    "not established",
    "does not establish",
    "not specify",
    "not specified",
    "does not specify",
    "no evidence",
    "not stated",
    "not explicitly",
    "unsupported",
    "missing evidence",
)

SPECIFIC_UNSUPPORTED_SUBJECT_MARKERS = (
    "deadline",
    "threshold",
    "template",
    "portal",
    "approval",
    "exact",
    "precise",
    "number",
    "percentage",
    "price",
    "certification",
)

REFUTATION_MARKERS = (
    "no",
    "not",
    "does not",
    "do not",
    "cannot",
    "no evidence",
    "not stated",
    "not established",
    "not explicitly",
    "not supported",
)

CONCEPT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("risk data aggregation", (r"\brisk data aggregation\b", r"\brda\b", r"\brdarr\b")),
    ("internal risk reporting", (r"\binternal risk report", r"\brisk reporting\b")),
    ("accurate and timely data", (r"\baccurate\b", r"\btimely\b", r"\bcomprehensive\b")),
    ("stress or crisis situations", (r"\bstress\b", r"\bcrisis\b")),
    ("governance", (r"\bgovernance\b", r"\bmanagement body\b", r"\bboard\b")),
    ("architecture", (r"\barchitecture\b", r"\bit infrastructure\b", r"\bsystems?\b")),
    ("controls", (r"\bcontrols?\b", r"\binternal control", r"\bcontrol function\b")),
    ("model development", (r"\bmodel development\b", r"\bdevelopment of models?\b")),
    ("model implementation and use", (r"\bmodel implementation\b", r"\bmodel use\b", r"\buse of models?\b")),
    ("validation", (r"\bvalidation\b", r"\bindependent validation\b")),
    ("limitations and use constraints", (r"\blimitations?\b", r"\buse constraints?\b")),
    ("monitoring", (r"\bmonitoring\b", r"\bmonitor\b", r"\bongoing basis\b")),
    ("model changes", (r"\bmodel changes?\b", r"\bchange management\b")),
    ("resilience", (r"\bresilience\b", r"\bresilient\b")),
    ("capital adequacy", (r"\bcapital adequacy\b", r"\bcapital requirement", r"\bcapital plan\b")),
    ("adverse scenarios", (r"\badverse scenarios?\b", r"\bscenario analysis\b")),
    ("risk concentrations", (r"\brisk concentrations?\b", r"\bconcentration risk\b")),
    ("SREP", (r"\bsrep\b", r"\bsupervisory review\b")),
    ("ICAAP", (r"\bicaap\b",)),
    ("ILAAP", (r"\bilaap\b",)),
    ("data lineage", (r"\bdata lineage\b", r"\blineage\b")),
    ("traceability", (r"\btraceability\b", r"\btraceable\b", r"\btrace\b")),
    ("auditability", (r"\bauditability\b", r"\baudit trail\b", r"\baudit traceability\b")),
    ("source systems", (r"\bsource systems?\b",)),
    ("remediation", (r"\bremediation\b", r"\bremediate\b", r"\bremedial\b")),
    ("temporary manual controls", (r"\btemporary\b", r"\bmanual workaround", r"\bmanual controls?\b")),
    ("transition", (r"\btransition\b", r"\btransitional\b")),
    ("ownership", (r"\bownership\b", r"\bowner\b", r"\baccountability\b")),
    ("climate risk", (r"\bclimate risk\b", r"\bclimate-related\b")),
    ("ESG risk", (r"\besg\b", r"\benvironmental, social and governance\b")),
    ("identify risks", (r"\bidentif(?:y|ies|ied|ying|ication)\b",)),
    ("measure risks", (r"\bmeasur(?:e|es|ed|ing|ement)\b",)),
    ("manage risks", (r"\bmanag(?:e|es|ed|ing|ement)\b",)),
    ("monitor risks", (r"\bmonitor\b", r"\bmonitoring\b", r"\bongoing basis\b")),
    ("transition risk", (r"\btransition risk\b",)),
    ("materiality", (r"\bmateriality\b", r"\bmaterial\b")),
    ("capital planning", (r"\bcapital planning\b", r"\bcapital plan\b")),
    ("time horizons", (r"\btime horizons?\b", r"\bshort[- ]term\b", r"\blong[- ]term\b")),
    ("retained responsibility", (r"\bretains? responsibility\b", r"\bremain responsible\b")),
    ("outsourcing governance", (r"\boutsourcing\b", r"\bthird[- ]party\b", r"\bservice provider\b")),
    ("risk assessment", (r"\brisk assessment\b", r"\bassess risks?\b")),
    ("exit planning", (r"\bexit planning\b", r"\bexit plan\b", r"\bexit strategy\b")),
    ("cloud configuration", (r"\bcloud\b", r"\bconfiguration\b", r"\bconfigure\b")),
    ("security", (r"\bsecurity\b", r"\binformation security\b")),
    ("ICT risk management", (r"\bict\b", r"\bict risk\b")),
    ("protect", (r"\bprotect(?:s|ed|ing|ion)?\b", r"\bprevent(?:s|ed|ing|ion)?\b", r"\bsafeguards?\b")),
    ("detect", (r"\bdetect(?:s|ed|ing|ion)?\b", r"\bidentify potential vulnerabilities\b", r"\bmonitor(?:s|ed|ing)?\b")),
    ("respond", (r"\brespond(?:s|ed|ing)?\b", r"\bresponse\b", r"\bincident response\b", r"\bproblem management\b")),
    ("recover", (r"\brecover(?:s|ed|ing|y)?\b", r"\bbackup\b", r"\bbackups\b", r"\bbusiness continuity\b")),
    ("incident management", (r"\bincident management\b", r"\bincident response\b", r"\bproblem management\b")),
    ("business continuity", (r"\bbusiness continuity\b", r"\bcontinuity\b")),
    ("access control", (r"\baccess control\b", r"\baccess rights?\b")),
    ("vulnerability and testing", (r"\bvulnerabilit", r"\btest(?:s|ed|ing)?\b", r"\bpenetration\b", r"\bassess(?:ed|ment|ing)?\b")),
    ("operational resilience", (r"\boperational resilience\b",)),
    ("important business services", (r"\bimportant business services?\b",)),
    ("impact tolerance", (r"\bimpact tolerance\b", r"\bimpact tolerances\b")),
    ("maximum tolerable disruption", (r"\bmaximum tolerable disruption\b", r"\bmaximum tolerable level\b", r"\btolerable disruption\b")),
    ("risk culture", (r"\brisk culture\b",)),
    ("staff training", (r"\bstaff training\b", r"\btraining\b")),
    ("board oversight", (r"\bboard oversight\b", r"\bboard\b", r"\bmanagement body\b")),
    ("remuneration", (r"\bremuneration\b", r"\bincentives?\b")),
    ("risk profile", (r"\brisk profile\b", r"\brisk appetite\b")),
    (
        "no exact requirement",
        (
            r"\bno exact\b",
            r"\bnot exact\b",
            r"\bnot an exact\b",
            r"\bnot specified\b",
            r"\bnot established\b",
            r"\bdoes not establish\b",
            r"\bdoes not specify\b",
            r"\bprecise\b",
            r"\bexact\b",
        ),
    ),
    ("avoid overclaiming", (r"\bavoid overclaim", r"\bdo not invent\b", r"\bnot established\b", r"\bprecise\b")),
    ("evidence limits", (r"\bevidence limit", r"\bincomplete evidence\b", r"\bevidence is incomplete\b", r"\bnot enough evidence\b")),
]


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _pattern_hit(text: str, patterns: Iterable[str]) -> bool:
    normalized = _normalize(text)
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns)


def split_claims(answer: str, max_claims: int = 6) -> list[str]:
    """Split a short answer into checkable claim-like sentences."""
    cleaned = re.sub(r"\s+", " ", answer or "").strip()
    if not cleaned:
        return []

    raw_parts: list[str] = []
    for sentence in SENTENCE_RE.split(cleaned):
        sentence = sentence.strip(" -;\t")
        if not sentence:
            continue
        subparts = re.split(r";\s+|\s+\band\b\s+(?=[A-Z]?[a-z]+\s)", sentence)
        raw_parts.extend(part.strip(" -;\t") for part in subparts if part.strip())

    claims: list[str] = []
    for part in raw_parts:
        tokens = TOKEN_RE.findall(part.lower())
        if len(tokens) < 5:
            continue
        if any(marker in part.lower() for marker in ("i don't know", "not enough reliable evidence")):
            continue
        if not part.endswith((".", "!", "?")):
            part += "."
        claims.append(part)
        if len(claims) >= max_claims:
            break
    return claims


def infer_required_concepts(
    query: str,
    contexts: list[str],
    *,
    max_concepts: int = 7,
) -> list[str]:
    """Infer reportable concepts that a complete answer should try to cover."""
    query_norm = _normalize(query)
    evidence_norm = _normalize("\n\n".join(contexts[:4]))
    broad_question = any(marker in query_norm for marker in BROAD_QUESTION_MARKERS)
    specific_unsupported = (
        any(marker in query_norm for marker in SPECIFIC_UNSUPPORTED_MARKERS)
        and any(marker in query_norm for marker in SPECIFIC_UNSUPPORTED_SUBJECT_MARKERS)
    )

    scored: list[tuple[int, int, str]] = []
    if specific_unsupported:
        forced = {"no exact requirement", "avoid overclaiming", "evidence limits"}
    else:
        forced = set()
    if "manual workaround" in query_norm or "manual workarounds" in query_norm:
        forced.update({
            "temporary manual controls",
            "transition",
            "controls",
            "ownership",
            "auditability",
            "remediation",
            "data lineage",
        })
    if "climate" in query_norm or "esg" in query_norm or "environmental" in query_norm:
        forced.update({
            "identify risks",
            "measure risks",
            "manage risks",
            "monitor risks",
            "governance",
            "transition risk",
            "climate risk",
            "ESG risk",
        })
    if "ict" in query_norm or "security risk management" in query_norm:
        forced.update({
            "governance",
            "identify risks",
            "protect",
            "detect",
            "respond",
            "recover",
            "vulnerability and testing",
            "incident management",
        })
    if "impact tolerance" in query_norm or "operational resilience" in query_norm:
        forced.update({
            "important business services",
            "maximum tolerable disruption",
            "impact tolerance",
            "operational resilience",
        })
    if "risk culture" in query_norm or "remuneration" in query_norm or "staff training" in query_norm:
        forced.update({
            "risk culture",
            "staff training",
            "board oversight",
            "risk profile",
            "remuneration",
            "governance",
        })

    for idx, (concept, patterns) in enumerate(CONCEPT_PATTERNS):
        query_hit = _pattern_hit(query_norm, patterns)
        evidence_hit = _pattern_hit(evidence_norm, patterns)
        if concept in forced:
            scored.append((-6, idx, concept))
            continue
        if not query_hit and not (broad_question and evidence_hit):
            continue
        score = 3 if query_hit else 1
        if evidence_hit:
            score += 1
        if specific_unsupported and concept in {"no exact requirement", "avoid overclaiming", "evidence limits"}:
            score += 3
        scored.append((-score, idx, concept))

    scored.sort()
    concepts: list[str] = []
    for _, _, concept in scored:
        if concept not in concepts:
            concepts.append(concept)
        if len(concepts) >= max_concepts:
            break
    return concepts


def audit_answer_quality(
    query: str,
    answer: str,
    contexts: list[str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic coverage/completeness signals for a generated answer."""
    cfg = config or {}
    concepts = infer_required_concepts(
        query,
        contexts,
        max_concepts=int(cfg.get("max_required_concepts", 7)),
    )
    answer_norm = _normalize(answer)
    hit_concepts: list[str] = []
    missing_concepts: list[str] = []
    concept_lookup = dict(CONCEPT_PATTERNS)

    for concept in concepts:
        patterns = concept_lookup.get(concept, (re.escape(concept),))
        if _pattern_hit(answer_norm, patterns):
            hit_concepts.append(concept)
        else:
            missing_concepts.append(concept)

    if concepts:
        concept_coverage = len(hit_concepts) / len(concepts)
    else:
        concept_coverage = 1.0

    claims = split_claims(answer, max_claims=int(cfg.get("max_claims", 6)))
    query_norm = _normalize(query)
    answer_has_refutation = any(marker in answer_norm for marker in REFUTATION_MARKERS)
    asks_specific_unsupported = (
        any(marker in query_norm for marker in SPECIFIC_UNSUPPORTED_MARKERS)
        and any(marker in query_norm for marker in SPECIFIC_UNSUPPORTED_SUBJECT_MARKERS)
    )

    return {
        "required_concepts": concepts,
        "hit_concepts": hit_concepts,
        "missing_concepts": missing_concepts,
        "concept_coverage": float(concept_coverage),
        "answer_completeness_score": float(concept_coverage),
        "answer_completeness_risk": float(1.0 - concept_coverage),
        "claim_count": len(claims),
        "claims": claims,
        "answer_has_refutation": bool(answer_has_refutation),
        "asks_specific_unsupported": bool(asks_specific_unsupported),
    }


def build_quality_feedback(audit: dict[str, Any], *, max_missing: int = 5) -> str:
    """Build a compact instruction for one rewrite attempt."""
    missing = [str(item) for item in audit.get("missing_concepts") or []]
    if not missing:
        return ""

    selected = ", ".join(missing[:max_missing])
    return (
        "The previous answer was too narrow. Rewrite the answer using only the "
        "provided context. If supported by the context, explicitly address these "
        f"missing concepts: {selected}. If any item is not established, say that "
        "it is not established instead of inventing details."
    )
