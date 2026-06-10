#!/usr/bin/env python3
"""Build an independent high-difficulty FinReg full-RAG benchmark.

The builder uses only local processed FinReg documents. It creates:

- benchmarks/finreg/full_rag_questions_hard_v2.jsonl
- benchmarks/finreg/full_rag_questions_hard_v2_validation.json
- benchmarks/finreg/full_rag_questions_hard_v2_review.csv

The output is intentionally evidence-heavy: every row carries supporting
passages, source metadata, and review guidance so the benchmark can be audited
before being used for final claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


QUESTION_COUNTS = {
    "factual_supported": 40,
    "false_premise": 40,
    "partial_support": 40,
    "cross_source_nuanced": 40,
}

EXPECTED_BEHAVIOR = {
    "factual_supported": "answer_with_source_support",
    "false_premise": "refute_or_abstain",
    "partial_support": "cautious_or_abstain",
    "cross_source_nuanced": "cautious_synthesis",
}

GOLD_ACTION = {
    "factual_supported": "answer",
    "false_premise": "refute",
    "partial_support": "qualify",
    "cross_source_nuanced": "qualify",
}

CHALLENGE_TYPES = [
    "single_source_multi_section",
    "cross_source_comparison",
    "false_attribution",
    "scope_confusion",
    "recommendation_as_requirement",
    "unsupported_numeric_precision",
    "unsupported_deadline",
    "partial_support",
    "exception_as_general_rule",
    "responsibility_misattribution",
    "conflicting_scope",
    "temporal_or_version_confusion",
]

FALSE_CHALLENGES = [
    "false_attribution",
    "scope_confusion",
    "recommendation_as_requirement",
    "unsupported_numeric_precision",
    "unsupported_deadline",
    "exception_as_general_rule",
    "responsibility_misattribution",
    "temporal_or_version_confusion",
]

PARTIAL_CHALLENGES = [
    "partial_support",
    "unsupported_numeric_precision",
    "unsupported_deadline",
    "scope_confusion",
    "responsibility_misattribution",
    "recommendation_as_requirement",
    "exception_as_general_rule",
]

CROSS_CHALLENGES = [
    "cross_source_comparison",
    "conflicting_scope",
    "single_source_multi_section",
    "scope_confusion",
]

BOILERPLATE_TERMS = {
    "share this page",
    "stay connected",
    "sign up",
    "privacy notice",
    "cookies notice",
    "email scam warning",
    "switchboard",
    "careers",
    "terms and conditions",
    "skip to main content",
    "press contacts",
    "follow @",
    "footer",
    "menu extranet",
    "our mission is to contribute",
    "related information",
    "legal information",
    "contact careers",
    "pdf full text",
    "this version",
    "topics:",
}

TOPIC_HINTS = [
    ("rdarr", "risk data aggregation and risk reporting"),
    ("risk data", "risk data aggregation and risk reporting"),
    ("model", "model risk management"),
    ("validation", "model validation"),
    ("climate", "climate-related financial risk management"),
    ("outsourcing", "outsourcing and third-party risk management"),
    ("third party", "third-party risk management"),
    ("ict", "ICT and security risk management"),
    ("srep", "SREP and supervisory stress testing"),
    ("stress", "stress testing governance"),
    ("liquidity", "liquidity and funding risk management"),
    ("governance", "internal governance and risk culture"),
    ("remuneration", "remuneration governance"),
    ("suitability", "suitability assessment"),
    ("branch", "branch and subsidiary supervision"),
    ("operational resilience", "operational resilience"),
]


@dataclass(frozen=True)
class Document:
    path: Path
    text: str
    norm_text: str
    title: str
    authority: str
    doc_id: str
    pages: list[dict[str, Any]]


@dataclass(frozen=True)
class Passage:
    doc: Document
    text: str
    sentences: tuple[str, ...]
    section: str
    page: int | None
    topic: str


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def authority_from_path(path: Path, metadata: dict[str, Any]) -> str:
    source_org = str(metadata.get("source_org") or "").strip()
    if source_org:
        return source_org
    parent = path.parent.name
    return {
        "bcbs": "BCBS",
        "eba": "EBA",
        "ecb": "ECB",
        "fed_occ": "Federal Reserve",
        "pra_boe": "PRA",
    }.get(parent, parent)


def read_documents(root: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(root.glob("**/*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        metadata_path = path.with_suffix(".metadata.json")
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pages_path = path.with_suffix(".pages.json")
        pages: list[dict[str, Any]] = []
        if pages_path.exists():
            loaded = json.loads(pages_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                pages = loaded
        docs.append(
            Document(
                path=path,
                text=text,
                norm_text=normalize_ws(text),
                title=str(metadata.get("title") or path.stem),
                authority=authority_from_path(path, metadata),
                doc_id=str(metadata.get("id") or path.stem),
                pages=pages,
            )
        )
    return docs


def split_sentences(text: str) -> list[str]:
    norm = normalize_ws(text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“])", norm)
    return [p.strip() for p in parts if p.strip()]


def infer_topic(title: str, text: str) -> str:
    title_low = title.lower()
    for token, topic in TOPIC_HINTS:
        if token in title_low:
            return topic
    haystack = f"{title} {text}".lower()
    for token, topic in TOPIC_HINTS:
        if token in haystack:
            return topic
    words = re.findall(r"[A-Za-z][A-Za-z-]{3,}", title)
    return " ".join(words[:6]).lower() or "financial regulation"


def find_page(doc: Document, passage: str) -> int | None:
    needle = normalize_ws(passage)[:140].lower()
    if not needle:
        return None
    for item in doc.pages:
        text = normalize_ws(str(item.get("text") or "")).lower()
        if needle in text:
            page = item.get("page")
            return int(page) if isinstance(page, int) else None
    if doc.pages:
        page = doc.pages[0].get("page")
        return int(page) if isinstance(page, int) else None
    return None


def find_section(doc: Document, passage: str) -> str:
    needle = normalize_ws(passage)[:80]
    raw_index = normalize_ws(doc.text[: doc.text.find(needle)]) if needle in doc.text else ""
    if raw_index:
        candidate_lines = raw_index.splitlines()[-20:]
    else:
        lines = doc.text.splitlines()
        candidate_lines = lines[:60]
    for line in reversed(candidate_lines):
        item = normalize_ws(line)
        if 5 <= len(item) <= 120 and not item.endswith("."):
            return item
    return doc.title


def is_useful_sentence(sentence: str) -> bool:
    low = sentence.lower()
    if any(term in low for term in BOILERPLATE_TERMS):
        return False
    if len(re.findall(r"[A-Za-z]{4,}", sentence)) < 7:
        return False
    regulatory_terms = [
        "risk",
        "management",
        "supervis",
        "govern",
        "validation",
        "model",
        "stress",
        "liquidity",
        "climate",
        "outsourcing",
        "third-party",
        "third party",
        "board",
        "firms",
        "banks",
        "data",
        "report",
        "control",
        "resilience",
    ]
    return any(term in low for term in regulatory_terms)


def is_bad_passage(passage: str) -> bool:
    low = passage.lower()
    if any(term in low for term in BOILERPLATE_TERMS):
        return True
    noisy_terms = ["http://", "https://", "@eba", "press@", "subscribe", "cookie", "sitemap"]
    if any(term in low for term in noisy_terms):
        return True
    if low.startswith("see, for example") or low.startswith("press release"):
        return True
    if len(re.findall(r"\b(page|footer|menu|contact|login)\b", low)) >= 3:
        return True
    return False


def extract_passages(docs: list[Document]) -> list[Passage]:
    passages: list[Passage] = []
    for doc in docs:
        sentences = split_sentences(doc.text)
        seen: set[str] = set()
        doc_passages: list[Passage] = []
        for i in range(0, max(0, len(sentences) - 2)):
            trio = sentences[i : i + 3]
            if sum(1 for sentence in trio if is_useful_sentence(sentence)) < 2:
                continue
            passage = normalize_ws(" ".join(trio))
            key = passage[:240].casefold()
            if key in seen or len(passage) < 220 or len(passage) > 900:
                continue
            if is_bad_passage(passage):
                continue
            seen.add(key)
            doc_passages.append(
                Passage(
                    doc=doc,
                    text=passage,
                    sentences=tuple(trio),
                    section=find_section(doc, passage),
                    page=find_page(doc, passage),
                    topic=infer_topic(doc.title, passage),
                )
            )
            if len(doc_passages) >= 18:
                break
        passages.extend(doc_passages)
    return passages


def point_from_sentence(sentence: str) -> str:
    sentence = normalize_ws(sentence)
    sentence = re.sub(r"^\d+(\.\d+)?\s+", "", sentence)
    if len(sentence) > 190:
        sentence = sentence[:187].rsplit(" ", 1)[0] + "..."
    return sentence[0].lower() + sentence[1:] if sentence[:1].isupper() else sentence


def support_points(passage: Passage, limit: int = 3) -> list[str]:
    points = [point_from_sentence(s) for s in passage.sentences if is_useful_sentence(s)]
    if len(points) < limit:
        points.extend(point_from_sentence(s) for s in passage.sentences if len(s.strip()) > 35)
    return unique(points, limit)


def unique(values: list[str], limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = normalize_ws(value)
        key = item.casefold()
        if item and key not in seen:
            out.append(item)
            seen.add(key)
        if limit and len(out) >= limit:
            break
    return out


def evidence_obj(passage: Passage, supports: list[str]) -> dict[str, Any]:
    return {
        "document_path": passage.doc.path.as_posix(),
        "document_name": passage.doc.title,
        "authority": passage.doc.authority,
        "section": passage.section,
        "page": passage.page,
        "evidence_text": passage.text,
        "supports": supports,
    }


def false_claim(topic: str, authority: str, challenge: str, other_authority: str) -> tuple[str, str]:
    if challenge == "false_attribution":
        claim = f"{other_authority} imposes the same {topic} obligation as {authority}"
        query = f"Does {other_authority} impose the same {topic} obligation that is usually associated with {authority}?"
    elif challenge == "unsupported_numeric_precision":
        claim = f"{authority} sets a fixed 5 percent quantitative threshold for {topic}"
        query = f"Does {authority} set a fixed 5 percent threshold for deciding whether {topic} controls are adequate?"
    elif challenge == "unsupported_deadline":
        claim = f"{authority} requires completion within five business days for {topic}"
        query = f"Does {authority} require firms to complete the {topic} process within five business days?"
    elif challenge == "recommendation_as_requirement":
        claim = f"{authority} converts the described {topic} practice into an automatic approval requirement"
        query = f"Does {authority} make the described {topic} practice an automatic precondition for supervisory approval?"
    elif challenge == "responsibility_misattribution":
        claim = f"{authority} transfers primary {topic} responsibility from the firm to the supervisor"
        query = f"Under {authority}'s approach, is primary responsibility for {topic} transferred from the firm to the supervisor?"
    elif challenge == "exception_as_general_rule":
        claim = f"{authority} treats a limited {topic} case as a rule for every institution"
        query = f"Does {authority} treat the limited {topic} case as a rule that applies to every institution?"
    elif challenge == "temporal_or_version_confusion":
        claim = f"{authority}'s current {topic} position replaces all earlier related expectations"
        query = f"Does {authority}'s current material on {topic} replace all earlier related supervisory expectations?"
    else:
        claim = f"{authority} applies the same {topic} rule to every firm regardless of scope or proportionality"
        query = f"Does {authority} apply the same {topic} rule to every firm regardless of scope, size, or proportionality?"
    return claim, query


def partial_unsupported_claim(topic: str, authority: str, challenge: str) -> tuple[str, str]:
    if challenge == "unsupported_numeric_precision":
        unsupported = f"{authority} requires a fixed 10 percent tolerance for {topic}"
    elif challenge == "unsupported_deadline":
        unsupported = f"{authority} requires board certification within 30 calendar days for {topic}"
    elif challenge == "responsibility_misattribution":
        unsupported = f"{authority} makes the external provider solely responsible for {topic}"
    elif challenge == "recommendation_as_requirement":
        unsupported = f"{authority} requires automatic supervisory approval before any {topic} change"
    elif challenge == "exception_as_general_rule":
        unsupported = f"{authority} applies the most restrictive {topic} treatment to all firms without exception"
    elif challenge == "scope_confusion":
        unsupported = f"{authority} applies the same {topic} expectation to every entity in every jurisdiction"
    else:
        unsupported = f"{authority} requires a single public reporting template for {topic}"
    query = (
        f"For {topic}, does {authority} both set out the supervisory expectation and require "
        f"the additional step that {unsupported.split(' requires ', 1)[-1] if ' requires ' in unsupported else unsupported}?"
    )
    return unsupported, query


class Builder:
    def __init__(self, passages: list[Passage], old_queries: list[str]) -> None:
        self.passages = sorted(passages, key=lambda p: (p.doc.authority, p.doc.doc_id, p.text))
        self.by_doc: dict[str, list[Passage]] = defaultdict(list)
        self.by_topic: dict[str, list[Passage]] = defaultdict(list)
        for passage in self.passages:
            self.by_doc[passage.doc.doc_id].append(passage)
            self.by_topic[passage.topic].append(passage)
        self.old_queries = old_queries
        self.doc_use: Counter[str] = Counter()
        self.topic_use: Counter[str] = Counter()
        self.evidence_use: Counter[str] = Counter()
        self.challenge_use: Counter[str] = Counter()
        self.next_index: dict[str, int] = defaultdict(int)
        self.rows: list[dict[str, Any]] = []

    def choose_passage(self, topic: str | None = None, exclude_docs: set[str] | None = None) -> Passage:
        exclude_docs = exclude_docs or set()
        primary_pool = self.by_topic.get(topic or "", []) if topic else self.passages
        pools = [primary_pool, self.passages] if len(primary_pool) >= 3 else [self.passages]
        for pool_number, pool in enumerate(pools):
            if not pool:
                continue
            for _ in range(len(pool) * 3):
                key = topic or "__all__"
                if pool_number:
                    key = "__all_fallback__"
                idx = self.next_index[key] % len(pool)
                self.next_index[key] += 1
                passage = pool[idx]
                doc_id = passage.doc.doc_id
                ev_key = f"{doc_id}:{passage.text[:160]}"
                if doc_id in exclude_docs:
                    continue
                if self.doc_use[doc_id] >= 10:
                    continue
                if self.evidence_use[ev_key] >= 2:
                    continue
                return passage
        raise RuntimeError("no eligible passage left")

    def reserve(self, passages: list[Passage]) -> None:
        for passage in passages:
            self.doc_use[passage.doc.doc_id] += 1
            self.topic_use[passage.topic] += 1
            self.evidence_use[f"{passage.doc.doc_id}:{passage.text[:160]}"] += 1

    def next_challenge(self, allowed: list[str]) -> str:
        ordered = sorted(allowed, key=lambda name: (self.challenge_use[name], allowed.index(name)))
        challenge = ordered[0]
        self.challenge_use[challenge] += 1
        return challenge

    def row_id(self) -> str:
        return f"hard_eval_{len(self.rows) + 1:03d}"

    def similarity_to_old(self, query: str) -> float:
        if not self.old_queries:
            return 0.0
        return max(SequenceMatcher(None, query.lower(), old.lower()).ratio() for old in self.old_queries)

    def add_row(self, row: dict[str, Any], passages: list[Passage]) -> None:
        row["expected_answer_points"] = unique(row["expected_answer_points"], 7)
        row["forbidden_claims"] = unique(row["forbidden_claims"], 5)
        row["difficulty"] = "hard"
        row["_old_query_similarity"] = round(self.similarity_to_old(str(row["query"])), 4)
        self.reserve(passages)
        self.rows.append(row)

    def build_factual(self) -> None:
        while sum(1 for r in self.rows if r["question_type"] == "factual_supported") < 40:
            p1 = self.choose_passage()
            p2 = self.choose_passage(topic=p1.topic, exclude_docs=set())
            if p2.doc.doc_id != p1.doc.doc_id and self.doc_use[p1.doc.doc_id] < 9:
                p2 = p1
            points = unique(support_points(p1, 3) + support_points(p2, 2), 5)
            query = (
                f"How does {p1.doc.authority}'s material connect {p1.topic} with the related "
                "governance, control, or supervisory expectations?"
            )
            row = {
                "id": self.row_id(),
                "topic": p1.topic,
                "question_type": "factual_supported",
                "challenge_type": self.next_challenge(
                    [
                        "single_source_multi_section",
                        "scope_confusion",
                        "conflicting_scope",
                        "temporal_or_version_confusion",
                        "responsibility_misattribution",
                        "recommendation_as_requirement",
                    ]
                ),
                "query": query,
                "expected_behavior": EXPECTED_BEHAVIOR["factual_supported"],
                "gold_action": GOLD_ACTION["factual_supported"],
                "expected_answer_points": points,
                "forbidden_claims": [
                    f"{p1.doc.authority} requires a fixed five business day deadline for {p1.topic}",
                    f"{p1.doc.authority} replaces firm governance with automatic supervisory approval for {p1.topic}",
                ],
                "gold_evidence": [evidence_obj(p1, points[:3])],
                "counter_evidence": [],
                "required_source_count": 1,
                "required_evidence_count": 1,
                "manual_focus": "Check whether the answer covers the supported supervisory expectation without adding precise obligations not in the source.",
            }
            self.add_row(row, [p1])

    def build_false(self) -> None:
        authorities = sorted({p.doc.authority for p in self.passages})
        while sum(1 for r in self.rows if r["question_type"] == "false_premise") < 40:
            p = self.choose_passage()
            challenge = self.next_challenge(FALSE_CHALLENGES)
            other = next(a for a in authorities if a != p.doc.authority)
            claim, query = false_claim(p.topic, p.doc.authority, challenge, other)
            points = unique(
                [f"the source supports {p.topic} only within {p.doc.authority}'s stated scope"]
                + support_points(p, 4),
                5,
            )
            row = {
                "id": self.row_id(),
                "topic": p.topic,
                "question_type": "false_premise",
                "challenge_type": challenge,
                "query": query,
                "expected_behavior": EXPECTED_BEHAVIOR["false_premise"],
                "gold_action": GOLD_ACTION["false_premise"],
                "expected_answer_points": points,
                "forbidden_claims": [
                    claim,
                    f"the source establishes the precise {p.topic} obligation stated in the question",
                    f"all firms must follow the same {p.topic} requirement without qualification",
                ],
                "gold_evidence": [evidence_obj(p, points[:3])],
                "counter_evidence": [
                    {
                        "claim": claim,
                        "reason": "The cited local source supports narrower supervisory expectations and does not establish the precise premise in the question.",
                        "document_path": p.doc.path.as_posix(),
                        "section": p.section,
                    }
                ],
                "required_source_count": 1,
                "required_evidence_count": 1,
                "manual_focus": "Verify that the answer rejects the plausible but unsupported premise rather than accepting it because nearby concepts are present.",
            }
            self.add_row(row, [p])

    def build_partial(self) -> None:
        while sum(1 for r in self.rows if r["question_type"] == "partial_support") < 40:
            p = self.choose_passage()
            challenge = self.next_challenge(PARTIAL_CHALLENGES)
            unsupported, query = partial_unsupported_claim(p.topic, p.doc.authority, challenge)
            supported_points = support_points(p, 4)
            points = unique(supported_points + [f"the added requirement is not established by the cited {p.doc.authority} material"], 6)
            row = {
                "id": self.row_id(),
                "topic": p.topic,
                "question_type": "partial_support",
                "challenge_type": challenge,
                "query": query,
                "expected_behavior": EXPECTED_BEHAVIOR["partial_support"],
                "gold_action": GOLD_ACTION["partial_support"],
                "expected_answer_points": points,
                "forbidden_claims": [
                    unsupported,
                    f"the supported {p.topic} expectation proves every additional operational detail in the question",
                    f"{p.doc.authority} requires a single mandatory template for {p.topic}",
                ],
                "gold_evidence": [evidence_obj(p, points[:3])],
                "counter_evidence": [
                    {
                        "claim": unsupported,
                        "reason": "The source supports the general supervisory topic but not the added precise requirement.",
                        "document_path": p.doc.path.as_posix(),
                        "section": p.section,
                    }
                ],
                "required_source_count": 1,
                "required_evidence_count": 1,
                "manual_focus": "Check that the answer separates the supported regulatory expectation from the unsupported added detail.",
            }
            self.add_row(row, [p])

    def build_cross(self) -> None:
        topics = sorted(self.by_topic, key=lambda t: -len(self.by_topic[t]))
        topic_i = 0
        while sum(1 for r in self.rows if r["question_type"] == "cross_source_nuanced") < 40:
            topic = topics[topic_i % len(topics)]
            topic_i += 1
            p1 = self.choose_passage(topic=topic)
            candidates = [
                p
                for p in self.by_topic.get(topic, [])
                if p.doc.doc_id != p1.doc.doc_id
                and p.doc.authority != p1.doc.authority
                and self.doc_use[p.doc.doc_id] < 10
            ]
            if not candidates:
                candidates = [
                    p
                    for p in self.passages
                    if p.doc.doc_id != p1.doc.doc_id
                    and p.doc.authority != p1.doc.authority
                    and self.doc_use[p.doc.doc_id] < 10
                ]
            p2 = candidates[self.next_index["__cross__"] % len(candidates)]
            self.next_index["__cross__"] += 1
            challenge = self.next_challenge(CROSS_CHALLENGES)
            points = unique(
                support_points(p1, 2)
                + support_points(p2, 2)
                + [f"the two sources should be synthesized without treating their scope as identical"],
                7,
            )
            query = (
                f"How should {p1.doc.authority} and {p2.doc.authority} be compared on {topic}, "
                "especially where their scope or supervisory emphasis is not identical?"
            )
            row = {
                "id": self.row_id(),
                "topic": topic,
                "question_type": "cross_source_nuanced",
                "challenge_type": challenge,
                "query": query,
                "expected_behavior": EXPECTED_BEHAVIOR["cross_source_nuanced"],
                "gold_action": GOLD_ACTION["cross_source_nuanced"],
                "expected_answer_points": points,
                "forbidden_claims": [
                    f"{p1.doc.authority} and {p2.doc.authority} impose identical {topic} requirements",
                    f"one source fully replaces the other source's {topic} scope",
                    f"all supervisors use the same operational rule for {topic}",
                ],
                "gold_evidence": [
                    evidence_obj(p1, points[:2]),
                    evidence_obj(p2, points[2:4]),
                ],
                "counter_evidence": [
                    {
                        "claim": f"{p1.doc.authority} and {p2.doc.authority} impose identical {topic} requirements",
                        "reason": "The sources address related themes but come from different authorities and do not state identical scope or obligations.",
                        "document_path": p1.doc.path.as_posix(),
                        "section": p1.section,
                    }
                ],
                "required_source_count": 2,
                "required_evidence_count": 2,
                "manual_focus": "Check whether the answer preserves both sources' scope and avoids flattening related but distinct supervisory expectations.",
            }
            self.add_row(row, [p1, p2])

    def build(self) -> list[dict[str, Any]]:
        self.build_cross()
        self.build_factual()
        self.build_false()
        self.build_partial()
        for index, row in enumerate(self.rows, start=1):
            row["id"] = f"hard_eval_{index:03d}"
        return self.rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False, separators=(",", ":")) + "\n")


def validate(rows: list[dict[str, Any]], old_queries: list[str]) -> dict[str, Any]:
    category_counts = Counter(row["question_type"] for row in rows)
    challenge_counts = Counter(row["challenge_type"] for row in rows)
    authority_counts: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()
    missing_paths: list[str] = []
    missing_evidence: list[str] = []
    too_few_points: list[str] = []
    too_few_forbidden: list[str] = []
    leakage: list[str] = []
    old_similar: list[dict[str, Any]] = []
    gold_action_mismatch: list[str] = []
    evidence_counter = 0
    counter_rows = 0

    expected_actions = {
        "answer_with_source_support": "answer",
        "refute_or_abstain": "refute",
        "cautious_or_abstain": "qualify",
        "cautious_synthesis": "qualify",
    }

    for row in rows:
        if len(row.get("expected_answer_points", [])) < 3:
            too_few_points.append(row["id"])
        if len(row.get("forbidden_claims", [])) < 2:
            too_few_forbidden.append(row["id"])
        if expected_actions.get(row.get("expected_behavior")) != row.get("gold_action"):
            gold_action_mismatch.append(row["id"])
        if row.get("counter_evidence"):
            counter_rows += 1
        qlow = str(row.get("query", "")).lower()
        for point in row.get("expected_answer_points", []):
            if len(point) > 35 and point.lower() in qlow:
                leakage.append(row["id"])
                break
        sims = [SequenceMatcher(None, qlow, str(old).lower()).ratio() for old in old_queries]
        if sims and max(sims) >= 0.72:
            old_similar.append({"id": row["id"], "similarity": round(max(sims), 4)})
        for ev in row.get("gold_evidence", []):
            evidence_counter += 1
            authority_counts[str(ev.get("authority"))] += 1
            path = Path(str(ev.get("document_path")))
            document_counts[path.as_posix()] += 1
            if not path.exists():
                missing_paths.append(path.as_posix())
                continue
            doc_norm = normalize_ws(path.read_text(encoding="utf-8", errors="ignore")).lower()
            ev_text = normalize_ws(str(ev.get("evidence_text") or "")).lower()
            if ev_text and ev_text not in doc_norm:
                missing_evidence.append(row["id"])

    ids = [row["id"] for row in rows]
    validation = {
        "total_questions": len(rows),
        "unique_ids": len(ids) == len(set(ids)),
        "category_distribution": dict(sorted(category_counts.items())),
        "challenge_type_distribution": dict(sorted(challenge_counts.items())),
        "authority_distribution": dict(sorted(authority_counts.items())),
        "document_distribution": dict(sorted(document_counts.items())),
        "single_source_questions": sum(1 for row in rows if int(row.get("required_source_count", 0)) == 1),
        "multi_source_questions": sum(1 for row in rows if int(row.get("required_source_count", 0)) >= 2),
        "counter_evidence_questions": counter_rows,
        "gold_evidence_count": evidence_counter,
        "max_challenge_type_count": max(challenge_counts.values()) if challenge_counts else 0,
        "max_document_question_count": max(document_counts.values()) if document_counts else 0,
        "missing_document_paths": sorted(set(missing_paths)),
        "missing_evidence_text_rows": sorted(set(missing_evidence)),
        "too_few_expected_answer_points": too_few_points,
        "too_few_forbidden_claims": too_few_forbidden,
        "possible_query_leakage_rows": sorted(set(leakage)),
        "old_benchmark_high_similarity_rows": old_similar,
        "gold_action_mismatch_rows": gold_action_mismatch,
        "checks": {
            "total_is_160": len(rows) == 160,
            "each_category_is_40": all(category_counts.get(k) == v for k, v in QUESTION_COUNTS.items()),
            "allowed_expected_behaviors": all(row.get("expected_behavior") in set(EXPECTED_BEHAVIOR.values()) for row in rows),
            "gold_evidence_present": all(row.get("gold_evidence") for row in rows),
            "false_and_partial_have_counter_evidence": all(
                row.get("counter_evidence")
                for row in rows
                if row.get("question_type") in {"false_premise", "partial_support"}
            ),
            "document_paths_exist": not missing_paths,
            "evidence_text_matches_documents": not missing_evidence,
            "min_three_answer_points": not too_few_points,
            "min_two_forbidden_claims": not too_few_forbidden,
            "challenge_type_under_15_percent": (max(challenge_counts.values()) if challenge_counts else 0) <= 24,
            "gold_action_matches_expected_behavior": not gold_action_mismatch,
        },
    }
    return validation


def write_validation(path: Path, validation: dict[str, Any]) -> None:
    path.write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "question_type",
        "challenge_type",
        "topic",
        "query",
        "expected_behavior",
        "gold_action",
        "required_source_count",
        "required_evidence_count",
        "authorities",
        "documents",
        "expected_answer_points",
        "forbidden_claims",
        "counter_claims",
        "manual_focus",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "question_type": row["question_type"],
                    "challenge_type": row["challenge_type"],
                    "topic": row["topic"],
                    "query": row["query"],
                    "expected_behavior": row["expected_behavior"],
                    "gold_action": row["gold_action"],
                    "required_source_count": row["required_source_count"],
                    "required_evidence_count": row["required_evidence_count"],
                    "authorities": "; ".join(ev["authority"] for ev in row["gold_evidence"]),
                    "documents": "; ".join(ev["document_path"] for ev in row["gold_evidence"]),
                    "expected_answer_points": " | ".join(row["expected_answer_points"]),
                    "forbidden_claims": " | ".join(row["forbidden_claims"]),
                    "counter_claims": " | ".join(c["claim"] for c in row.get("counter_evidence", [])),
                    "manual_focus": row["manual_focus"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/processed/finreg"))
    parser.add_argument("--old", type=Path, action="append", default=[
        Path("benchmarks/finreg/full_rag_questions.jsonl"),
        Path("benchmarks/finreg/full_rag_questions_hard.jsonl"),
    ])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/finreg/full_rag_questions_hard_v2.jsonl"))
    parser.add_argument("--validation-output", type=Path, default=Path("benchmarks/finreg/full_rag_questions_hard_v2_validation.json"))
    parser.add_argument("--review-output", type=Path, default=Path("benchmarks/finreg/full_rag_questions_hard_v2_review.csv"))
    args = parser.parse_args()

    old_queries: list[str] = []
    for path in args.old:
        old_queries.extend(str(row.get("query") or "") for row in read_jsonl(path))

    docs = read_documents(args.source_root)
    passages = extract_passages(docs)
    if len(passages) < 160:
        raise SystemExit(f"not enough candidate passages: {len(passages)}")

    builder = Builder(passages, old_queries)
    rows = builder.build()
    validation = validate(rows, old_queries)

    write_jsonl(args.output, rows)
    write_validation(args.validation_output, validation)
    write_review_csv(args.review_output, rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote validation to {args.validation_output}")
    print(f"Wrote review CSV to {args.review_output}")
    print(json.dumps(validation["checks"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
