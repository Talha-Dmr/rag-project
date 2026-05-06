#!/usr/bin/env python3
"""Build a larger corpus-derived FinReg NLI candidate pool for review.

This script intentionally writes candidate examples, not trusted gold labels.
Rows are marked with review_status=pending so they can be sampled/reviewed before
training. The goal is to create diverse FinReg-style premise/hypothesis pairs
without loading an LLM.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

LABELS = ("entailment", "neutral", "contradiction")

THEME_KEYWORDS = {
    "rdarr": ["data aggregation", "risk reporting", "data lineage", "ad-hoc reports", "bcbs 239"],
    "governance": ["governance", "management body", "board", "oversight", "risk culture"],
    "model_risk": ["model risk", "validation", "effective challenge", "internal models"],
    "climate": ["climate", "environmental", "physical risks", "transition risks"],
    "outsourcing": ["outsourcing", "third party", "third-party", "service providers"],
    "liquidity": ["liquidity", "intraday", "payment and settlement"],
    "stress_testing": ["stress test", "stress testing", "scenario"],
    "srep": ["srep", "icaap", "ilaap", "supervisory review"],
    "ict": ["ict", "security", "cyber", "information and communication"],
    "operational_resilience": ["operational resilience", "impact tolerances", "business services"],
    "branch_governance": ["branches", "subsidiaries", "third-country", "international banks"],
}

BAD_LINE_PATTERNS = [
    "share this page",
    "stay connected",
    "copyright",
    "cookies",
    "privacy",
    "sitemap",
    "contact",
    "back to top",
    "last update",
    "sign up",
    "pdf",
    "http://",
    "https://",
    "www.",
    "skip to main content",
    "menu extranet",
    "log in about us",
    "registration link",
    "interested in participating",
    "we will confirm the participation",
    "the event will comprise",
    "question and answer session",
    "respondent suggested",
    "another respondent",
    "the committee wishes to thank",
    "feedback and comments",
    "consultation paper",
    "public consultation",
    "press release",
    "respondent",
    "comments summary",
    "summary of responses",
    "the eba's analysis",
    "amendments to the proposals",
    "the guidelines have been amended",
    "final report on",
    "gl on ",
    "policy statement",
    "supervisory statement 26 march",
    "stakeholder meeting",
    "submit them by email",
    "are you happy with this page",
    "prudential regulation //",
    "senior representatives from the banking industry",
    "panel discussion",
    "close ‘",
    "open ‘",
    "submitting comments",
    "submit your comments",
    "using these templates",
    "please provide relevant examples",
    "midnight cet",
    "following the feedback received",
    "requested clarification",
    "summary of proposals",
    "consultative version",
    "this cp",
    "proposals outlined in this cp",
    "the proposals in this cp",
    "footnote",
    "looking for comments",
    "draft principles",
    "tem plates",
    "final notice",
    " c onsideration",
    " th e ",
    " wi th ",
    " t heir ",
    "they reminded",
    "we also facilitate",
    "guide is designed",
    "executive summary",
    "the final guidelines",
    "a new chapter",
    "principle 1",
    "principle 5",
    "industry participants",
    "published today",
    "the core principles are used",
    "has assessed how",
    "wants banks",
    "this update takes into account",
    "the revised guidance emphasises",
    "saw a need",
    "the review will involve",
    "aggregate results",
    "possibl e",
    "result s",
    "ove rseeing",
    "supervisory fu nction",
    "guidelines have been clarified",
    "has been clarified and changed",
    "mentioned sentence",
    "this paragraph is relevant",
    "the eba considers that this paragraph",
    "the eba considers that the mentioned",
    "changed to:",
    "changed to ‘",
    "changed to '",
]

BAD_LINE_REGEXES = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"\[page\s+\d+\]",
        r"\bpage\s+\d+\s+of\s+\d+\b",
        r"\bparagraph\s+\d+\b",
        r"^\d+(?:\.\d+)?\s+",
        r"^\d+\s+this section should be read\b",
        r"^\d+\s+the\s+\w+\s+guidelines\b",
        r"^\d+\s+final report\b",
        r"^\d+\s+gl on\b",
        r"\.\d+\s+[A-Z]",
        r"\b\d+\s+Critical functions are defined\b",
        r"\b[\w.+-]+@[\w.-]+\.\w+\b",
    ]
]


def infer_source_org(path: Path) -> str:
    parts = set(path.parts)
    if "bcbs" in parts:
        return "BCBS"
    if "eba" in parts:
        return "EBA"
    if "ecb" in parts:
        return "ECB"
    if "pra_boe" in parts:
        return "PRA-BoE"
    if "fed_occ" in parts:
        return "Federal Reserve"
    return "unknown"


def infer_theme(text: str, path: Path) -> str:
    haystack = f"{path.name} {text}".lower()
    best_theme = "general"
    best_score = 0
    for theme, keywords in THEME_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in haystack)
        if score > best_score:
            best_theme = theme
            best_score = score
    return best_theme


def normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r-–—")


def split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [normalize_space(chunk) for chunk in chunks if normalize_space(chunk)]


def is_good_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if any(pattern in lowered for pattern in BAD_LINE_PATTERNS):
        return False
    if any(regex.search(sentence) for regex in BAD_LINE_REGEXES):
        return False
    if not (80 <= len(sentence) <= 360):
        return False
    if sentence.count(",") > 8:
        return False
    if sentence.count("|") > 0:
        return False
    if any(char in sentence for char in ["•", "", "…"]):
        return False
    if "?" in sentence:
        return False
    if sentence.count("(") != sentence.count(")"):
        return False
    if re.search(r"\s+[a-z]\.$", sentence):
        return False
    if "[" in sentence or "]" in sentence:
        return False
    if sentence.count(";") > 5:
        return False
    if re.match(r"^(close|open|skip|menu)\b", lowered):
        return False
    alpha_chars = [ch for ch in sentence if ch.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
        if upper_ratio > 0.35:
            return False
    if len(sentence.split()) < 10:
        return False
    return True


def sentence_score(sentence: str) -> int:
    lowered = sentence.lower()
    score = 0
    for keywords in THEME_KEYWORDS.values():
        score += sum(1 for keyword in keywords if keyword in lowered)
    score += sum(
        1
        for marker in [
            "should",
            "must",
            "expected",
            "requires",
            "important",
            "critical",
            "not legally binding",
            "risk management",
            "supervisory",
        ]
        if marker in lowered
    )
    return score


def replace_once(pattern: str, repl: str) -> Callable[[str], str | None]:
    regex = re.compile(pattern, flags=re.IGNORECASE)

    def apply(sentence: str) -> str | None:
        if not regex.search(sentence):
            return None
        return regex.sub(repl, sentence, count=1)

    return apply


CONTRADICTION_TRANSFORMS: list[tuple[str, Callable[[str], str | None]]] = [
    ("not_legally_binding_inversion", replace_once(r"\bis not legally binding\b", "is legally binding")),
    ("should_also_not", replace_once(r"\bshould also\b", "should not")),
    ("should_not", replace_once(r"\bshould\b", "should not")),
    ("must_also_not", replace_once(r"\bmust also\b", "must not")),
    ("must_not", replace_once(r"\bmust\b", "must not")),
    ("expected_not", replace_once(r"\bare expected to\b", "are not expected to")),
    ("expects_not", replace_once(r"\bexpects\b", "does not expect")),
    ("requires_not", replace_once(r"\brequires\b", "does not require")),
    ("include_exclude", replace_once(r"\binclude\b", "exclude")),
    ("includes_excludes", replace_once(r"\bincludes\b", "excludes")),
    ("important_not", replace_once(r"\bis important\b", "is not important")),
    ("critical_not", replace_once(r"\bare critical\b", "are not critical")),
    ("can_cannot", replace_once(r"\bcan lead to\b", "cannot lead to")),
    ("will_also_not", replace_once(r"\bwill also\b", "will not")),
    ("will_not", replace_once(r"\bwill\b", "will not")),
]


def make_contradiction(sentence: str) -> tuple[str, str] | None:
    for name, transform in CONTRADICTION_TRANSFORMS:
        if name in {"should_not", "must_not", "will_not"} and re.search(r"\bnot\b", sentence, re.IGNORECASE):
            continue
        out = transform(sentence)
        if out and out != sentence and is_good_sentence(out):
            if re.search(r"\bnot\s+not\b", out, flags=re.IGNORECASE):
                continue
            if re.search(r"\b(?:should|must|will)\s+not\s+[–-]", out, flags=re.IGNORECASE):
                continue
            return name, out
    return None


def load_source_sentences(raw_root: Path, max_sentences_per_doc: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_root.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        candidates = [
            s for s in split_sentences(text)
            if is_good_sentence(s) and sentence_score(s) > 0
        ]
        candidates = sorted(candidates, key=lambda s: (sentence_score(s), len(s)), reverse=True)
        top = candidates[: max_sentences_per_doc * 3]
        rng.shuffle(top)
        for sentence in top[:max_sentences_per_doc]:
            rows.append(
                {
                    "sentence": sentence,
                    "source_file": str(path),
                    "source_org": infer_source_org(path),
                    "theme": infer_theme(sentence, path),
                }
            )
    return rows


def build_pool(
    sentences: list[dict[str, Any]],
    seed: int,
    max_per_label: int,
    id_prefix: str,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows_by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}

    contradiction_sources: list[tuple[dict[str, Any], str, str]] = []
    for item in sentences:
        transformed = make_contradiction(item["sentence"])
        if transformed is not None:
            transform_name, contradiction = transformed
            contradiction_sources.append((item, transform_name, contradiction))

    for item, transform_name, contradiction in contradiction_sources:
        base_meta = {
            "source_file": item["source_file"],
            "source_org": item["source_org"],
            "theme": item["theme"],
            "builder": "build_finreg_detector_candidate_pool.py",
            "review_status": "pending",
            "quality": "auto_candidate",
        }
        rows_by_label["entailment"].append(
            {
                "premise": item["sentence"],
                "hypothesis": item["sentence"],
                "label": "entailment",
                "metadata": {**base_meta, "pair_type": "extractive_entailment"},
            }
        )
        rows_by_label["contradiction"].append(
            {
                "premise": item["sentence"],
                "hypothesis": contradiction,
                "label": "contradiction",
                "metadata": {
                    **base_meta,
                    "pair_type": "minimal_contradiction",
                    "transform": transform_name,
                },
            }
        )

    by_other_theme = list(sentences)
    for item in sentences:
        candidates = [
            other
            for other in by_other_theme
            if other["source_file"] != item["source_file"]
            and other["theme"] != item["theme"]
            and other["source_org"] != item["source_org"]
        ]
        if not candidates:
            continue
        other = rng.choice(candidates)
        rows_by_label["neutral"].append(
            {
                "premise": item["sentence"],
                "hypothesis": other["sentence"],
                "label": "neutral",
                "metadata": {
                    "source_file": item["source_file"],
                    "source_org": item["source_org"],
                    "theme": item["theme"],
                    "neutral_hypothesis_source_file": other["source_file"],
                    "neutral_hypothesis_source_org": other["source_org"],
                    "neutral_hypothesis_theme": other["theme"],
                    "builder": "build_finreg_detector_candidate_pool.py",
                    "review_status": "pending",
                    "quality": "auto_candidate",
                    "pair_type": "cross_source_neutral",
                },
            }
        )

    selected: list[dict[str, Any]] = []
    for label, rows in rows_by_label.items():
        rng.shuffle(rows)
        selected.extend(rows[:max_per_label])

    rng.shuffle(selected)
    for idx, row in enumerate(selected, start=1):
        row["id"] = f"{id_prefix}_{idx:04d}"
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "labels": dict(Counter(row["label"] for row in rows)),
        "source_orgs": dict(Counter(row["metadata"].get("source_org") for row in rows)),
        "themes": dict(Counter(row["metadata"].get("theme") for row in rows)),
        "pair_types": dict(Counter(row["metadata"].get("pair_type") for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/processed/finreg")
    parser.add_argument("--output", default="data/domain_finreg/detector_candidate_pool_v11.jsonl")
    parser.add_argument("--summary", default="data/domain_finreg/detector_candidate_pool_v11_summary.json")
    parser.add_argument("--id-prefix", default="fdcp_v11")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-sentences-per-doc", type=int, default=8)
    parser.add_argument("--max-per-label", type=int, default=120)
    args = parser.parse_args()

    sentences = load_source_sentences(
        raw_root=Path(args.raw_root),
        max_sentences_per_doc=args.max_sentences_per_doc,
        seed=args.seed,
    )
    pool = build_pool(
        sentences=sentences,
        seed=args.seed,
        max_per_label=args.max_per_label,
        id_prefix=args.id_prefix,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in pool:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "raw_root": args.raw_root,
        "seed": args.seed,
        "max_sentences_per_doc": args.max_sentences_per_doc,
        "max_per_label": args.max_per_label,
        "candidate_sentences": len(sentences),
        "pool": summarize(pool),
        "note": "Auto-generated candidate pool. Review before using for gold eval or training.",
    }
    summary_path = Path(args.summary)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
