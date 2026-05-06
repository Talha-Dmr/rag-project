#!/usr/bin/env python3
"""Build a larger RAG-aware pseudo-NLI dataset from the FinReg corpus.

This is not a gold dataset. It creates training candidates closer to the actual
detector call shape: retrieved context as premise, answer-like claim as hypothesis.
The held-out FinReg detector eval set is copied unchanged as test.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


LABELS = ("entailment", "neutral", "contradiction")
LABEL_TO_ID = {"entailment": 0, "neutral": 1, "contradiction": 2}

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

BAD_PATTERNS = [
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
    "http://",
    "https://",
    "www.",
    "skip to main content",
    "registration link",
    "respondent",
    "comments summary",
    "summary of responses",
    "the eba's analysis",
    "amendments to the proposals",
    "the guidelines have been amended",
    "final report on",
    "gl on ",
    "policy statement",
    "stakeholder meeting",
    "submit them by email",
    "are you happy with this page",
    "prudential regulation //",
    "submitting comments",
    "submit your comments",
    "following the feedback received",
    "requested clarification",
    "summary of proposals",
    "consultative version",
    "this cp",
    "footnote",
    "looking for comments",
    "draft principles",
    "final notice",
    "they reminded",
    "we also facilitate",
    "guide is designed",
    "executive summary",
    "the final guidelines",
    "a new chapter",
    "industry participants",
    "published today",
    "the core principles are used",
    "has assessed how",
    "this update takes into account",
    "saw a need",
    "the review will involve",
    "aggregate results",
    "possibl e",
    "result s",
    "ove rseeing",
    "supervisory fu nction",
    "this is already included",
    "already included",
    "no need to have",
    "no further changes",
    "superseded",
    "has been superseded",
    "main changes to",
    "summarised below",
    "this new chapter",
    "new chapter includes",
    "the guidelines should therefore specify",
    "should therefore specify",
    "the eba considers",
    "r egulated",
    "rel iability",
    "the ir ",
    "r isk",
    "consid e r",
    "inventor ies",
    "accordi ng",
    "specif ic",
    "amended to clarify",
    "it asked if",
    "the use cloud services",
    "will be repealed by these guidelines",
    "full y",
    "outco mes",
    "existin g",
    "f inancial",
    "group ,",
    "bottom -up",
    "time- sensitive",
    "th e ",
]

BAD_REGEXES = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"\[page\s+\d+\]",
        r"\bpage\s+\d+\s+of\s+\d+\b",
        r"\bparagraph\s+\d+\b",
        r"^\d+(?:\.\d+)?\s+",
        r"\.\d+\s+[A-Z]",
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
    text = text.replace(" - ", "-")
    return text.strip(" \t\n\r-–—")


def split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " ")
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [normalize_space(chunk) for chunk in chunks if normalize_space(chunk)]


def is_good_text(text: str, min_len: int = 70, max_len: int = 520) -> bool:
    lowered = text.lower()
    if any(pattern in lowered for pattern in BAD_PATTERNS):
        return False
    if any(regex.search(text) for regex in BAD_REGEXES):
        return False
    if not (min_len <= len(text) <= max_len):
        return False
    if any(char in text for char in ["|", "•", "", "…"]):
        return False
    if "?" in text:
        return False
    if text.count("(") != text.count(")"):
        return False
    if len(text.split()) < 9:
        return False
    alpha_chars = [ch for ch in text if ch.isalpha()]
    if alpha_chars:
        upper_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
        if upper_ratio > 0.35:
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
            "risk management",
            "supervisory",
            "governance",
            "controls",
            "framework",
        ]
        if marker in lowered
    )
    return score


def clean_claim(sentence: str) -> str | None:
    claim = re.sub(r"^\s*[A-Z][A-Za-z ]{2,50}\s+(?=The |A |An |In |For |Where |While |This )", "", sentence)
    claim = re.sub(r"\s*\([^)]{0,80}\)", "", claim)
    claim = re.sub(r"\s+", " ", claim).strip()
    split_markers = [
        ", including ",
        ", in particular ",
        ", among others, ",
        ", for example ",
        ";",
    ]
    for marker in split_markers:
        if marker in claim and len(claim.split(marker, 1)[0]) >= 80:
            claim = claim.split(marker, 1)[0].strip() + "."
            break
    if len(claim) > 260:
        parts = re.split(r",\s+(?=(and|while|where|which|that)\b)", claim)
        if parts and len(parts[0]) >= 90:
            claim = parts[0].strip() + "."
    claim = normalize_space(claim)
    if claim and not claim.endswith((".", "!", "?")):
        claim += "."
    if not claim or not is_good_text(claim, min_len=55, max_len=320):
        return None
    if claim.rstrip(".") == normalize_space(sentence).rstrip("."):
        return None
    return claim


def replace_once(pattern: str, repl: str) -> Callable[[str], str | None]:
    regex = re.compile(pattern, flags=re.IGNORECASE)

    def apply(text: str) -> str | None:
        if not regex.search(text):
            return None
        return regex.sub(repl, text, count=1)

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
    ("will_also_not", replace_once(r"\bwill also\b", "will not")),
    ("will_not", replace_once(r"\bwill\b", "will not")),
]


def make_contradiction(claim: str) -> tuple[str, str] | None:
    for name, transform in CONTRADICTION_TRANSFORMS:
        if name in {"should_not", "must_not", "will_not"} and re.search(r"\bnot\b", claim, re.IGNORECASE):
            continue
        out = transform(claim)
        if not out or out == claim:
            continue
        if re.search(r"\bnot\s+not\b", out, flags=re.IGNORECASE):
            continue
        if not is_good_text(out, min_len=55, max_len=340):
            continue
        return name, out
    return None


def context_window(sentences: list[str], idx: int) -> str:
    parts = []
    if idx > 0 and is_good_text(sentences[idx - 1], min_len=55, max_len=360):
        parts.append(sentences[idx - 1])
    parts.append(sentences[idx])
    if idx + 1 < len(sentences) and is_good_text(sentences[idx + 1], min_len=55, max_len=360):
        parts.append(sentences[idx + 1])
    context = normalize_space(" ".join(parts))
    if len(context) > 780 or not is_good_text(context, min_len=70, max_len=780):
        context = sentences[idx]
    return context


def load_claim_sources(raw_root: Path, max_sentences_per_doc: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_root.rglob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        sentences = split_sentences(text)
        candidates: list[dict[str, Any]] = []
        for idx, sentence in enumerate(sentences):
            if not is_good_text(sentence, min_len=75, max_len=420):
                continue
            if sentence_score(sentence) <= 0:
                continue
            claim = clean_claim(sentence)
            if not claim:
                continue
            transformed = make_contradiction(claim)
            if not transformed:
                continue
            context = context_window(sentences, idx)
            if not is_good_text(context, min_len=70, max_len=780):
                continue
            candidates.append(
                {
                    "sentence": sentence,
                    "claim": claim,
                    "contradiction_transform": transformed[0],
                    "contradiction_claim": transformed[1],
                    "context": context,
                    "source_file": str(path),
                    "source_org": infer_source_org(path),
                    "theme": infer_theme(sentence, path),
                    "score": sentence_score(sentence),
                }
            )

        candidates = sorted(candidates, key=lambda r: (r["score"], len(r["claim"])), reverse=True)
        top = candidates[: max_sentences_per_doc * 3]
        rng.shuffle(top)
        rows.extend(top[:max_sentences_per_doc])
    return rows


def build_rows(sources: list[dict[str, Any]], seed: int, max_per_label: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows_by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}

    for item in sources:
        base_meta = {
            "source_file": item["source_file"],
            "source_org": item["source_org"],
            "theme": item["theme"],
            "builder": "build_finreg_detector_ragaware_dataset.py",
            "quality": "pseudo_label",
        }
        rows_by_label["entailment"].append(
            {
                "premise": item["context"],
                "hypothesis": item["claim"],
                "label": "entailment",
                "metadata": {**base_meta, "pair_type": "compressed_supported_claim"},
            }
        )
        rows_by_label["contradiction"].append(
            {
                "premise": item["context"],
                "hypothesis": item["contradiction_claim"],
                "label": "contradiction",
                "metadata": {
                    **base_meta,
                    "pair_type": "context_claim_minimal_contradiction",
                    "transform": item["contradiction_transform"],
                },
            }
        )

    for item in sources:
        candidates = [
            other
            for other in sources
            if other["source_file"] != item["source_file"]
            and other["source_org"] != item["source_org"]
            and other["theme"] != item["theme"]
        ]
        if not candidates:
            continue
        other = rng.choice(candidates)
        rows_by_label["neutral"].append(
            {
                "premise": item["context"],
                "hypothesis": other["claim"],
                "label": "neutral",
                "metadata": {
                    "source_file": item["source_file"],
                    "source_org": item["source_org"],
                    "theme": item["theme"],
                    "neutral_hypothesis_source_file": other["source_file"],
                    "neutral_hypothesis_source_org": other["source_org"],
                    "neutral_hypothesis_theme": other["theme"],
                    "builder": "build_finreg_detector_ragaware_dataset.py",
                    "quality": "pseudo_label",
                    "pair_type": "context_cross_source_claim_neutral",
                },
            }
        )

    def select_source_balanced(label_rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        by_org: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in label_rows:
            by_org[row["metadata"].get("source_org", "unknown")].append(row)
        for org_rows in by_org.values():
            rng.shuffle(org_rows)

        selected_rows: list[dict[str, Any]] = []
        orgs = sorted(by_org)
        while len(selected_rows) < limit and any(by_org[org] for org in orgs):
            for org in orgs:
                if by_org[org]:
                    selected_rows.append(by_org[org].pop())
                    if len(selected_rows) >= limit:
                        break
        return selected_rows

    selected: list[dict[str, Any]] = []
    per_label = min(max_per_label, *(len(rows_by_label[label]) for label in LABELS))
    for label in LABELS:
        selected.extend(select_source_balanced(list(rows_by_label[label]), per_label))

    rng.shuffle(selected)
    for idx, row in enumerate(selected, start=1):
        row["id"] = f"fragd_v1_{idx:05d}"
    return selected


def split_rows(
    rows: list[dict[str, Any]],
    val_per_label: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for label in LABELS:
        label_rows = list(by_label[label])
        rng.shuffle(label_rows)
        val_rows.extend(label_rows[:val_per_label])
        train_rows.extend(label_rows[val_per_label:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], numeric_labels: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            out = dict(row)
            if numeric_labels:
                out["label"] = LABEL_TO_ID[out["label"]]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


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
    parser.add_argument("--output-dir", default="data/training/nli_dataset_finreg_detector_ragaware_v3")
    parser.add_argument("--test-source", default="data/domain_finreg/detector_eval_finreg_v1.jsonl")
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--max-sentences-per-doc", type=int, default=24)
    parser.add_argument("--max-per-label", type=int, default=180)
    parser.add_argument("--val-per-label", type=int, default=24)
    parser.add_argument("--numeric-labels", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    sources = load_claim_sources(Path(args.raw_root), args.max_sentences_per_doc, args.seed)
    rows = build_rows(sources, args.seed, args.max_per_label)
    train_rows, val_rows = split_rows(rows, args.val_per_label, args.seed)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows, args.numeric_labels)
    write_jsonl(output_dir / "val.jsonl", val_rows, args.numeric_labels)

    test_source = Path(args.test_source)
    if not test_source.exists():
        raise FileNotFoundError(test_source)
    shutil.copyfile(test_source, output_dir / "test.jsonl")

    test_rows = []
    with (output_dir / "test.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_rows.append(json.loads(line))

    summary = {
        "raw_root": args.raw_root,
        "seed": args.seed,
        "max_sentences_per_doc": args.max_sentences_per_doc,
        "max_per_label": args.max_per_label,
        "val_per_label": args.val_per_label,
        "candidate_sources": len(sources),
        "selected": summarize(rows),
        "train": summarize(train_rows),
        "val": summarize(val_rows),
        "test_source": str(test_source),
        "test": summarize(test_rows),
        "note": "Pseudo-labeled RAG-aware detector data. Do not treat as gold eval.",
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
