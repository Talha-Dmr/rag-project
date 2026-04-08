#!/usr/bin/env python3
"""
Build the active 50-question FinReg confirmation set from the current refined v2 seed set.

The repository is now FinReg-first. This script intentionally no longer builds health or
disaster sets.
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _read_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dedup_keep_order(questions: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for q in questions:
        key = _normalize(q)
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _generate_conflicts(sources: List[str], topics: List[str], needed: int) -> List[str]:
    templates = [
        "Do {s1} and {s2} differ on {topic} when evidence certainty is low?",
        "Are supervisory expectations for {topic} aligned between {s1} and {s2}?",
        "When materials diverge on {topic}, which of {s1} and {s2} is more conservative?",
        "How do {s1} and {s2} frame disagreement on {topic} under uncertainty?",
        "Do {s1} and {s2} imply different operational actions for {topic} in difficult cases?",
    ]

    questions: List[str] = []
    pairs: List[Tuple[str, str]] = list(combinations(sources, 2))
    idx = 0
    for s1, s2 in pairs:
        for topic in topics:
            template = templates[idx % len(templates)]
            questions.append(template.format(s1=s1, s2=s2, topic=topic))
            idx += 1
            if len(questions) >= needed:
                return _dedup_keep_order(questions)[:needed]

    return _dedup_keep_order(questions)[:needed]


def _generate_sanity(source: str, topics: List[str], needed: int) -> List[str]:
    templates = [
        "What is the primary supervisory objective of {topic} in {source} guidance?",
        "Which governance concern does {source} mainly address through {topic}?",
        "How does {source} frame good practice for {topic}?",
        "What implementation constraint is emphasized by {source} for {topic}?",
        "Why does {source} treat {topic} as a control priority?",
    ]
    questions: List[str] = []
    for i, topic in enumerate(topics):
        template = templates[i % len(templates)]
        questions.append(template.format(source=source, topic=topic))
        if len(questions) >= needed:
            break
    return _dedup_keep_order(questions)[:needed]


def build_finreg(total: int = 50, sanity_target: int = 10) -> List[Dict[str, str]]:
    seed_path = Path("data/domain_finreg/questions_finreg_conflict_phase1_refined_v2.jsonl")
    seed = _read_jsonl(seed_path)
    seed_sanity = [x for x in seed if x.get("type") == "sanity"]
    seed_conflict = [x for x in seed if x.get("type") == "conflict"]

    if len(seed) > total:
        raise ValueError(f"{seed_path} has more than {total} rows")

    add_sanity = max(0, sanity_target - len(seed_sanity))
    conflict_target = total - sanity_target
    add_conflict = max(0, conflict_target - len(seed_conflict))

    sources = ["BCBS", "EBA", "ECB", "PRA"]
    topics = [
        "risk aggregation timeliness under stress",
        "board and senior management responsibility for data quality",
        "manual workarounds in risk reporting controls",
        "documentation and governance for internal model changes",
        "climate-risk integration into governance and risk management",
        "outsourcing controls for risk-reporting systems",
        "materiality and escalation for reporting errors",
        "auditability and traceability for risk data systems",
        "intraday liquidity monitoring during the day",
        "group-wide governance across subsidiaries and consolidated groups",
    ]

    extra_sanity = _generate_sanity("BCBS", topics, add_sanity)
    extra_conflict = _generate_conflicts(sources, topics, add_conflict)

    all_sanity = _dedup_keep_order([x["query"] for x in seed_sanity] + extra_sanity)[:sanity_target]
    all_conflict = _dedup_keep_order([x["query"] for x in seed_conflict] + extra_conflict)[:conflict_target]

    rows: List[Dict[str, str]] = []
    idx = 1
    for q in all_sanity:
        rows.append({"id": f"fq{idx:02d}", "type": "sanity", "query": q})
        idx += 1
    for q in all_conflict:
        rows.append({"id": f"fq{idx:02d}", "type": "conflict", "query": q})
        idx += 1

    if len(rows) != total:
        raise ValueError(f"expected {total} rows, got {len(rows)}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the active 50Q FinReg question set")
    parser.add_argument("--total", type=int, default=50, help="Total questions to produce")
    parser.add_argument("--sanity", type=int, default=10, help="Sanity question count")
    parser.add_argument(
        "--out",
        default="data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    rows = build_finreg(total=args.total, sanity_target=args.sanity)
    out_path = Path(args.out)
    _write_jsonl(out_path, rows)
    print(f"finreg: wrote {len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()
