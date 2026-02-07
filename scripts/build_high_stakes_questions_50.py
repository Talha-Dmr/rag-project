#!/usr/bin/env python3
"""
Build 50-question high-stakes sets for health, financial regulation, and disaster risk.

It keeps existing 20Q seed questions and adds domain-specific questions to reach:
- 10 sanity
- 40 conflict
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
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
        "Are decision thresholds for {topic} aligned between {s1} and {s2} in recent guidance?",
        "When sources conflict on {topic}, which of {s1} and {s2} is more conservative?",
        "How do {s1} and {s2} justify disagreement on {topic} under uncertainty?",
        "Do {s1} and {s2} recommend different operational actions for {topic} in edge cases?",
    ]

    questions: List[str] = []
    pairs: List[Tuple[str, str]] = list(combinations(sources, 2))
    idx = 0
    for s1, s2 in pairs:
        for topic in topics:
            t = templates[idx % len(templates)]
            questions.append(t.format(s1=s1, s2=s2, topic=topic))
            idx += 1
            if len(questions) >= needed:
                return _dedup_keep_order(questions)[:needed]

    return _dedup_keep_order(questions)[:needed]


def _generate_sanity(source: str, topics: List[str], needed: int) -> List[str]:
    templates = [
        "What is the primary objective of {topic} in {source} guidance?",
        "Which risk does {source} mainly address through {topic}?",
        "How does {source} define success criteria for {topic}?",
        "What implementation constraint is emphasized by {source} for {topic}?",
        "Why does {source} treat {topic} as a governance priority?",
    ]
    questions: List[str] = []
    for i, topic in enumerate(topics):
        t = templates[i % len(templates)]
        questions.append(t.format(source=source, topic=topic))
        if len(questions) >= needed:
            break
    return _dedup_keep_order(questions)[:needed]


@dataclass
class DomainSpec:
    name: str
    id_prefix: str
    seed_path: Path
    out_path: Path
    sources: List[str]
    topics: List[str]
    sanity_source: str


def build_domain(spec: DomainSpec, total: int = 50, sanity_target: int = 10) -> List[Dict[str, str]]:
    seed = _read_jsonl(spec.seed_path)
    seed_sanity = [x for x in seed if x.get("type") == "sanity"]
    seed_conflict = [x for x in seed if x.get("type") == "conflict"]

    if len(seed) > total:
        raise ValueError(f"{spec.seed_path} has more than {total} rows")

    add_sanity = max(0, sanity_target - len(seed_sanity))
    conflict_target = total - sanity_target
    add_conflict = max(0, conflict_target - len(seed_conflict))

    extra_sanity = _generate_sanity(spec.sanity_source, spec.topics, add_sanity)
    extra_conflict = _generate_conflicts(spec.sources, spec.topics, add_conflict)

    all_sanity = [x["query"] for x in seed_sanity] + extra_sanity
    all_conflict = [x["query"] for x in seed_conflict] + extra_conflict

    all_sanity = _dedup_keep_order(all_sanity)[:sanity_target]
    all_conflict = _dedup_keep_order(all_conflict)[:conflict_target]

    rows: List[Dict[str, str]] = []
    idx = 1
    for q in all_sanity:
        rows.append({"id": f"{spec.id_prefix}{idx:02d}", "type": "sanity", "query": q})
        idx += 1
    for q in all_conflict:
        rows.append({"id": f"{spec.id_prefix}{idx:02d}", "type": "conflict", "query": q})
        idx += 1

    if len(rows) != total:
        raise ValueError(f"{spec.name}: expected {total} rows, got {len(rows)}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 50Q high-stakes domain question sets")
    parser.add_argument("--total", type=int, default=50, help="Total questions per domain")
    parser.add_argument("--sanity", type=int, default=10, help="Sanity question count per domain")
    args = parser.parse_args()

    specs = [
        DomainSpec(
            name="health",
            id_prefix="hq",
            seed_path=Path("data/domain_health/questions_health_conflict.jsonl"),
            out_path=Path("data/domain_health/questions_health_conflict_50.jsonl"),
            sources=["WHO", "CDC", "NICE", "ECDC"],
            topics=[
                "booster prioritization",
                "isolation duration",
                "mask guidance by setting",
                "antiviral eligibility for mild cases",
                "asymptomatic testing cadence",
                "pediatric vaccination interval",
                "return-to-work criteria for clinicians",
                "travel screening and quarantine",
                "conditional recommendation criteria",
                "evidence certainty thresholds",
            ],
            sanity_source="WHO",
        ),
        DomainSpec(
            name="finreg",
            id_prefix="fq",
            seed_path=Path("data/domain_finreg/questions_finreg_conflict.jsonl"),
            out_path=Path("data/domain_finreg/questions_finreg_conflict_50.jsonl"),
            sources=["BCBS", "EBA", "ECB", "Federal Reserve", "PRA"],
            topics=[
                "risk aggregation timeliness",
                "board accountability for data quality",
                "manual adjustments in regulatory reporting",
                "model validation frequency",
                "climate risk in ICAAP",
                "outsourcing controls for risk systems",
                "materiality thresholds for reporting errors",
                "audit trail retention",
                "intraday liquidity monitoring",
                "AI model explainability requirements",
            ],
            sanity_source="BCBS",
        ),
        DomainSpec(
            name="disaster",
            id_prefix="dq",
            seed_path=Path("data/domain_disaster/questions_disaster_conflict.jsonl"),
            out_path=Path("data/domain_disaster/questions_disaster_conflict_50.jsonl"),
            sources=["NOAA", "IPCC", "UNDRR", "WMO", "FEMA"],
            topics=[
                "drought outlook interpretation",
                "flood return-period assumptions",
                "heat attribution confidence",
                "compound-hazard evacuation triggers",
                "early warning false-alarm tradeoff",
                "adaptation option ranking",
                "wildfire readiness thresholds",
                "coastal risk scenario selection",
                "disaster financing priorities",
                "vulnerable-population prioritization",
            ],
            sanity_source="UNDRR",
        ),
    ]

    for spec in specs:
        rows = build_domain(spec, total=args.total, sanity_target=args.sanity)
        _write_jsonl(spec.out_path, rows)
        print(f"{spec.name}: wrote {len(rows)} -> {spec.out_path}")


if __name__ == "__main__":
    main()

