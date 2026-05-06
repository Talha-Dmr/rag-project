#!/usr/bin/env python3
"""
Build a small manual-eval set with model outputs and sources.

The repository is now FinReg-first. The script still supports explicit domain specs, but the
default path is the active FinReg baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


def load_questions(path: Path) -> List[Dict]:
    questions: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def load_label_map(path: Optional[Path]) -> Dict[str, Dict]:
    if not path or not path.exists():
        return {}
    rows: Dict[str, Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item.get("id")
            if qid:
                rows[str(qid)] = item
    return rows


def format_sources(sources: List[Dict], max_labels: int = 3) -> str:
    if not sources:
        return "none"
    order: List[str] = []
    pages_by: Dict[str, List[int]] = {}
    for item in sources:
        label = item.get("label") or Path(item.get("source", "")).name or "source"
        if label not in pages_by:
            pages_by[label] = []
            order.append(label)
        page = item.get("page")
        if page is not None and page not in pages_by[label]:
            pages_by[label].append(page)
    parts: List[str] = []
    for label in order:
        pages = sorted(pages_by[label])
        if pages:
            parts.append(f"{label} p{','.join(str(p) for p in pages)}")
        else:
            parts.append(label)
    if len(parts) > max_labels:
        extra = len(parts) - max_labels
        parts = parts[:max_labels] + [f"+{extra} more"]
    return "; ".join(parts)


def run_domain(
    domain: str,
    config_name: str,
    questions: List[Dict],
    limit: int,
    seed: int,
    label_map: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    if limit and limit < len(questions):
        random.seed(seed)
        questions = random.sample(questions, k=limit)

    config = load_config(config_name)
    rag = RAGPipeline.from_config(config)
    abstain_message = (config.get("gating", {}).get("abstain_message", "") or "").strip()

    records: List[Dict] = []
    for item in questions:
        query = item.get("query", "")
        if not query:
            continue
        result = rag.query(
            query_text=query,
            return_context=False,
            return_sources=True,
            detect_hallucinations=True
        )
        answer = (result.get("answer") or "").strip()
        gating = result.get("gating") or {}
        action = gating.get("action", "none")
        sources = result.get("sources") or []
        record = {
            "domain": domain,
            "id": item.get("id"),
            "type": item.get("type", "unknown"),
            "query": query,
            "answer": answer,
            "abstain": bool(abstain_message and answer == abstain_message),
            "action": action,
            "sources": sources,
            "gating": gating,
        }
        labels = (label_map or {}).get(str(item.get("id")), {})
        if labels:
            record["labels"] = {k: v for k, v in labels.items() if k != "id"}
        records.append(record)
    return records


def parse_domain_spec(value: str) -> Dict[str, str]:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3 or not all(parts):
        raise ValueError(
            "Invalid --domain format. Use: name,config,questions_path"
        )
    return {
        "name": parts[0],
        "config": parts[1],
        "questions": parts[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manual evaluation set")
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="Domain spec: name,config,questions_path (repeatable)",
    )
    parser.add_argument(
        "--finreg-config",
        default="gating_finreg_ebcar_logit_mi_sc009",
        help="Default FinReg config name",
    )
    parser.add_argument(
        "--finreg-questions",
        default="data/domain_finreg/questions_finreg_conflict_phase1_refined_v2.jsonl",
        help="Default FinReg question set",
    )
    parser.add_argument(
        "--label-map",
        default="data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_labels.jsonl",
        help="Optional JSONL label map keyed by question id",
    )
    parser.add_argument("--per-domain", type=int, default=15, help="Questions per domain")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="evaluation_results/manual_eval")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.domain:
        domain_specs = [parse_domain_spec(v) for v in args.domain]
    else:
        domain_specs = [
            {
                "name": "finreg",
                "config": args.finreg_config,
                "questions": args.finreg_questions,
            },
        ]

    label_map = load_label_map(Path(args.label_map) if args.label_map else None)

    records = []
    for spec in domain_specs:
        questions = load_questions(Path(spec["questions"]))
        records.extend(
            run_domain(
                spec["name"],
                spec["config"],
                questions,
                args.per_domain,
                args.seed,
                label_map if spec["name"] == "finreg" else None,
            )
        )

    jsonl_path = out_dir / "manual_eval_set.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    csv_path = out_dir / "manual_eval_set.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain",
            "id",
            "type",
            "task_family",
            "ambiguity_family",
            "support_level",
            "question_style",
            "theme",
            "query",
            "answer",
            "abstain",
            "action",
            "sources",
            "label_correct",
            "notes",
        ])
        for rec in records:
            writer.writerow([
                rec.get("domain"),
                rec.get("id"),
                rec.get("type"),
                (rec.get("labels") or {}).get("task_family", ""),
                (rec.get("labels") or {}).get("ambiguity_family", ""),
                (rec.get("labels") or {}).get("support_level", ""),
                (rec.get("labels") or {}).get("question_style", ""),
                (rec.get("labels") or {}).get("theme", ""),
                rec.get("query"),
                rec.get("answer"),
                rec.get("abstain"),
                rec.get("action"),
                format_sources(rec.get("sources") or []),
                "",
                "",
            ])

    print(f"Wrote {len(records)} items")
    print(f"- JSONL: {jsonl_path}")
    print(f"- CSV:   {csv_path}")


if __name__ == "__main__":
    main()
