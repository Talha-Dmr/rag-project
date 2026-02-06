#!/usr/bin/env python3
"""
Collect qualitative RAG examples (question, answer, gating) for docs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


def load_questions(path: Path, qtype: str | None) -> List[Dict]:
    questions: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if qtype and item.get("type") != qtype:
                continue
            questions.append(item)
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect qualitative examples")
    parser.add_argument("--config", required=True, help="Config name (without .yaml)")
    parser.add_argument("--questions", required=True, help="Path to JSONL questions")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--type", dest="qtype", default="conflict", help="Filter by type")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    args = parser.parse_args()

    questions = load_questions(Path(args.questions), args.qtype)
    if not questions:
        raise SystemExit("No questions found for requested type")

    random.seed(args.seed)
    sample = random.sample(questions, k=min(args.limit, len(questions)))

    config = load_config(args.config)
    rag = RAGPipeline.from_config(config)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out:
        for item in sample:
            query = item.get("query", "")
            result = rag.query(
                query_text=query,
                return_context=False,
                return_sources=True,
                detect_hallucinations=True
            )
            record = {
                "id": item.get("id"),
                "type": item.get("type"),
                "query": query,
                "answer": result.get("answer"),
                "abstain": result.get("answer")
                == config.get("gating", {}).get("abstain_message", ""),
                "gating": result.get("gating"),
                "sources": result.get("sources", []),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(sample)} examples to {out_path}")


if __name__ == "__main__":
    main()
