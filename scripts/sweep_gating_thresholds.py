#!/usr/bin/env python3
"""Sweep retrieval-score gating thresholds on a fixed AmbigQA sample."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


def load_samples(path: Path, n: int, seed: int) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    random.Random(seed).shuffle(data)
    return data[:n]


def answer_has_any(answer: str, answer_list: List[str]) -> bool:
    ans_lower = answer.lower()
    for a in answer_list:
        if a and a.lower() in ans_lower:
            return True
    return False


def extract_reference_answers(example: Dict) -> List[str]:
    refs = []
    for ann in example.get("annotations", []):
        if ann.get("type") == "singleAnswer":
            refs.extend(ann.get("answer", []))
        elif ann.get("type") == "multipleQAs":
            for qa in ann.get("qaPairs", []):
                refs.extend(qa.get("answer", []))
    return list({r for r in refs if isinstance(r, str)})


def run_eval(rag: RAGPipeline, samples: List[Dict], override: Dict) -> Dict:
    total = 0
    abstain = 0
    hit = 0

    for ex in samples:
        q = ex.get("question", "")
        refs = extract_reference_answers(ex)
        result = rag.query(q, gating=override)
        ans = result.get("answer", "") or ""

        if "I don't know" in ans or "Bu soruya güvenilir şekilde yanıt veremiyorum" in ans:
            abstain += 1
        else:
            if refs and answer_has_any(ans, refs):
                hit += 1
        total += 1

    return {
        "total": total,
        "abstain": abstain,
        "abstain_rate": abstain / total if total else 0.0,
        "hit": hit,
        "hit_rate": hit / total if total else 0.0,
    }


def parse_thresholds(raw: str) -> List[Tuple[float, float]]:
    pairs = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid threshold pair: {item}")
        pairs.append((float(parts[0]), float(parts[1])))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data", default="data/ambiguity_datasets/02_ambigqa/dev_light.json")
    parser.add_argument("--config", default="gating_demo_ebcar")
    parser.add_argument(
        "--thresholds",
        default="0.20:0.10,0.25:0.15,0.30:0.20",
        help="Comma-separated pairs: min_retrieval_score:min_mean_retrieval_score",
    )
    args = parser.parse_args()

    samples = load_samples(Path(args.data), args.n, args.seed)
    cfg = load_config(args.config)
    rag = RAGPipeline.from_config(cfg)

    thresholds = parse_thresholds(args.thresholds)
    for min_score, min_mean in thresholds:
        override = {
            "min_retrieval_score": min_score,
            "min_mean_retrieval_score": min_mean,
        }
        metrics = run_eval(rag, samples, override)
        print(f"thresholds max={min_score} mean={min_mean} -> {metrics}")


if __name__ == "__main__":
    main()
