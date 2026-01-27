#!/usr/bin/env python3
"""Quick eval on AmbigQA dev_light sample with baseline vs EBCAR+gating."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

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


def run_eval(cfg_name: str, samples: List[Dict]) -> Dict:
    cfg = load_config(cfg_name)
    rag = RAGPipeline.from_config(cfg)

    total = 0
    abstain = 0
    hit = 0

    for ex in samples:
        q = ex.get("question", "")
        refs = extract_reference_answers(ex)
        result = rag.query(q)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", default="data/ambiguity_datasets/02_ambigqa/dev_light.json")
    args = parser.parse_args()

    samples = load_samples(Path(args.data), args.n, args.seed)

    baseline = run_eval("gating_demo", samples)
    ebcar = run_eval("gating_demo_ebcar", samples)

    print("baseline:", baseline)
    print("ebcar:", ebcar)


if __name__ == "__main__":
    main()
