#!/usr/bin/env python3
"""
Self-consistency proxy baseline for energy questions.

For each query:
- retrieve context once
- generate N answers with the same temperature
- compute agreement via embedding cosine similarity
- abstain if agreement is below threshold
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


def load_questions(path: Path):
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def mean_pairwise_similarity(embeddings: List[List[float]]) -> float:
    if len(embeddings) < 2:
        return 1.0
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(
                float(
                    np.dot(embeddings[i], embeddings[j])
                    / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                )
            )
    return float(np.mean(sims)) if sims else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-consistency proxy baseline")
    parser.add_argument(
        "--config",
        default="gating_energy_ebcar",
        help="Config name (without .yaml)"
    )
    parser.add_argument(
        "--questions",
        default="data/domain_energy/questions_energy_conflict.jsonl",
        help="Path to JSONL questions"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of questions"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of sampled answers per query"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--agree_threshold",
        type=float,
        default=0.7,
        help="Mean pairwise similarity threshold for abstain"
    )
    args = parser.parse_args()

    if args.n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    config = load_config(args.config)
    rag = RAGPipeline.from_config(config)

    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]

    action_counts = Counter()
    type_counts = Counter()
    abstain_counts = Counter()

    for item in questions:
        qid = item.get("id")
        qtype = item.get("type", "unknown")
        query = item.get("query", "")
        if not query:
            continue

        retrieved_docs = rag.retriever.retrieve(query, k=rag.retriever.k)
        if rag.reranker:
            rerank_top_k = rag.reranker_top_k or rag.retriever.k
            retrieved_docs = rag.reranker.rerank(query, retrieved_docs, top_k=rerank_top_k)

        context_texts = [doc['content'] for doc in retrieved_docs]

        answers = [
            rag.llm.generate_with_context(
                query,
                context_texts,
                temperature=args.temperature
            )
            for _ in range(args.n_samples)
        ]

        embeddings = rag.embedder.embed_batch(answers)
        mean_sim = mean_pairwise_similarity(embeddings)

        abstain = mean_sim < args.agree_threshold
        action = "abstain" if abstain else "none"

        action_counts[action] += 1
        type_counts[qtype] += 1
        if abstain:
            abstain_counts[qtype] += 1

        print(f"{qid} [{qtype}] mean_sim={mean_sim:.3f} action={action}")

    total = sum(type_counts.values())
    total_abstain = sum(abstain_counts.values())

    print("\nSummary")
    print(f"n_samples: {args.n_samples}")
    print(f"temperature: {args.temperature}")
    print(f"agree_threshold: {args.agree_threshold}")
    print(f"total: {total}")
    print(f"abstain: {total_abstain} ({total_abstain / total:.2f})" if total else "abstain: 0")
    print(f"actions: {dict(action_counts)}")
    print(f"by_type: {dict(type_counts)}")
    print(f"abstain_by_type: {dict(abstain_counts)}")


if __name__ == "__main__":
    main()
