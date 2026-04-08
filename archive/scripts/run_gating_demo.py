#!/usr/bin/env python3
"""
Run a small RAG gating demo with hallucination-based uncertainty.
"""

import argparse
from pathlib import Path

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


def run_demo(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    rag = RAGPipeline.from_config(config)

    if args.index:
        source = Path(args.source)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        if args.reset:
            rag.vector_store.delete_collection()
        rag.index_documents(str(source), recursive=False)

    queries = args.queries or [
        "When was the last time Brazil won the World Cup?",
        "Who has the most blocks in the NBA?",
        "Where did the story of The Sound of Music take place?"
    ]

    for idx, query in enumerate(queries, 1):
        print("\n" + "=" * 80)
        print(f"[{idx}] Query: {query}")
        result = rag.query(
            query_text=query,
            k=args.k,
            return_context=False,
            detect_hallucinations=True
        )

        print(f"Answer: {result.get('answer')}")
        print(f"Docs retrieved: {result.get('num_docs_retrieved')}")

        gating = result.get("gating")
        if gating:
            print("Gating:")
            print(f"  action: {gating.get('action')}")
            print(f"  attempts: {gating.get('attempts')}")
            print(f"  k_used: {gating.get('k_used')}")
            print(f"  stats: {gating.get('stats')}")
        else:
            print("Gating: not enabled")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG gating demo")
    parser.add_argument(
        "--config",
        default="gating_demo",
        help="Config name (without .yaml)"
    )
    parser.add_argument(
        "--source",
        default="",
        help="Path to a document file to index (use with --index)"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index the --source file before running queries"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete collection before indexing"
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=None,
        help="Optional list of queries to run"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Initial retrieval depth"
    )

    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
