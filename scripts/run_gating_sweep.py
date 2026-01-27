#!/usr/bin/env python3
"""
Run a small gating threshold sweep on a single document source.
"""

import argparse
from pathlib import Path

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


def run_sweep(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    rag = RAGPipeline.from_config(config)

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    # Index once
    rag.index_documents(str(source), recursive=False)

    queries = [
        "Bu raporun konusu nedir?",
        "Projede kullanılan ana bileşenler nelerdir?",
        "Bu projede hangi metriklerle değerlendirme yapılmış?"
    ]

    thresholds = [float(t) for t in args.uncertainty_thresholds.split(",")]

    for threshold in thresholds:
        print("\n" + "=" * 80)
        print(f"Sweep: uncertainty_threshold = {threshold}")
        gating_cfg = {
            "enabled": True,
            "strategy": args.strategy,
            "contradiction_rate_threshold": args.contradiction_rate_threshold,
            "contradiction_prob_threshold": args.contradiction_prob_threshold,
            "uncertainty_threshold": threshold,
            "max_retries": args.max_retries,
            "k_multiplier": args.k_multiplier,
            "max_k": args.max_k,
            "abstain_message": args.abstain_message,
        }

        for idx, query in enumerate(queries, 1):
            print("\n" + "-" * 80)
            print(f"[{idx}] Query: {query}")
            result = rag.query(
                query_text=query,
                k=args.k,
                return_context=False,
                detect_hallucinations=True,
                gating=gating_cfg
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
    parser = argparse.ArgumentParser(description="Run gating threshold sweep")
    parser.add_argument("--config", default="gating_demo", help="Config name (without .yaml)")
    parser.add_argument("--source", default="designprojectfinal.pdf", help="Document to index")
    parser.add_argument("--k", type=int, default=5, help="Initial retrieval depth")
    parser.add_argument("--strategy", default="retrieve_more", choices=["retrieve_more", "abstain"])
    parser.add_argument("--contradiction-rate-threshold", type=float, default=0.5)
    parser.add_argument("--contradiction-prob-threshold", type=float, default=0.7)
    parser.add_argument("--uncertainty-thresholds", default="0.4,0.5,0.6")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--k-multiplier", type=float, default=2.0)
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument(
        "--abstain-message",
        default="Bu soruya güvenilir şekilde yanıt veremiyorum."
    )

    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
