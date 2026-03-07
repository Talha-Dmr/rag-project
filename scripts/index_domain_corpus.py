#!/usr/bin/env python3
"""
Index a corpus into the vector store defined by a config.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.core.config_loader import load_config
from src.rag.rag_pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Index corpus for a domain config")
    parser.add_argument("--config", required=True, help="Config name (without .yaml)")
    parser.add_argument("--corpus", required=True, help="Path to corpus file/folder")
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete and recreate collection before indexing.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories when corpus is a folder.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help=(
            "Force lightweight indexing mode: disable hallucination detector and use a tiny "
            "CPU LLM to avoid heavyweight model/tokenizer requirements."
        ),
    )
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus path not found: {corpus_path}")

    config = load_config(args.config)

    # Indexing does not require generation or detector inference; keep it lightweight/stable.
    # This also avoids tokenizer/version mismatches on ephemeral environments (e.g., cloud pods).
    index_only = args.index_only or os.getenv("INDEX_ONLY", "1") == "1"
    if index_only:
        llm_cfg = dict(config.get("llm", {}) or {})
        llm_cfg["model_name"] = os.getenv("INDEX_LLM_MODEL", "sshleifer/tiny-gpt2")
        llm_cfg["device"] = os.getenv("INDEX_LLM_DEVICE", "cpu")
        llm_cfg["max_tokens"] = int(os.getenv("INDEX_LLM_MAX_TOKENS", "16"))
        config["llm"] = llm_cfg

        detector_cfg = dict(config.get("hallucination_detector", {}) or {})
        detector_cfg["enabled"] = False
        config["hallucination_detector"] = detector_cfg

    rag = RAGPipeline.from_config(config)

    before = rag.vector_store.get_count()
    if args.reset_collection:
        rag.vector_store.delete_collection()
        before = rag.vector_store.get_count()

    indexed_chunks = rag.index_documents(str(corpus_path), recursive=args.recursive)
    after = rag.vector_store.get_count()

    print(
        f"Indexed chunks={indexed_chunks} | collection_count_before={before} | "
        f"collection_count_after={after}"
    )


if __name__ == "__main__":
    main()
