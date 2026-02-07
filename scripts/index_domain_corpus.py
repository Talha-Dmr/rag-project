#!/usr/bin/env python3
"""
Index a corpus into the vector store defined by a config.
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus path not found: {corpus_path}")

    config = load_config(args.config)
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
