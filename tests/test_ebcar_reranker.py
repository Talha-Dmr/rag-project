"""Unit tests for EBCAR reranker core helpers."""
from typing import Any, Dict, List

import pytest

from src.core.base_classes import BaseEmbedder
from src.reranking.rerankers.ebcar_reranker import EBCARReranker


class TinyEmbedder(BaseEmbedder):
    """Bag-of-words embedder for deterministic tests."""

    def __init__(self):
        super().__init__()
        self.vocab = ("capital", "turkey", "ankara", "inflation")

    def embed_text(self, text: str):
        tokens = text.lower().split()
        return [tokens.count(term) for term in self.vocab]

    def embed_batch(self, texts: List[str]):
        return [self.embed_text(text) for text in texts]

    def get_dimension(self):
        return len(self.vocab)


def build_reranker(**config) -> EBCARReranker:
    cfg: Dict[str, Any] = {"embedder": TinyEmbedder()}
    cfg.update(config)
    return EBCARReranker(cfg)


def make_doc(content: str, *, doc_id: str, score: float = 0.5) -> Dict[str, Any]:
    return {"content": content, "metadata": {"id": doc_id}, "score": score}


def test_reranker_handles_empty_inputs():
    """Empty document lists return an empty ranking."""
    reranker = build_reranker()
    assert reranker.rerank("query", []) == []


def test_reranker_skips_blank_docs():
    """Documents without text are ignored before scoring."""
    reranker = build_reranker()
    docs = [
        {"content": "   ", "metadata": {"id": "blank"}, "score": 0.1},
        make_doc("Ankara is the capital of Turkey", doc_id="ankara"),
    ]

    ranked = reranker.rerank("What is the capital of Turkey?", docs)

    assert len(ranked) == 1
    assert ranked[0]["metadata"]["id"] == "ankara"


def test_reranker_prioritizes_semantic_alignment():
    """Semantically aligned passages receive the top score."""
    reranker = build_reranker()
    docs = [
        make_doc("Inflation hurts purchasing power", doc_id="inflation", score=0.9),
        make_doc("Ankara is the capital of Turkey", doc_id="ankara", score=0.3),
    ]

    ranked = reranker.rerank("What is the capital of Turkey?", docs)

    assert ranked[0]["metadata"]["id"] == "ankara"
    assert ranked[0]["score"] > ranked[1]["score"]


def test_reranker_preserves_original_score_metadata():
    """Original retriever scores are stored in metadata when present."""
    reranker = build_reranker()
    docs = [make_doc("Ankara is the capital of Turkey", doc_id="ankara", score=0.42)]

    ranked = reranker.rerank("capital?", docs)

    assert ranked[0]["metadata"]["original_score"] == pytest.approx(0.42)
    assert "ebcar_features" in ranked[0]["metadata"]


def test_reranker_respects_top_k_argument():
    """top_k truncates the output list after sorting."""
    reranker = build_reranker()
    docs = [
        make_doc("Ankara is the capital of Turkey", doc_id="ankara"),
        make_doc("Inflation hurts purchasing power", doc_id="inflation"),
    ]

    ranked = reranker.rerank("capital?", docs, top_k=1)

    assert len(ranked) == 1
    assert ranked[0]["metadata"]["id"] == "ankara"
