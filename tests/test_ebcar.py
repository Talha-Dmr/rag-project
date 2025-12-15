import pytest

from src.core.base_classes import BaseEmbedder
from src.reranking.rerankers.ebcar_reranker import EBCARReranker


class DummyEmbedder(BaseEmbedder):
    """Deterministic toy embedder for tests."""

    def __init__(self):
        super().__init__()
        self.vocab = ("ankara", "capital", "turkey", "inflation")

    def embed_text(self, text: str):
        tokens = text.lower().split()
        return [tokens.count(term) for term in self.vocab]

    def embed_batch(self, texts):
        return [self.embed_text(text) for text in texts]

    def get_dimension(self):
        return len(self.vocab)


def build_reranker(**config):
    cfg = {"embedder": DummyEmbedder()}
    cfg.update(config)
    return EBCARReranker(cfg)


def test_ebcar_prioritizes_semantic_alignment():
    reranker = build_reranker()
    query = "What is the capital of Turkey?"
    docs = [
        {"content": "Ankara is the capital of Turkey", "metadata": {"id": "ankara"}},
        {"content": "Inflation hurts purchasing power", "metadata": {"id": "inflation"}},
    ]

    ranked = reranker.rerank(query, docs)

    assert [doc["metadata"]["id"] for doc in ranked] == ["ankara", "inflation"]
    top_features = ranked[0]["metadata"]["ebcar_features"]
    assert 0.0 <= top_features["semantic"] <= 1.0


def test_ebcar_uses_retriever_signal_when_semantics_tie():
    reranker = build_reranker()
    docs = [
        {"content": "generic text", "metadata": {"id": 1}, "score": 0.1},
        {"content": "generic text", "metadata": {"id": 2}, "score": 0.9},
    ]

    ranked = reranker.rerank("query", docs)
    assert ranked[0]["metadata"]["id"] == 2
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_ebcar_filters_empty_documents():
    reranker = build_reranker()
    docs = [
        {"content": "   ", "metadata": {"id": 1}},
        {"metadata": {"id": 2}},
    ]

    assert reranker.rerank("query", docs) == []
