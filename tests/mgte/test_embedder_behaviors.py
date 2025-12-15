import numpy as np
import pytest
from unittest.mock import patch

from src.embeddings.mgte_embedder import MGTEEmbedder


def cosine_similarity(vec_a, vec_b) -> float:
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@pytest.fixture(scope="module")
def mgte_embedder():
    from transformers import AutoModel

    original_from_pretrained = AutoModel.from_pretrained

    def patched_from_pretrained(*args, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return original_from_pretrained(*args, **kwargs)

    with patch("transformers.AutoModel.from_pretrained", patched_from_pretrained):
        embedder = MGTEEmbedder(model_name="Alibaba-NLP/gte-multilingual-reranker-base", device="cpu")
    if hasattr(embedder, "model"):
        embedder.model.eval()
    return embedder


def test_mgte_generates_embeddings_for_each_input(mgte_embedder):
    """Sanity check that embeddings are produced per input."""
    texts = [
        "Ankara is the capital of Turkey.",
        "Istanbul hosts the Bosphorus strait.",
        "Tea is popular in Turkey.",
    ]

    embeddings = mgte_embedder.embed_batch(texts)

    assert len(embeddings) == len(texts)
    dim = mgte_embedder.get_dimension()
    assert all(len(vec) == dim for vec in embeddings)


def test_semantic_similarity_prefers_paraphrases(mgte_embedder):
    """Paraphrased sentences stay closer than unrelated text."""
    text_a = "What is the best way to boil pasta?"
    text_b = "How should pasta be cooked properly?"
    text_irrelevant = "Mount Everest is the tallest mountain on Earth."

    emb_a = mgte_embedder.embed_text(text_a)
    emb_b = mgte_embedder.embed_text(text_b)
    emb_irrelevant = mgte_embedder.embed_text(text_irrelevant)

    sim_paraphrase = cosine_similarity(emb_a, emb_b)
    sim_unrelated = cosine_similarity(emb_a, emb_irrelevant)

    assert sim_paraphrase > sim_unrelated


def test_paraphrase_surface_forms_remain_close(mgte_embedder):
    """Strengthened check: multiple paraphrases outrank off-topic text."""
    base = "Flights from Paris to Berlin take about two hours."
    paraphrases = [
        "A trip by air between Paris and Berlin lasts roughly two hours.",
        "It takes around two hours to fly Berlin to Paris.",
    ]
    distraction = "Blue whales communicate with infrasonic calls under water."

    base_emb = mgte_embedder.embed_text(base)
    para_embs = [mgte_embedder.embed_text(text) for text in paraphrases]
    distraction_emb = mgte_embedder.embed_text(distraction)

    para_sims = [cosine_similarity(base_emb, emb) for emb in para_embs]
    distraction_sim = cosine_similarity(base_emb, distraction_emb)

    assert min(para_sims) > distraction_sim


def test_negative_pairs_are_farther_apart(mgte_embedder):
    """Related civic policy texts embed closer than unrelated sports text."""
    policy_a = "The city council approved funding for new bike lanes downtown."
    policy_b = "Lawmakers voted to expand urban cycling infrastructure this year."
    sports = "The championship game ended with a dramatic overtime goal."

    emb_a = mgte_embedder.embed_text(policy_a)
    emb_b = mgte_embedder.embed_text(policy_b)
    emb_sports = mgte_embedder.embed_text(sports)

    sim_policy = cosine_similarity(emb_a, emb_b)
    sim_cross = cosine_similarity(emb_a, emb_sports)

    assert sim_policy > sim_cross


def test_long_context_inputs_produce_embeddings(mgte_embedder):
    """Very long passages still emit non-empty embeddings."""
    long_text = " ".join(["Solar observatories monitor coronal mass ejections"] * 400)

    embedding = mgte_embedder.embed_text(long_text)

    assert len(embedding) == mgte_embedder.get_dimension()
    assert any(abs(value) > 0 for value in embedding)


def test_embeddings_are_deterministic_per_input(mgte_embedder):
    """Repeated encoding of same text yields nearly identical vectors."""
    text = "Health officials recommend vaccines to prevent measles outbreaks."

    first = np.asarray(mgte_embedder.embed_text(text), dtype=float)
    second = np.asarray(mgte_embedder.embed_text(text), dtype=float)

    assert np.allclose(first, second, atol=1e-5, rtol=1e-5)








