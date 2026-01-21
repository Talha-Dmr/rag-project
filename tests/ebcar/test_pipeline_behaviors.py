import statistics
from typing import Any, Dict, List, Optional

import pytest

from src.core.base_classes import BaseEmbedder
from src.reranking.rerankers.ebcar_reranker import EBCARReranker


class ThematicToyEmbedder(BaseEmbedder):
    """Bag-of-topics embedder mirroring coarse semantic buckets in tests."""

    TOPICS = [
        {"capital", "turkey", "ankara", "where"},
        {"inflation", "purchasing", "power"},
        {"helios", "accord", "vote", "assembly", "treaty", "senate", "record", "consensus"},
        {"world", "cup", "france", "moscow", "final", "winner"},
        {"jaguar", "i-pace", "charge", "charger", "battery", "car"},
        {"jaguar", "animal", "rainforest", "pounce", "sprint", "wildlife"},
        {"solar", "storm", "storms", "space", "satellite", "flare", "geomagnetic"},
        {"garden", "roses", "compost", "herbs"},
        {"insulin", "banting", "best", "hormone", "patients", "discovered"},
    ]

    def embed_text(self, text: str):
        tokens = [token for token in text.lower().split() if token]
        topic_counts = [sum(token in vocab for token in tokens) for vocab in self.TOPICS]
        return topic_counts

    def embed_batch(self, texts: List[str]):
        return [self.embed_text(text) for text in texts]

    def get_dimension(self):
        return len(self.TOPICS)


def build_reranker(**config) -> EBCARReranker:
    cfg = {"embedder": ThematicToyEmbedder(), "length_normalizer": 60}
    cfg.update(config)
    return EBCARReranker(cfg)


def make_doc(
    content: str,
    *,
    doc_id: str,
    score: float = 0.5,
    position: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    combined_meta = {"id": doc_id}
    if metadata:
        combined_meta.update(metadata)
    doc = {"content": content, "metadata": combined_meta, "score": score}
    if position is not None:
        doc["position"] = position
    return doc


@pytest.fixture
def reranker():
    return build_reranker()


def test_ebcar_sanity_outputs_floats(reranker):
    """Basic smoke test: reranker returns float scores matching candidate count."""
    query = "Where is the capital of Turkey located?"
    docs = [
        make_doc("Ankara is the capital of Turkey", doc_id="ankara", score=0.8),
        make_doc("Inflation hurts purchasing power everywhere", doc_id="inflation", score=0.3),
        make_doc("   ", doc_id="blank"),
    ]

    ranked = reranker.rerank(query, docs)

    assert len(ranked) == 2
    assert all(isinstance(item["score"], float) for item in ranked)
    assert ranked[0]["score"] >= ranked[1]["score"]


def test_conflict_sensitive_documents_get_penalized(reranker):
    """Contradictory passages with low confidence aggregate below official record."""
    query = "Did the Helios Accord pass the assembly vote?"
    docs = [
        make_doc(
            "The senate approved the Helios Accord in a late vote",
            doc_id="passed",
            score=0.35,
            metadata={"confidence": 0.2},
        ),
        make_doc(
            "The Helios Accord was rejected and sent back to committee",
            doc_id="rejected",
            score=0.33,
            metadata={"confidence": 0.1},
        ),
        make_doc(
            "The official record shows the measure was delayed until consensus formed",
            doc_id="record",
            score=0.9,
            metadata={"evidence_count": 3},
        ),
    ]

    ranked = reranker.rerank(query, docs)
    top = ranked[0]
    conflict_scores = [item["score"] for item in ranked if item["metadata"]["id"] in {"passed", "rejected"}]

    assert top["metadata"]["id"] == "record"
    assert len(conflict_scores) == 2
    assert statistics.fmean(conflict_scores) < top["score"]


def test_evidence_synergy_rewards_complements(reranker):
    """Complementary passages with shared evidence outrank misleading singletons."""
    query = "Who won the 2018 World Cup final and where was it played?"
    docs = [
        make_doc(
            "France won the 2018 World Cup final",
            doc_id="winner",
            score=0.45,
            metadata={"supporting_facts": ["winner", "venue"]},
        ),
        make_doc(
            "The final was played in Moscow Luzhniki Stadium",
            doc_id="venue",
            score=0.46,
            metadata={"supporting_facts": ["winner", "venue"]},
        ),
        make_doc(
            "Croatia lifted the trophy in Madrid shocking commentators",
            doc_id="misinfo",
            score=0.7,
            metadata={"confidence": 0.05},
        ),
    ]

    ranked = reranker.rerank(query, docs)
    top_pair = {ranked[0]["metadata"]["id"], ranked[1]["metadata"]["id"]}

    assert top_pair == {"winner", "venue"}
    assert ranked[2]["metadata"]["id"] == "misinfo"
    assert ranked[1]["score"] > ranked[2]["score"]


def test_ambiguous_query_keeps_multiple_interpretations(reranker):
    """Ambiguous queries keep both valid readings above unrelated passages."""
    query = "How fast can a jaguar charge?"
    docs = [
        make_doc(
            "Jaguar I-Pace batteries can charge overnight on a home charger",
            doc_id="ev",
            score=0.6,
        ),
        make_doc(
            "A jaguar can charge through the rainforest in short sprints",
            doc_id="animal",
            score=0.58,
        ),
        make_doc(
            "Garden roses thrive when compost is added each spring",
            doc_id="garden",
            score=0.55,
            metadata={"confidence": 0.2},
        ),
    ]

    ranked = reranker.rerank(query, docs)
    top_two = {ranked[0]["metadata"]["id"], ranked[1]["metadata"]["id"]}

    assert top_two == {"ev", "animal"}
    assert ranked[2]["metadata"]["id"] == "garden"


def test_noise_robustness_keeps_irrelevant_low(reranker):
    """Irrelevant passages stay below relevant ones despite noisy additions."""
    query = "What warnings do satellites issue about solar storms?"
    docs = [
        make_doc(
            "Space weather centers issue solar storm alerts to satellite operators",
            doc_id="alerts",
            score=0.75,
            metadata={"evidence_count": 2},
        ),
        make_doc(
            "Engineers reroute communications during strong geomagnetic flares",
            doc_id="mitigation",
            score=0.7,
        ),
    ]
    noise_docs = [
        make_doc(
            f"Herbs and roses prefer compost batch {idx}",
            doc_id=f"noise-{idx}",
            score=0.2 + idx * 0.01,
            metadata={"confidence": 0.1},
        )
        for idx in range(3)
    ]
    ranked = reranker.rerank(query, docs + noise_docs)

    assert [item["metadata"]["id"] for item in ranked[:2]] == ["alerts", "mitigation"]
    assert all(item["metadata"]["id"].startswith("noise") for item in ranked[2:])


def test_scaling_preserves_top_relevance(reranker):
    """A single clearly relevant passage stays top-ranked as noise scales."""
    query = "Who discovered insulin?"
    relevant = make_doc(
        "Frederick Banting and Charles Best discovered insulin saving patients",
        doc_id="insulin",
        score=0.95,
        metadata={"evidence_count": 4},
    )
    noise_docs = [
        make_doc(
            f"Background gardening note {idx}",
            doc_id=f"garden-{idx}",
            score=0.3,
            metadata={"confidence": 0.1},
        )
        for idx in range(20)
    ]
    ranked = reranker.rerank(query, [relevant] + noise_docs)

    assert ranked[0]["metadata"]["id"] == "insulin"
    assert all(item["metadata"]["id"].startswith("garden-") for item in ranked[1:])


