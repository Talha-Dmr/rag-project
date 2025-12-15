import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.core.base_classes import BaseEmbedder, BaseReranker
from src.core.logger import get_logger
from src.embeddings.base_embedder import EmbedderFactory
from src.reranking.base_reranker import register_reranker

logger = get_logger(__name__)


@register_reranker("ebcar")
class EBCARReranker(BaseReranker):
    """Evidence-Based Context-Aware Reranker.

    Combines semantic similarity with retrieval-time heuristics. The previous
    placeholder attempted to load a checkpoint that does not exist which caused
    import failures during test collection.
    """

    DEFAULT_WEIGHTS = {
        'semantic': 0.6,
        'retriever': 0.2,
        'position': 0.1,
        'evidence': 0.07,
        'length': 0.03,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})

        weights_override = self.config.get('weights') or {}
        self.feature_weights = {**self.DEFAULT_WEIGHTS, **weights_override}
        self.length_normalizer = float(self.config.get('length_normalizer', 120.0))

        # embedder (mGTE embedding için)
        embedder_name = self.config.get("embedder_name", "mgte")
        self.embedder = EmbedderFactory.create(embedder_name)

        self._embedder: Optional[BaseEmbedder] = None
        embedder_candidate = self.config.get('embedder')
        if isinstance(embedder_candidate, BaseEmbedder) or (
            embedder_candidate is not None and hasattr(embedder_candidate, 'embed_text')
        ):
            self._embedder = embedder_candidate  # type: ignore[assignment]

        self.embedder_type = (
            self.config.get('embedder_type')
            or self.config.get('embedder_name')
            or 'huggingface'
        )
        embedder_cfg = self.config.get('embedder_config')
        if embedder_cfg is None and isinstance(embedder_candidate, dict):
            embedder_cfg = embedder_candidate
        self.embedder_config = embedder_cfg or {}

        logger.info(
            'Initialized EBCAR reranker (embedder=%s, owns_embedder=%s)',
            self.embedder_type,
            self._embedder is None,
        )

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        candidates = [doc for doc in documents if self._has_content(doc)]
        if not candidates:
            return []

        embedder = self._get_embedder()
        query_vector = self._to_vector(embedder.embed_text(query))
        doc_vectors = self._embed_documents(embedder, candidates)
        retriever_range = self._score_range(candidates)

        for idx, (doc, doc_vec) in enumerate(zip(candidates, doc_vectors)):
            metadata = doc.setdefault('metadata', {})
            if 'original_score' not in metadata and isinstance(doc.get('score'), (int, float)):
                metadata['original_score'] = float(doc['score'])

            features = self._compute_features(
                doc=doc,
                query_vector=query_vector,
                doc_vector=doc_vec,
                doc_index=idx,
                retriever_range=retriever_range
            )
            final_score = self._combine_features(features)
            doc['rerank_score'] = final_score
            doc['score'] = final_score
            metadata['ebcar_features'] = features

        candidates.sort(key=lambda item: item['score'], reverse=True)
        if top_k:
            candidates = candidates[:top_k]
        return candidates

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------
    def _get_embedder(self) -> BaseEmbedder:
        if self._embedder is None:
            logger.info('Loading embedder for EBCAR: %s', self.embedder_type)
            self._embedder = EmbedderFactory.create(self.embedder_type, self.embedder_config)
        return self._embedder

    def _embed_documents(
        self,
        embedder: BaseEmbedder,
        documents: Sequence[Dict[str, Any]]
    ) -> List[np.ndarray]:
        texts = [doc['content'] for doc in documents]
        vectors: List[np.ndarray] = []

        try:
            batch_vectors = embedder.embed_batch(texts)
        except NotImplementedError:
            batch_vectors = None
        except Exception as exc:
            logger.warning('Batch embedding failed (%s), falling back to per-item encoding', exc)
            batch_vectors = None

        if batch_vectors is not None and len(batch_vectors) == len(texts):
            vectors = [self._to_vector(vec) for vec in batch_vectors]
        else:
            vectors = [self._to_vector(embedder.embed_text(text)) for text in texts]
        return vectors

    @staticmethod
    def _to_vector(raw_vector: Any) -> np.ndarray:
        vector = np.asarray(raw_vector, dtype=float)
        if vector.ndim == 0:
            vector = np.expand_dims(vector, 0)
        return vector

    @staticmethod
    def _has_content(doc: Dict[str, Any]) -> bool:
        content = doc.get('content')
        return isinstance(content, str) and content.strip() != ''

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------
    def _compute_features(
        self,
        doc: Dict[str, Any],
        query_vector: np.ndarray,
        doc_vector: np.ndarray,
        doc_index: int,
        retriever_range: Optional[Sequence[float]],
    ) -> Dict[str, float]:
        base_score = doc.get('score')
        metadata = doc.get('metadata', {})
        features = {
            'semantic': float(self._cosine_similarity(query_vector, doc_vector)),
            'retriever': self._scaled_score(base_score, retriever_range),
            'position': self._position_signal(doc, doc_index),
            'evidence': self._evidence_signal(metadata),
            'length': self._length_signal(doc.get('content', '')),
        }
        return features

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / denom)

    @staticmethod
    def _score_range(documents: Sequence[Dict[str, Any]]) -> Optional[Sequence[float]]:
        scores = [float(doc['score']) for doc in documents if isinstance(doc.get('score'), (int, float))]
        if not scores:
            return None
        return min(scores), max(scores)

    @staticmethod
    def _scaled_score(value: Any, value_range: Optional[Sequence[float]]) -> float:
        if not isinstance(value, (int, float)):
            return 0.0
        if not value_range:
            return float(value)
        low, high = value_range
        if math.isclose(low, high):
            return 1.0 if value > low else 0.0
        return float((value - low) / (high - low))

    @staticmethod
    def _position_signal(doc: Dict[str, Any], fallback_index: int) -> float:
        raw_pos = doc.get('position')
        if raw_pos is None:
            raw_pos = doc.get('metadata', {}).get('position')
        if raw_pos is None:
            raw_pos = fallback_index
        raw_pos = max(int(raw_pos), 0)
        return 1.0 / math.log2(3 + raw_pos)

    def _evidence_signal(self, metadata: Dict[str, Any]) -> float:
        evidence_count = metadata.get('evidence_count')
        if evidence_count is None:
            supporting = metadata.get('supporting_facts')
            if isinstance(supporting, Sequence) and not isinstance(supporting, (str, bytes)):
                evidence_count = len(supporting)
        if not isinstance(evidence_count, (int, float)):
            confidence = metadata.get('confidence')
            if isinstance(confidence, (int, float)):
                return max(float(confidence), 0.0)
            evidence_count = 0
        return math.log1p(float(evidence_count))

    def _length_signal(self, text: str) -> float:
        tokens = len(text.split())
        if tokens == 0:
            return 0.0
        return 1.0 - math.exp(-tokens / self.length_normalizer)

    def _combine_features(self, features: Dict[str, float]) -> float:
        score = 0.0
        for name, weight in self.feature_weights.items():
            score += weight * features.get(name, 0.0)
        return score
