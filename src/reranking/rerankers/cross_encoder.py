"""
Cross-encoder based reranking.
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from src.core.base_classes import BaseReranker
from src.reranking.base_reranker import register_reranker
from src.core.logger import get_logger

logger = get_logger(__name__)


@register_reranker("cross_encoder")
class CrossEncoderReranker(BaseReranker):
    """
    Reranks documents using a cross-encoder model
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model_name = self.config.get(
            'model_name',
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        self.top_k = self.config.get('top_k', 5)
        self.device = self.config.get('device', 'cpu')

        logger.info(f"Loading cross-encoder model: {self.model_name}")

        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Successfully loaded cross-encoder: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder {self.model_name}: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query-document relevance

        Args:
            query: Query text
            documents: List of documents with 'content' field
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            return []

        # Filter valid documents
        valid_docs = [doc for doc in documents if doc.get('content', '').strip()]
        if not valid_docs:
            logger.warning("No valid documents found for Cross-Encoder reranking.")
            return []

        top_k = top_k or self.top_k

        try:
            # Prepare query-document pairs
            pairs = [[query, doc['content']] for doc in valid_docs]

            # Get cross-encoder scores
            logger.debug(f"Reranking {len(valid_docs)} documents")
            scores = self.model.predict(pairs)

            # Update scores and sort
            for i, doc in enumerate(valid_docs):
                # Preserve original score
                doc['original_score'] = doc.get('score', 0.0)
                # Assign new score
                doc['score'] = float(scores[i])

            # Sort by score
            reranked = sorted(
                valid_docs,
                key=lambda x: x['score'],
                reverse=True
            )

            # Return top_k
            result = reranked[:top_k]
            logger.info(f"Reranked to top {len(result)} documents")

            return result

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:top_k]
        