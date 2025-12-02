"""
BM25 based reranking.
"""

from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from src.core.base_classes import BaseReranker
from src.reranking.base_reranker import register_reranker
from src.core.logger import get_logger

logger = get_logger(__name__)


@register_reranker("bm25")
class BM25Reranker(BaseReranker):
    """
    Reranks documents using BM25 algorithm
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.top_k = self.config.get('top_k', 5)
        logger.info("Initialized BM25 reranker")

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using BM25

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
            logger.warning("No valid documents found for BM25 reranking.")
            return []

        top_k = top_k or self.top_k

        try:
            # Tokenize documents
            logger.debug(f"Tokenizing {len(valid_docs)} documents for BM25")
            tokenized_docs = [
                doc['content'].lower().split()
                for doc in valid_docs
            ]

            # Create BM25 index
            bm25 = BM25Okapi(tokenized_docs)

            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores
            scores = bm25.get_scores(tokenized_query)

            # Update scores
            for i, doc in enumerate(valid_docs):
                # Preserve original score
                doc['original_score'] = doc.get('score', 0.0)
                # Assign new score
                doc['score'] = float(scores[i])

            # Sort by BM25 score
            reranked = sorted(
                valid_docs,
                key=lambda x: x['score'],
                reverse=True
            )

            # Return top_k
            result = reranked[:top_k]
            logger.info(f"BM25 reranked to top {len(result)} documents")

            return result

        except Exception as e:
            logger.error(f"Error during BM25 reranking: {e}")
            return documents[:top_k]
        