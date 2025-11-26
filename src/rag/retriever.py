"""
Retriever for RAG system.
"""

from typing import List, Dict, Any, Optional
from src.embeddings.base_embedder import EmbedderFactory
from src.vector_stores.base_store import VectorStoreFactory
from src.core.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """
    Retrieves relevant documents for a query
    """

    def __init__(
        self,
        embedder,
        vector_store,
        k: int = 10,
        score_threshold: float = 0.0
    ):
        """
        Initialize Retriever

        Args:
            embedder: Embedder instance
            vector_store: Vector store instance
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.k = k
        self.score_threshold = score_threshold

        logger.info(f"Initialized Retriever: k={k}, threshold={score_threshold}")

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query

        Args:
            query: Query text
            k: Number of documents (overrides default)
            filter_dict: Optional metadata filters

        Returns:
            List of documents with scores
        """
        k = k or self.k

        try:
            # Embed query
            logger.debug(f"Embedding query: {query[:100]}...")
            query_embedding = self.embedder.embed_text(query)

            # Search vector store
            logger.debug(f"Searching for top {k} documents")
            results = self.vector_store.search(
                query_embedding=query_embedding,
                k=k,
                filter_dict=filter_dict
            )

            # Filter by score threshold
            filtered_results = [
                doc for doc in results
                if doc.get('score', 0) >= self.score_threshold
            ]

            logger.info(
                f"Retrieved {len(filtered_results)}/{len(results)} documents "
                f"above threshold {self.score_threshold}"
            )

            return filtered_results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Retriever':
        """
        Create Retriever from configuration

        Args:
            config: Configuration dictionary

        Returns:
            Retriever instance
        """
        # Create embedder
        embedder_config = config.get('embeddings', {})
        embedder = EmbedderFactory.create('huggingface', embedder_config)

        # Create vector store
        vector_store_config = config.get('vector_store', {})
        store_type = vector_store_config.get('type', 'chroma')
        vector_store = VectorStoreFactory.create(store_type, vector_store_config.get('config', {}))

        # Retrieval params
        retrieval_config = config.get('retrieval', {})
        k = retrieval_config.get('k', 10)
        score_threshold = retrieval_config.get('score_threshold', 0.0)

        return cls(
            embedder=embedder,
            vector_store=vector_store,
            k=k,
            score_threshold=score_threshold
        )
