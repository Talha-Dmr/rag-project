"""
HuggingFace embeddings implementation using sentence-transformers.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.base_classes import BaseEmbedder
from src.embeddings.base_embedder import register_embedder
from src.core.logger import get_logger

logger = get_logger(__name__)


@register_embedder("huggingface")
class HuggingFaceEmbedder(BaseEmbedder):
    """
    Embedder using HuggingFace sentence-transformers models
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model_name = self.config.get('model_name', 'sentence-transformers/all-mpnet-base-v2')
        self.device = self.config.get('device', 'cpu')
        self.cache_folder = self.config.get('cache_folder', None)
        self.batch_size = self.config.get('batch_size', 32)

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")

        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                cache_folder=self.cache_folder
            )
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text

        Args:
            text: Input text

        Returns:
            Embedding vector as list of floats
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        try:
            logger.debug(f"Embedding batch of {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise

    def get_dimension(self) -> int:
        """
        Get embedding dimension

        Returns:
            Dimension of embedding vectors
        """
        return self.model.get_sentence_embedding_dimension()

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(similarity)
