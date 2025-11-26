"""
Semantic chunking strategy.

Splits text based on semantic similarity between sentences.
Groups similar sentences together into coherent chunks.
"""

from typing import List, Dict, Any, Optional
import re
import numpy as np
from src.core.base_classes import BaseChunker
from src.chunking.base_chunker import register_chunker
from src.embeddings.base_embedder import EmbedderFactory
from src.core.logger import get_logger

logger = get_logger(__name__)


@register_chunker("semantic")
class SemanticChunker(BaseChunker):
    """
    Chunks text based on semantic similarity

    Groups sentences with high semantic similarity together
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Semantic chunking parameters
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        self.min_chunk_size = self.config.get('min_chunk_size', 100)
        self.max_chunk_size = self.config.get('max_chunk_size', 1000)

        # Embedder configuration
        embedder_config = self.config.get('embedder', {})
        embedder_type = embedder_config.get('type', 'huggingface')

        logger.info(f"Initializing SemanticChunker with embedder: {embedder_type}")

        try:
            self.embedder = EmbedderFactory.create(embedder_type, embedder_config)
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise

        logger.info(
            f"Semantic chunker initialized: threshold={self.similarity_threshold}, "
            f"min_size={self.min_chunk_size}, max_size={self.max_chunk_size}"
        )

    def chunk(self, text: str) -> List[str]:
        """
        Split text into semantic chunks

        Args:
            text: Input text

        Returns:
            List of semantically coherent chunks
        """
        if not text or not text.strip():
            return []

        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            return [text]

        # Embed all sentences
        try:
            embeddings = self.embedder.embed_batch(sentences)
        except Exception as e:
            logger.warning(f"Failed to embed sentences, falling back to simple chunking: {e}")
            return self._fallback_chunking(text)

        # Group sentences by similarity
        chunks = self._group_by_similarity(sentences, embeddings)

        logger.debug(f"Created {len(chunks)} semantic chunks from {len(sentences)} sentences")

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Simple sentence splitting (can be improved with nltk or spacy)
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-Я])')
        sentences = sentence_endings.split(text)

        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _group_by_similarity(self, sentences: List[str], embeddings: List[List[float]]) -> List[str]:
        """
        Group sentences by semantic similarity

        Args:
            sentences: List of sentences
            embeddings: List of sentence embeddings

        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = [sentences[0]]
        current_chunk_length = len(sentences[0])

        for i in range(1, len(sentences)):
            # Compute similarity between current and previous sentence
            prev_emb = np.array(embeddings[i - 1])
            curr_emb = np.array(embeddings[i])

            similarity = self._cosine_similarity(prev_emb, curr_emb)

            # Check if we should add to current chunk or start new one
            sentence_length = len(sentences[i])
            would_exceed_max = current_chunk_length + sentence_length > self.max_chunk_size

            if similarity >= self.similarity_threshold and not would_exceed_max:
                # Add to current chunk
                current_chunk.append(sentences[i])
                current_chunk_length += sentence_length
            else:
                # Start new chunk
                if current_chunk_length >= self.min_chunk_size or len(chunks) == 0:
                    chunks.append(' '.join(current_chunk))
                else:
                    # Chunk too small, merge with previous if possible
                    if chunks:
                        chunks[-1] += ' ' + ' '.join(current_chunk)
                    else:
                        chunks.append(' '.join(current_chunk))

                current_chunk = [sentences[i]]
                current_chunk_length = sentence_length

        # Add final chunk
        if current_chunk:
            if current_chunk_length >= self.min_chunk_size or len(chunks) == 0:
                chunks.append(' '.join(current_chunk))
            else:
                if chunks:
                    chunks[-1] += ' ' + ' '.join(current_chunk)
                else:
                    chunks.append(' '.join(current_chunk))

        return chunks

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product == 0:
            return 0.0

        return dot_product / norm_product

    def _fallback_chunking(self, text: str) -> List[str]:
        """Fallback to simple chunking if embedding fails"""
        # Simple character-based chunking
        chunk_size = self.max_chunk_size
        chunks = []

        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents semantically

        Args:
            documents: List of document dicts with 'content' field

        Returns:
            List of semantically chunked documents with metadata
        """
        chunked_docs = []

        for doc_idx, document in enumerate(documents):
            content = document.get('content', '')
            metadata = document.get('metadata', {})

            try:
                chunks = self.chunk(content)

                for chunk_idx, chunk in enumerate(chunks):
                    chunked_doc = {
                        'content': chunk,
                        'metadata': {
                            **metadata,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(chunks),
                            'chunking_strategy': 'semantic',
                            'similarity_threshold': self.similarity_threshold,
                            'original_doc_index': doc_idx
                        }
                    }
                    chunked_docs.append(chunked_doc)

            except Exception as e:
                logger.error(f"Error chunking document {doc_idx}: {e}")
                # Add original document as single chunk
                chunked_docs.append({
                    'content': content,
                    'metadata': {
                        **metadata,
                        'chunk_index': 0,
                        'total_chunks': 1,
                        'chunking_strategy': 'semantic_fallback',
                        'original_doc_index': doc_idx,
                        'error': str(e)
                    }
                })

        logger.info(
            f"Semantically chunked {len(documents)} documents into {len(chunked_docs)} chunks"
        )

        return chunked_docs
