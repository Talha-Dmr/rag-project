"""
Fixed-size chunking strategy.

Splits text into chunks of fixed character length with optional overlap.
"""

from typing import List, Dict, Any, Optional
from src.core.base_classes import BaseChunker
from src.chunking.base_chunker import register_chunker
from src.core.logger import get_logger

logger = get_logger(__name__)


@register_chunker("fixed_size")
class FixedSizeChunker(BaseChunker):
    """
    Chunks text into fixed-size pieces

    Simple but effective for many use cases
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.chunk_size = self.config.get('chunk_size', 512)
        self.chunk_overlap = self.config.get('chunk_overlap', 50)

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        logger.info(
            f"Initialized FixedSizeChunker: size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

    def chunk(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap

        logger.debug(f"Split text into {len(chunks)} fixed-size chunks")
        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents

        Args:
            documents: List of document dicts with 'content' field

        Returns:
            List of chunked documents with metadata
        """
        chunked_docs = []

        for doc_idx, document in enumerate(documents):
            content = document.get('content', '')
            metadata = document.get('metadata', {})

            chunks = self.chunk(content)

            for chunk_idx, chunk in enumerate(chunks):
                chunked_doc = {
                    'content': chunk,
                    'metadata': {
                        **metadata,  # Preserve original metadata
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'chunking_strategy': 'fixed_size',
                        'chunk_size': self.chunk_size,
                        'chunk_overlap': self.chunk_overlap,
                        'original_doc_index': doc_idx
                    }
                }
                chunked_docs.append(chunked_doc)

        logger.info(
            f"Chunked {len(documents)} documents into {len(chunked_docs)} fixed-size chunks"
        )

        return chunked_docs
