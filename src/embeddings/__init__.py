"""
Embeddings module for generating text embeddings.
"""

from src.embeddings.base_embedder import EmbedderFactory, register_embedder
from src.embeddings.huggingface_embedder import HuggingFaceEmbedder
from src.embeddings.mgte_embedder import MGTEEmbedder

__all__ = [
    'EmbedderFactory',
    'HuggingFaceEmbedder',
    'MGTEEmbedder',
    'register_embedder',
]
