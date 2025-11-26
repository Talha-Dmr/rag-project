"""
Embeddings module for generating text embeddings.
"""

from src.embeddings.base_embedder import EmbedderFactory, register_embedder
from src.embeddings.huggingface_embedder import HuggingFaceEmbedder

__all__ = [
    'EmbedderFactory',
    'HuggingFaceEmbedder',
    'register_embedder',
]
