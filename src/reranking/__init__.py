"""
Reranking module for improving retrieval quality.
"""

from src.reranking.base_reranker import RerankerFactory, register_reranker

# Import rerankers to ensure registration
from src.reranking import rerankers

__all__ = [
    'RerankerFactory',
    'register_reranker',
]
