"""
Vector stores module for similarity search.
"""

from src.vector_stores.base_store import VectorStoreFactory, register_store

# Import stores to ensure registration
from src.vector_stores import stores

__all__ = [
    'VectorStoreFactory',
    'register_store',
]
