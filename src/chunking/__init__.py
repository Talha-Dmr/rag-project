"""
Chunking module for splitting documents into chunks.
"""

from src.chunking.base_chunker import ChunkerFactory, register_chunker

# Import strategies to ensure registration
from src.chunking import strategies

__all__ = [
    'ChunkerFactory',
    'register_chunker',
]
