"""
Base chunker interface and factory pattern implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from src.core.base_classes import BaseChunker
from src.core.logger import get_logger

logger = get_logger(__name__)


class ChunkerFactory:
    """
    Factory for creating chunker instances
    """

    _chunkers: Dict[str, Type[BaseChunker]] = {}

    @classmethod
    def register(cls, name: str, chunker_class: Type[BaseChunker]) -> None:
        """
        Register a chunker class

        Args:
            name: Chunker name/strategy
            chunker_class: Chunker class
        """
        cls._chunkers[name] = chunker_class
        logger.info(f"Registered chunker: {name}")

    @classmethod
    def create(cls, strategy: str, config: Optional[Dict[str, Any]] = None) -> BaseChunker:
        """
        Create a chunker instance

        Args:
            strategy: Chunking strategy ('semantic', 'fixed_size', etc.)
            config: Configuration dictionary

        Returns:
            Chunker instance

        Raises:
            ValueError: If strategy not found
        """
        if strategy not in cls._chunkers:
            available = ', '.join(cls._chunkers.keys())
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. Available strategies: {available}"
            )

        chunker_class = cls._chunkers[strategy]
        return chunker_class(config)

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available chunking strategies"""
        return list(cls._chunkers.keys())


def register_chunker(name: str):
    """
    Decorator to register a chunker class

    Args:
        name: Chunker name/strategy
    """
    def decorator(chunker_class: Type[BaseChunker]):
        ChunkerFactory.register(name, chunker_class)
        return chunker_class
    return decorator
