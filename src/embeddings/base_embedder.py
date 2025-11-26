"""
Base embedder interface and factory pattern.
"""

from typing import Dict, Any, Optional, Type, List
from src.core.base_classes import BaseEmbedder
from src.core.logger import get_logger

logger = get_logger(__name__)


class EmbedderFactory:
    """Factory for creating embedder instances"""

    _embedders: Dict[str, Type[BaseEmbedder]] = {}

    @classmethod
    def register(cls, name: str, embedder_class: Type[BaseEmbedder]) -> None:
        """Register an embedder class"""
        cls._embedders[name] = embedder_class
        logger.info(f"Registered embedder: {name}")

    @classmethod
    def create(cls, embedder_type: str, config: Optional[Dict[str, Any]] = None) -> BaseEmbedder:
        """Create an embedder instance"""
        if embedder_type not in cls._embedders:
            available = ', '.join(cls._embedders.keys())
            raise ValueError(
                f"Unknown embedder type: {embedder_type}. Available: {available}"
            )

        embedder_class = cls._embedders[embedder_type]
        return embedder_class(config)

    @classmethod
    def get_available_embedders(cls) -> List[str]:
        """Get list of available embedders"""
        return list(cls._embedders.keys())


def register_embedder(name: str):
    """Decorator to register an embedder class"""
    def decorator(embedder_class: Type[BaseEmbedder]):
        EmbedderFactory.register(name, embedder_class)
        return embedder_class
    return decorator
