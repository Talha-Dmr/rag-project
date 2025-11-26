"""
Base reranker interface and factory pattern.
"""

from typing import Dict, Any, Optional, Type, List
from src.core.base_classes import BaseReranker
from src.core.logger import get_logger

logger = get_logger(__name__)


class RerankerFactory:
    """Factory for creating reranker instances"""

    _rerankers: Dict[str, Type[BaseReranker]] = {}

    @classmethod
    def register(cls, name: str, reranker_class: Type[BaseReranker]) -> None:
        """Register a reranker class"""
        cls._rerankers[name] = reranker_class
        logger.info(f"Registered reranker: {name}")

    @classmethod
    def create(cls, reranker_type: str, config: Optional[Dict[str, Any]] = None) -> BaseReranker:
        """Create a reranker instance"""
        if reranker_type not in cls._rerankers:
            available = ', '.join(cls._rerankers.keys())
            raise ValueError(
                f"Unknown reranker type: {reranker_type}. Available: {available}"
            )

        reranker_class = cls._rerankers[reranker_type]
        return reranker_class(config)

    @classmethod
    def get_available_rerankers(cls) -> List[str]:
        """Get list of available rerankers"""
        return list(cls._rerankers.keys())


def register_reranker(name: str):
    """Decorator to register a reranker class"""
    def decorator(reranker_class: Type[BaseReranker]):
        RerankerFactory.register(name, reranker_class)
        return reranker_class
    return decorator
