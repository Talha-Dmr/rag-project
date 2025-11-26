"""
Base vector store interface and factory pattern.
"""

from typing import Dict, Any, Optional, Type, List
from src.core.base_classes import BaseVectorStore
from src.core.logger import get_logger

logger = get_logger(__name__)


class VectorStoreFactory:
    """Factory for creating vector store instances"""

    _stores: Dict[str, Type[BaseVectorStore]] = {}

    @classmethod
    def register(cls, name: str, store_class: Type[BaseVectorStore]) -> None:
        """Register a vector store class"""
        cls._stores[name] = store_class
        logger.info(f"Registered vector store: {name}")

    @classmethod
    def create(cls, store_type: str, config: Optional[Dict[str, Any]] = None) -> BaseVectorStore:
        """Create a vector store instance"""
        if store_type not in cls._stores:
            available = ', '.join(cls._stores.keys())
            raise ValueError(
                f"Unknown vector store type: {store_type}. Available: {available}"
            )

        store_class = cls._stores[store_type]
        return store_class(config)

    @classmethod
    def get_available_stores(cls) -> List[str]:
        """Get list of available vector stores"""
        return list(cls._stores.keys())


def register_store(name: str):
    """Decorator to register a vector store class"""
    def decorator(store_class: Type[BaseVectorStore]):
        VectorStoreFactory.register(name, store_class)
        return store_class
    return decorator
