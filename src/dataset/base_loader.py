"""
Base data loader interface and factory pattern implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Type
from pathlib import Path
from src.core.base_classes import BaseDataLoader
from src.core.logger import get_logger

logger = get_logger(__name__)


class DataLoaderFactory:
    """
    Factory for creating data loader instances
    """

    _loaders: Dict[str, Type[BaseDataLoader]] = {}

    @classmethod
    def register(cls, name: str, loader_class: Type[BaseDataLoader]) -> None:
        """
        Register a data loader class

        Args:
            name: Loader name/type
            loader_class: Loader class
        """
        cls._loaders[name] = loader_class
        logger.info(f"Registered data loader: {name}")

    @classmethod
    def create(cls, loader_type: str, config: Optional[Dict[str, Any]] = None) -> BaseDataLoader:
        """
        Create a data loader instance

        Args:
            loader_type: Type of loader ('pdf', 'text', 'json', etc.)
            config: Configuration dictionary

        Returns:
            Data loader instance

        Raises:
            ValueError: If loader type not found
        """
        if loader_type not in cls._loaders:
            available = ', '.join(cls._loaders.keys())
            raise ValueError(
                f"Unknown loader type: {loader_type}. Available loaders: {available}"
            )

        loader_class = cls._loaders[loader_type]
        return loader_class(config)

    @classmethod
    def get_available_loaders(cls) -> List[str]:
        """Get list of available loader types"""
        return list(cls._loaders.keys())


def register_loader(name: str):
    """
    Decorator to register a data loader class

    Args:
        name: Loader name/type
    """
    def decorator(loader_class: Type[BaseDataLoader]):
        DataLoaderFactory.register(name, loader_class)
        return loader_class
    return decorator
