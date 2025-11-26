"""
Dataset module for loading documents from various sources.
"""

from src.dataset.base_loader import DataLoaderFactory, register_loader
from src.dataset.data_manager import DataManager

# Import loaders to ensure registration
from src.dataset import loaders

__all__ = [
    'DataLoaderFactory',
    'DataManager',
    'register_loader',
]
