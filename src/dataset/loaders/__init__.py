"""
Document loaders for various file formats.
"""

# Import loaders to trigger registration
from src.dataset.loaders.pdf_loader import PDFLoader
from src.dataset.loaders.text_loader import TextLoader
from src.dataset.loaders.json_loader import JSONLoader

__all__ = [
    'PDFLoader',
    'TextLoader',
    'JSONLoader',
]
