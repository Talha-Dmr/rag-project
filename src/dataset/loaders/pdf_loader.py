"""
PDF document loader implementation.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from pypdf import PdfReader
from src.core.base_classes import BaseDataLoader
from src.dataset.base_loader import register_loader
from src.core.logger import get_logger

logger = get_logger(__name__)


@register_loader("pdf")
class PDFLoader(BaseDataLoader):
    """
    Loader for PDF documents
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.extract_images = self.config.get('extract_images', False)

    def load(self, source: str) -> List[Dict[str, Any]]:
        """
        Load a PDF file

        Args:
            source: Path to PDF file

        Returns:
            List of documents (one per page) with metadata
        """
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")

        if not source_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {source}")

        logger.info(f"Loading PDF: {source}")

        documents = []

        try:
            reader = PdfReader(str(source_path))

            for page_num, page in enumerate(reader.pages, start=1):
                # Extract text
                text = page.extract_text()

                if text.strip():  # Only include pages with text
                    doc = {
                        'content': text,
                        'metadata': {
                            'source': str(source_path),
                            'page': page_num,
                            'total_pages': len(reader.pages),
                            'loader': 'pdf'
                        }
                    }
                    documents.append(doc)

            logger.info(f"Loaded {len(documents)} pages from PDF: {source}")

        except Exception as e:
            logger.error(f"Error loading PDF {source}: {e}")
            raise

        return documents

    def load_batch(self, sources: List[str]) -> List[Dict[str, Any]]:
        """
        Load multiple PDF files

        Args:
            sources: List of PDF file paths

        Returns:
            Combined list of documents from all PDFs
        """
        all_documents = []

        for source in sources:
            try:
                documents = self.load(source)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to load {source}: {e}")
                # Continue with other files

        logger.info(f"Loaded {len(all_documents)} total pages from {len(sources)} PDFs")

        return all_documents
