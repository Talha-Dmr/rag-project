"""
Plain text document loader implementation.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from src.core.base_classes import BaseDataLoader
from src.dataset.base_loader import register_loader
from src.core.logger import get_logger

logger = get_logger(__name__)


@register_loader("text")
class TextLoader(BaseDataLoader):
    """
    Loader for plain text files (.txt, .md, etc.)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.encoding = self.config.get('encoding', 'utf-8')
        self.split_by = self.config.get('split_by', None)  # None, 'paragraph', 'line'

    def load(self, source: str) -> List[Dict[str, Any]]:
        """
        Load a text file

        Args:
            source: Path to text file

        Returns:
            List of documents with metadata
        """
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"Text file not found: {source}")

        logger.info(f"Loading text file: {source}")

        try:
            with open(source_path, 'r', encoding=self.encoding) as f:
                content = f.read()

            # Split content if requested
            if self.split_by == 'paragraph':
                documents = self._split_by_paragraph(content, source_path)
            elif self.split_by == 'line':
                documents = self._split_by_line(content, source_path)
            else:
                documents = [{
                    'content': content,
                    'metadata': {
                        'source': str(source_path),
                        'loader': 'text',
                        'encoding': self.encoding
                    }
                }]

            logger.info(f"Loaded {len(documents)} document(s) from: {source}")

        except Exception as e:
            logger.error(f"Error loading text file {source}: {e}")
            raise

        return documents

    def _split_by_paragraph(self, content: str, source_path: Path) -> List[Dict[str, Any]]:
        """Split content by paragraphs (double newline)"""
        paragraphs = content.split('\n\n')
        documents = []

        for i, para in enumerate(paragraphs, start=1):
            para = para.strip()
            if para:  # Skip empty paragraphs
                documents.append({
                    'content': para,
                    'metadata': {
                        'source': str(source_path),
                        'paragraph': i,
                        'loader': 'text',
                        'split_by': 'paragraph'
                    }
                })

        return documents

    def _split_by_line(self, content: str, source_path: Path) -> List[Dict[str, Any]]:
        """Split content by lines"""
        lines = content.split('\n')
        documents = []

        for i, line in enumerate(lines, start=1):
            line = line.strip()
            if line:  # Skip empty lines
                documents.append({
                    'content': line,
                    'metadata': {
                        'source': str(source_path),
                        'line': i,
                        'loader': 'text',
                        'split_by': 'line'
                    }
                })

        return documents

    def load_batch(self, sources: List[str]) -> List[Dict[str, Any]]:
        """
        Load multiple text files

        Args:
            sources: List of text file paths

        Returns:
            Combined list of documents from all files
        """
        all_documents = []

        for source in sources:
            try:
                documents = self.load(source)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Failed to load {source}: {e}")
                # Continue with other files

        logger.info(f"Loaded {len(all_documents)} total documents from {len(sources)} files")

        return all_documents
