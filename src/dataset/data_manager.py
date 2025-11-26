"""
Data manager for orchestrating dataset loading operations.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from src.dataset.base_loader import DataLoaderFactory
from src.core.logger import get_logger

logger = get_logger(__name__)


class DataManager:
    """
    Orchestrates data loading from various sources

    Automatically detects file types and uses appropriate loaders
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DataManager

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.default_loader_type = self.config.get('default_loader', 'text')

        # File extension to loader type mapping
        self.extension_map = {
            '.pdf': 'pdf',
            '.txt': 'text',
            '.md': 'text',
            '.json': 'json',
            '.jsonl': 'json',
        }

    def load_from_path(
        self,
        path: str,
        loader_type: Optional[str] = None,
        recursive: bool = False,
        file_extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load documents from a file or directory

        Args:
            path: File path or directory path
            loader_type: Specific loader to use (auto-detected if None)
            recursive: If True and path is directory, search recursively
            file_extensions: List of extensions to include (e.g., ['.pdf', '.txt'])

        Returns:
            List of loaded documents
        """
        source_path = Path(path)

        if not source_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        # Handle file
        if source_path.is_file():
            return self._load_file(source_path, loader_type)

        # Handle directory
        elif source_path.is_dir():
            return self._load_directory(source_path, loader_type, recursive, file_extensions)

        else:
            raise ValueError(f"Invalid path: {path}")

    def _load_file(
        self,
        file_path: Path,
        loader_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load a single file"""

        # Auto-detect loader type from extension
        if loader_type is None:
            loader_type = self._detect_loader_type(file_path)

        logger.info(f"Loading file {file_path} with {loader_type} loader")

        # Create loader and load file
        loader = DataLoaderFactory.create(loader_type, self.config.get(loader_type, {}))
        documents = loader.load(str(file_path))

        return documents

    def _load_directory(
        self,
        dir_path: Path,
        loader_type: Optional[str] = None,
        recursive: bool = False,
        file_extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Load all files in a directory"""

        logger.info(f"Loading directory: {dir_path} (recursive={recursive})")

        # Find all files
        if recursive:
            files = list(dir_path.rglob('*'))
        else:
            files = list(dir_path.glob('*'))

        # Filter to actual files and by extension
        files = [
            f for f in files
            if f.is_file() and (
                file_extensions is None or f.suffix.lower() in file_extensions
            )
        ]

        if not files:
            logger.warning(f"No files found in directory: {dir_path}")
            return []

        logger.info(f"Found {len(files)} files to load")

        # Group files by loader type
        files_by_type: Dict[str, List[Path]] = {}

        for file in files:
            ftype = loader_type or self._detect_loader_type(file)

            if ftype not in files_by_type:
                files_by_type[ftype] = []

            files_by_type[ftype].append(file)

        # Load files by type in batches
        all_documents = []

        for ftype, file_list in files_by_type.items():
            logger.info(f"Loading {len(file_list)} {ftype} files")

            try:
                loader = DataLoaderFactory.create(ftype, self.config.get(ftype, {}))
                documents = loader.load_batch([str(f) for f in file_list])
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error loading {ftype} files: {e}")
                # Continue with other file types

        logger.info(f"Loaded {len(all_documents)} total documents from directory")

        return all_documents

    def _detect_loader_type(self, file_path: Path) -> str:
        """
        Detect appropriate loader type from file extension

        Args:
            file_path: File path

        Returns:
            Loader type string
        """
        extension = file_path.suffix.lower()

        if extension in self.extension_map:
            return self.extension_map[extension]

        logger.warning(
            f"Unknown file extension {extension}, using default loader: {self.default_loader_type}"
        )
        return self.default_loader_type

    def load_multiple_paths(
        self,
        paths: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Load documents from multiple paths

        Args:
            paths: List of file or directory paths
            **kwargs: Additional arguments passed to load_from_path

        Returns:
            Combined list of documents
        """
        all_documents = []

        for path in paths:
            try:
                documents = self.load_from_path(path, **kwargs)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error loading from {path}: {e}")
                # Continue with other paths

        logger.info(f"Loaded {len(all_documents)} total documents from {len(paths)} paths")

        return all_documents

    def get_document_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about loaded documents

        Args:
            documents: List of documents

        Returns:
            Statistics dictionary
        """
        stats = {
            'total_documents': len(documents),
            'total_characters': sum(len(doc['content']) for doc in documents),
            'sources': set(),
            'loaders': {}
        }

        for doc in documents:
            metadata = doc.get('metadata', {})

            # Track sources
            source = metadata.get('source')
            if source:
                stats['sources'].add(source)

            # Track loader types
            loader = metadata.get('loader', 'unknown')
            stats['loaders'][loader] = stats['loaders'].get(loader, 0) + 1

        stats['sources'] = list(stats['sources'])
        stats['unique_sources'] = len(stats['sources'])

        return stats
