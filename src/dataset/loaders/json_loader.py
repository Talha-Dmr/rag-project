"""
JSON document loader implementation.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.core.base_classes import BaseDataLoader
from src.dataset.base_loader import register_loader
from src.core.logger import get_logger

logger = get_logger(__name__)


@register_loader("json")
class JSONLoader(BaseDataLoader):
    """
    Loader for JSON files

    Expects JSON in one of these formats:
    1. Single object: {"field": "value", ...}
    2. Array of objects: [{"field": "value"}, ...]
    3. Object with documents key: {"documents": [{...}, {...}]}
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.content_field = self.config.get('content_field', 'content')
        self.documents_key = self.config.get('documents_key', None)

    def load(self, source: str) -> List[Dict[str, Any]]:
        """
        Load a JSON file

        Args:
            source: Path to JSON file

        Returns:
            List of documents with metadata
        """
        source_path = Path(source)

        if not source_path.exists():
            raise FileNotFoundError(f"JSON file not found: {source}")

        if not source_path.suffix.lower() in ['.json', '.jsonl']:
            raise ValueError(f"File is not JSON: {source}")

        logger.info(f"Loading JSON: {source}")

        try:
            # Handle JSONL (JSON Lines) format
            if source_path.suffix.lower() == '.jsonl':
                return self._load_jsonl(source_path)

            # Regular JSON
            with open(source_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents = self._parse_json_data(data, source_path)

            logger.info(f"Loaded {len(documents)} documents from JSON: {source}")

        except Exception as e:
            logger.error(f"Error loading JSON {source}: {e}")
            raise

        return documents

    def _load_jsonl(self, source_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL (JSON Lines) format"""
        documents = []

        with open(source_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        doc = self._object_to_document(obj, source_path, line_num)
                        documents.append(doc)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")

        return documents

    def _parse_json_data(
        self,
        data: Any,
        source_path: Path
    ) -> List[Dict[str, Any]]:
        """Parse JSON data into documents"""

        # If documents_key specified, extract that field
        if self.documents_key and isinstance(data, dict):
            data = data.get(self.documents_key, data)

        # Handle array of objects
        if isinstance(data, list):
            documents = []
            for i, obj in enumerate(data):
                doc = self._object_to_document(obj, source_path, i)
                documents.append(doc)
            return documents

        # Handle single object
        elif isinstance(data, dict):
            doc = self._object_to_document(data, source_path, 0)
            return [doc]

        else:
            raise ValueError(f"Unsupported JSON structure in {source_path}")

    def _object_to_document(
        self,
        obj: Dict[str, Any],
        source_path: Path,
        index: int
    ) -> Dict[str, Any]:
        """Convert a JSON object to a document"""

        # Extract content field
        if self.content_field in obj:
            content = obj[self.content_field]
        else:
            # If no content field, use entire object as string
            content = json.dumps(obj, ensure_ascii=False)

        # Separate content from metadata
        metadata = {k: v for k, v in obj.items() if k != self.content_field}
        metadata.update({
            'source': str(source_path),
            'index': index,
            'loader': 'json'
        })

        return {
            'content': str(content),
            'metadata': metadata
        }

    def load_batch(self, sources: List[str]) -> List[Dict[str, Any]]:
        """
        Load multiple JSON files

        Args:
            sources: List of JSON file paths

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

        logger.info(f"Loaded {len(all_documents)} total documents from {len(sources)} JSON files")

        return all_documents
