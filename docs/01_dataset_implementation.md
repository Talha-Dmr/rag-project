# Dataset Implementation Documentation

## Overview

The Dataset module provides a flexible and extensible system for loading documents from various sources. It supports multiple file formats and uses a factory pattern for easy extension.

## Architecture

### Components

1. **BaseDataLoader**: Abstract interface for all data loaders
2. **DataLoaderFactory**: Factory for creating and registering loaders
3. **Concrete Loaders**: PDF, Text, JSON implementations
4. **DataManager**: High-level orchestration and auto-detection

### Design Patterns

- **Abstract Factory Pattern**: `DataLoaderFactory` creates appropriate loaders
- **Strategy Pattern**: Different loading strategies for different file types
- **Decorator Pattern**: `@register_loader` for automatic registration

## Supported Formats

### 1. PDF Documents

```python
from src.dataset import DataLoaderFactory

# Create PDF loader
loader = DataLoaderFactory.create('pdf')

# Load single PDF
documents = loader.load('path/to/document.pdf')

# Load multiple PDFs
documents = loader.load_batch(['doc1.pdf', 'doc2.pdf'])
```

**Features**:
- Extracts text from each page
- Preserves page metadata
- Handles multi-page documents

**Configuration**:
```yaml
pdf:
  extract_images: false  # Future: extract images
```

### 2. Text Files

```python
# Create text loader
loader = DataLoaderFactory.create('text')

# Load entire file
documents = loader.load('document.txt')

# Load with paragraph splitting
loader = DataLoaderFactory.create('text', {'split_by': 'paragraph'})
documents = loader.load('document.txt')
```

**Supported Extensions**: `.txt`, `.md`

**Configuration**:
```yaml
text:
  encoding: utf-8
  split_by: null  # Options: null, 'paragraph', 'line'
```

### 3. JSON Files

```python
# Create JSON loader
loader = DataLoaderFactory.create('json', {
    'content_field': 'text',
    'documents_key': 'data'
})

documents = loader.load('data.json')
```

**Supported Formats**:
- Single object: `{"content": "text", ...}`
- Array: `[{...}, {...}]`
- Nested: `{"documents": [{...}]}`
- JSONL: One JSON object per line

**Configuration**:
```yaml
json:
  content_field: content  # Field containing main text
  documents_key: null     # Key to extract array from nested JSON
```

## Using DataManager

The `DataManager` provides high-level functionality:

### Auto-detection

```python
from src.dataset import DataManager

manager = DataManager()

# Auto-detects file type from extension
documents = manager.load_from_path('document.pdf')
```

### Directory Loading

```python
# Load all files in directory
documents = manager.load_from_path(
    'path/to/directory',
    recursive=True,
    file_extensions=['.pdf', '.txt']
)
```

### Batch Loading

```python
# Load from multiple paths
paths = ['doc1.pdf', 'doc2.txt', 'data.json']
documents = manager.load_multiple_paths(paths)

# Get statistics
stats = manager.get_document_stats(documents)
print(stats)
# {
#     'total_documents': 150,
#     'total_characters': 50000,
#     'unique_sources': 3,
#     'loaders': {'pdf': 100, 'text': 30, 'json': 20}
# }
```

## Document Structure

All loaders return documents in a consistent format:

```python
{
    'content': str,      # Main document text
    'metadata': {
        'source': str,   # File path
        'loader': str,   # Loader type
        # Loader-specific metadata
        'page': int,     # PDF: page number
        'index': int,    # JSON: array index
        # ... etc
    }
}
```

## Creating Custom Loaders

### Step 1: Implement BaseDataLoader

```python
from src.core.base_classes import BaseDataLoader
from src.dataset.base_loader import register_loader

@register_loader("custom")
class CustomLoader(BaseDataLoader):
    def __init__(self, config=None):
        super().__init__(config)
        # Custom initialization

    def load(self, source: str):
        # Load from source
        documents = []
        # ... loading logic ...
        return documents

    def load_batch(self, sources: List[str]):
        # Batch loading logic
        all_docs = []
        for source in sources:
            all_docs.extend(self.load(source))
        return all_docs
```

### Step 2: Import in __init__.py

```python
# src/dataset/loaders/__init__.py
from src.dataset.loaders.custom_loader import CustomLoader
```

### Step 3: Use Your Loader

```python
from src.dataset import DataLoaderFactory

loader = DataLoaderFactory.create('custom', config)
documents = loader.load('path/to/file')
```

## Configuration

Example `config/base_config.yaml`:

```yaml
data_loader:
  type: "pdf"           # Default loader type
  batch_size: 10

  # Loader-specific configs
  pdf:
    extract_images: false

  text:
    encoding: utf-8
    split_by: null

  json:
    content_field: content
    documents_key: null
```

## Error Handling

All loaders handle errors gracefully:

```python
try:
    documents = loader.load('missing.pdf')
except FileNotFoundError:
    print("File not found")

# DataManager continues on errors
documents = manager.load_multiple_paths(['file1.pdf', 'missing.pdf'])
# Logs error for missing.pdf but returns documents from file1.pdf
```

## Best Practices

1. **Use DataManager for flexibility**: Auto-detection and error handling
2. **Configure via YAML**: Keep code clean, configs separate
3. **Check document stats**: Validate loaded data
4. **Handle metadata**: Use metadata for filtering and tracking
5. **Batch when possible**: More efficient than individual loads

## Performance Tips

1. **Batch Loading**: Use `load_batch()` for multiple files
2. **Filter Extensions**: Specify `file_extensions` when loading directories
3. **Lazy Loading**: Load documents on-demand if dataset is large
4. **Caching**: Cache loaded documents if reusing

## Integration with Other Modules

```python
from src.dataset import DataManager
from src.chunking import ChunkerFactory

# Load documents
manager = DataManager()
documents = manager.load_from_path('data/')

# Pass to chunking
chunker = ChunkerFactory.create('semantic')
chunks = chunker.chunk_documents(documents)
```

## Troubleshooting

### Issue: "Unknown loader type"
**Solution**: Check if loader is registered. Import the loader module.

### Issue: Encoding errors
**Solution**: Specify encoding in text loader config: `{'encoding': 'latin-1'}`

### Issue: No documents loaded
**Solution**: Check file extensions match, verify files exist, check logs for errors

## API Reference

### DataLoaderFactory

```python
DataLoaderFactory.register(name: str, loader_class: Type)
DataLoaderFactory.create(loader_type: str, config: Dict) -> BaseDataLoader
DataLoaderFactory.get_available_loaders() -> List[str]
```

### DataManager

```python
DataManager(config: Dict)
load_from_path(path, loader_type, recursive, file_extensions) -> List[Dict]
load_multiple_paths(paths, **kwargs) -> List[Dict]
get_document_stats(documents) -> Dict
```

### BaseDataLoader

```python
load(source: str) -> List[Dict[str, Any]]
load_batch(sources: List[str]) -> List[Dict[str, Any]]
```

## Examples

See `examples/dataset_examples.py` for complete working examples.
