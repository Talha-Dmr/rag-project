# Chunking Implementation Documentation

## Overview

The Chunking module splits documents into smaller, semantically meaningful pieces. It supports multiple chunking strategies and uses a factory pattern for extensibility.

## Why Chunking?

Chunking is essential for RAG systems because:
1. **Context Window Limits**: LLMs have token limits
2. **Better Retrieval**: Smaller chunks improve search precision
3. **Semantic Coherence**: Groups related content together
4. **Performance**: Reduces processing time per query

## Architecture

### Components

1. **BaseChunker**: Abstract interface for chunking strategies
2. **ChunkerFactory**: Factory for creating and registering chunkers
3. **Chunking Strategies**: Different algorithms for text splitting
   - **FixedSizeChunker**: Simple character-based splitting
   - **SemanticChunker**: Embedding-based semantic splitting

### Design Patterns

- **Strategy Pattern**: Different chunking algorithms
- **Factory Pattern**: Easy creation and registration
- **Decorator Pattern**: `@register_chunker` for auto-registration

## Available Strategies

### 1. Fixed-Size Chunking

**Best for**: Simple use cases, consistent chunk sizes

```python
from src.chunking import ChunkerFactory

# Create fixed-size chunker
chunker = ChunkerFactory.create('fixed_size', {
    'chunk_size': 512,
    'chunk_overlap': 50
})

# Chunk single text
text = "Your long document text here..."
chunks = chunker.chunk(text)

# Chunk documents
documents = [{'content': '...', 'metadata': {...}}]
chunked_docs = chunker.chunk_documents(documents)
```

**Parameters**:
- `chunk_size`: Characters per chunk (default: 512)
- `chunk_overlap`: Overlapping characters between chunks (default: 50)

**Pros**:
- Fast and simple
- Predictable chunk sizes
- No dependencies on ML models

**Cons**:
- May split sentences/paragraphs awkwardly
- Doesn't consider semantic meaning

### 2. Semantic Chunking

**Best for**: Maintaining semantic coherence, high-quality retrieval

```python
from src.chunking import ChunkerFactory

# Create semantic chunker
chunker = ChunkerFactory.create('semantic', {
    'similarity_threshold': 0.7,
    'min_chunk_size': 100,
    'max_chunk_size': 1000,
    'embedder': {
        'type': 'huggingface',
        'model_name': 'sentence-transformers/all-mpnet-base-v2',
        'device': 'cpu'
    }
})

# Chunk text
chunks = chunker.chunk(text)
```

**How It Works**:
1. Splits text into sentences
2. Generates embeddings for each sentence
3. Computes similarity between consecutive sentences
4. Groups similar sentences together
5. Respects min/max chunk size constraints

**Parameters**:
- `similarity_threshold`: Minimum similarity to group sentences (0-1, default: 0.7)
- `min_chunk_size`: Minimum characters per chunk (default: 100)
- `max_chunk_size`: Maximum characters per chunk (default: 1000)
- `embedder`: Embedder configuration (see below)

**Pros**:
- Maintains semantic coherence
- Natural break points
- Better for retrieval quality

**Cons**:
- Slower than fixed-size (requires embeddings)
- Variable chunk sizes
- Requires embedding model

## Embedder Configuration

Semantic chunking requires an embedder:

```yaml
chunking:
  strategy: semantic
  config:
    similarity_threshold: 0.7
    min_chunk_size: 100
    max_chunk_size: 1000
    embedder:
      type: huggingface
      model_name: sentence-transformers/all-mpnet-base-v2
      device: cpu  # or 'cuda' for GPU
      cache_folder: ./models/embeddings
      batch_size: 32
```

**Recommended Models**:
- `all-mpnet-base-v2`: Best quality, 768 dimensions
- `all-MiniLM-L6-v2`: Fast, 384 dimensions
- `multi-qa-mpnet-base-dot-v1`: Good for Q&A

## Document Structure

Input documents:
```python
{
    'content': str,  # Text to chunk
    'metadata': {...}  # Preserved in chunks
}
```

Output chunks:
```python
{
    'content': str,  # Chunk text
    'metadata': {
        ...  # Original metadata
        'chunk_index': 0,
        'total_chunks': 5,
        'chunking_strategy': 'semantic',
        'original_doc_index': 0,
        # Strategy-specific metadata
    }
}
```

## Usage Examples

### Basic Usage

```python
from src.dataset import DataManager
from src.chunking import ChunkerFactory

# Load documents
manager = DataManager()
documents = manager.load_from_path('documents/')

# Create chunker
chunker = ChunkerFactory.create('semantic')

# Chunk documents
chunks = chunker.chunk_documents(documents)

print(f"Created {len(chunks)} chunks from {len(documents)} documents")
```

### With Configuration File

```python
from src.core.config_loader import load_config
from src.chunking import ChunkerFactory

# Load config
config = load_config('base_config')
chunking_config = config['chunking']

# Create chunker from config
chunker = ChunkerFactory.create(
    chunking_config['strategy'],
    chunking_config['config']
)

chunks = chunker.chunk_documents(documents)
```

### Comparing Strategies

```python
from src.chunking import ChunkerFactory

text = "Your test document..."

# Fixed-size chunking
fixed_chunker = ChunkerFactory.create('fixed_size')
fixed_chunks = fixed_chunker.chunk(text)

# Semantic chunking
semantic_chunker = ChunkerFactory.create('semantic')
semantic_chunks = semantic_chunker.chunk(text)

print(f"Fixed-size: {len(fixed_chunks)} chunks")
print(f"Semantic: {len(semantic_chunks)} chunks")
```

## Creating Custom Chunking Strategies

### Step 1: Implement BaseChunker

```python
from src.core.base_classes import BaseChunker
from src.chunking.base_chunker import register_chunker

@register_chunker("custom")
class CustomChunker(BaseChunker):
    def __init__(self, config=None):
        super().__init__(config)
        # Custom initialization

    def chunk(self, text: str) -> List[str]:
        # Your chunking logic
        chunks = []
        # ... split text ...
        return chunks

    def chunk_documents(self, documents):
        # Chunk multiple documents
        chunked_docs = []
        for doc in documents:
            chunks = self.chunk(doc['content'])
            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    'content': chunk,
                    'metadata': {
                        **doc['metadata'],
                        'chunk_index': i,
                        'chunking_strategy': 'custom'
                    }
                })
        return chunked_docs
```

### Step 2: Register and Use

```python
# Import to register
from src.chunking.strategies.custom import CustomChunker

# Use it
chunker = ChunkerFactory.create('custom', config)
```

## Configuration Examples

### config/base_config.yaml

```yaml
chunking:
  # Choose strategy: fixed_size, semantic
  strategy: semantic

  config:
    # Semantic chunking settings
    similarity_threshold: 0.7
    min_chunk_size: 100
    max_chunk_size: 1000

    # Embedder for semantic chunking
    embedder:
      type: huggingface
      model_name: sentence-transformers/all-mpnet-base-v2
      device: ${DEVICE:cpu}
      cache_folder: ${EMBEDDING_CACHE_DIR:./models/embeddings}
      batch_size: 32
```

### Fixed-Size Config

```yaml
chunking:
  strategy: fixed_size
  config:
    chunk_size: 512
    chunk_overlap: 50
```

## Performance Considerations

### Fixed-Size Chunking
- **Speed**: Very fast, O(n) complexity
- **Memory**: Minimal
- **Best for**: Large datasets, simple use cases

### Semantic Chunking
- **Speed**: Slower, requires embedding generation
- **Memory**: Higher (stores sentence embeddings)
- **Optimization**:
  - Use GPU if available
  - Use smaller embedding models for speed
  - Batch process documents
- **Best for**: Quality-focused applications

## Best Practices

1. **Choose Strategy Based on Use Case**:
   - Simple Q&A: Fixed-size often sufficient
   - Complex reasoning: Semantic chunking better

2. **Tune Parameters**:
   - Start with defaults
   - Adjust based on document type
   - Test retrieval quality

3. **Monitor Chunk Statistics**:
   ```python
   chunks = chunker.chunk_documents(docs)

   sizes = [len(c['content']) for c in chunks]
   print(f"Avg size: {sum(sizes)/len(sizes):.0f}")
   print(f"Min: {min(sizes)}, Max: {max(sizes)}")
   ```

4. **Consider Chunk Overlap** (fixed-size):
   - Use 10-20% overlap for context preservation
   - Balance between redundancy and coverage

5. **Semantic Chunking Tuning**:
   - Lower threshold (0.6): More, smaller chunks
   - Higher threshold (0.8): Fewer, larger chunks

## Integration with Other Modules

```python
from src.dataset import DataManager
from src.chunking import ChunkerFactory
from src.embeddings import EmbedderFactory
from src.vector_stores import VectorStoreFactory

# Load and chunk documents
manager = DataManager()
documents = manager.load_from_path('data/')

chunker = ChunkerFactory.create('semantic')
chunks = chunker.chunk_documents(documents)

# Embed and store chunks
embedder = EmbedderFactory.create('huggingface')
texts = [c['content'] for c in chunks]
embeddings = embedder.embed_batch(texts)

store = VectorStoreFactory.create('chroma')
store.add_documents(texts, embeddings, [c['metadata'] for c in chunks])
```

## Troubleshooting

### Issue: Semantic chunking is slow
**Solutions**:
- Use GPU: `device: 'cuda'`
- Use smaller model: `all-MiniLM-L6-v2`
- Increase batch_size
- Consider fixed-size for large datasets

### Issue: Chunks are too small/large
**Solutions**:
- Fixed-size: Adjust `chunk_size`
- Semantic: Adjust `min_chunk_size` and `max_chunk_size`
- Check document formatting

### Issue: "Failed to initialize embedder"
**Solutions**:
- Check model name is correct
- Ensure internet connection for first download
- Verify cache_folder permissions

### Issue: Memory error during semantic chunking
**Solutions**:
- Process documents in smaller batches
- Reduce batch_size
- Use smaller embedding model

## API Reference

### ChunkerFactory

```python
ChunkerFactory.register(name: str, chunker_class: Type)
ChunkerFactory.create(strategy: str, config: Dict) -> BaseChunker
ChunkerFactory.get_available_strategies() -> List[str]
```

### BaseChunker

```python
chunk(text: str) -> List[str]
chunk_documents(documents: List[Dict]) -> List[Dict]
```

### FixedSizeChunker

```python
FixedSizeChunker(config: Dict)
# config: chunk_size, chunk_overlap
```

### SemanticChunker

```python
SemanticChunker(config: Dict)
# config: similarity_threshold, min_chunk_size, max_chunk_size, embedder
```

## Examples

See `examples/chunking_examples.py` for complete working examples.
