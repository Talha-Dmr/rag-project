# RAG Implementation Documentation

## Overview

The RAG (Retrieval-Augmented Generation) module is the core of the system, combining document retrieval with LLM generation to provide accurate, context-aware responses.

## Architecture

### Components

1. **Vector Store**: Stores document embeddings for similarity search
2. **Embedder**: Generates embeddings from text
3. **Retriever**: Fetches relevant documents based on query similarity
4. **LLM Wrapper**: Generates responses using HuggingFace models
5. **RAG Pipeline**: Orchestrates the entire process

### Data Flow

```
Query → Embedder → Vector Store Search → Retriever
                                             ↓
                                    Retrieved Documents
                                             ↓
                              LLM (with context) → Answer
```

## Quick Start

### Basic Usage

```python
from src.rag import RAGPipeline
from src.core.config_loader import load_config

# Load configuration
config = load_config('base_config')

# Create RAG pipeline
rag = RAGPipeline.from_config(config)

# Index documents
rag.index_documents('path/to/documents/', recursive=True)

# Query
result = rag.query("What is RAG?")
print(result['answer'])
```

## Vector Stores

### ChromaDB (Primary Implementation)

ChromaDB provides persistent vector storage with similarity search.

```python
from src.vector_stores import VectorStoreFactory

# Create ChromaDB store
store = VectorStoreFactory.create('chroma', {
    'persist_directory': './data/vector_db/chroma',
    'collection_name': 'documents'
})

# Add documents
store.add_documents(texts, embeddings, metadatas)

# Search
results = store.search(query_embedding, k=10)

# Get count
count = store.get_count()
```

**Features**:
- Persistent storage
- Cosine similarity search
- Metadata filtering
- Automatic indexing

**Configuration**:
```yaml
vector_store:
  type: chroma
  config:
    persist_directory: ./data/vector_db/chroma
    collection_name: documents
```

## Embeddings

### HuggingFace Embedder

Uses sentence-transformers for high-quality embeddings.

```python
from src.embeddings import EmbedderFactory

# Create embedder
embedder = EmbedderFactory.create('huggingface', {
    'model_name': 'sentence-transformers/all-mpnet-base-v2',
    'device': 'cpu',
    'cache_folder': './models/embeddings'
})

# Single embedding
embedding = embedder.embed_text("Sample text")

# Batch embeddings
embeddings = embedder.embed_batch(["text1", "text2", "text3"])

# Get dimension
dim = embedder.get_dimension()  # 768 for all-mpnet-base-v2
```

**Recommended Models**:
- `all-mpnet-base-v2`: Best quality, 768 dim
- `all-MiniLM-L6-v2`: Fast, 384 dim
- `multi-qa-mpnet-base-dot-v1`: Optimized for Q&A

**Configuration**:
```yaml
embeddings:
  model_name: sentence-transformers/all-mpnet-base-v2
  device: cpu  # or cuda
  cache_folder: ./models/embeddings
  batch_size: 32
```

## LLM Wrapper

### HuggingFace LLM

Wraps HuggingFace causal language models.

```python
from src.rag import HuggingFaceLLM

# Create LLM
llm = HuggingFaceLLM({
    'model_name': 'meta-llama/Llama-2-7b-chat-hf',
    'device': 'cpu',
    'max_new_tokens': 512,
    'temperature': 0.7
})

# Generate from prompt
response = llm.generate("What is machine learning?")

# Generate with context
context = ["Context document 1", "Context document 2"]
response = llm.generate_with_context(
    query="What is RAG?",
    context=context
)
```

**Supported Models**:
- Llama-2-7b-chat-hf
- Mistral-7B-Instruct-v0.1
- Phi-2
- Flan-T5 series

**Configuration**:
```yaml
llm:
  model_name: meta-llama/Llama-2-7b-chat-hf
  device: cpu  # or cuda
  cache_folder: ./models/llm
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  load_in_8bit: false  # Set true for GPU memory optimization
```

## Retriever

Retrieves relevant documents based on query similarity.

```python
from src.rag import Retriever
from src.embeddings import EmbedderFactory
from src.vector_stores import VectorStoreFactory

# Create components
embedder = EmbedderFactory.create('huggingface', config)
vector_store = VectorStoreFactory.create('chroma', config)

# Create retriever
retriever = Retriever(
    embedder=embedder,
    vector_store=vector_store,
    k=10,
    score_threshold=0.7
)

# Retrieve documents
docs = retriever.retrieve("What is RAG?", k=5)

for doc in docs:
    print(f"Score: {doc['score']:.3f}")
    print(f"Content: {doc['content'][:100]}...")
```

**Parameters**:
- `k`: Number of documents to retrieve
- `score_threshold`: Minimum similarity score (0-1)
- `filter_dict`: Metadata filters (optional)

**Configuration**:
```yaml
retrieval:
  k: 10
  score_threshold: 0.7
```

## RAG Pipeline

### Complete Workflow

```python
from src.rag import RAGPipeline
from src.core.config_loader import load_config

# Load config
config = load_config('base_config')

# Create pipeline
rag = RAGPipeline.from_config(config)

# 1. Index documents
num_chunks = rag.index_documents(
    source='data/documents/',
    recursive=True,
    file_extensions=['.pdf', '.txt']
)
print(f"Indexed {num_chunks} chunks")

# 2. Query the system
result = rag.query(
    query_text="What is retrieval-augmented generation?",
    k=5,
    return_context=True
)

print(f"Answer: {result['answer']}")
print(f"Retrieved {result['num_docs_retrieved']} documents")

if 'context' in result:
    for i, doc in enumerate(result['context']):
        print(f"\nDocument {i+1} (score: {doc['score']:.3f}):")
        print(doc['content'][:200])
```

### Manual Pipeline Construction

```python
from src.dataset import DataManager
from src.chunking import ChunkerFactory
from src.embeddings import EmbedderFactory
from src.vector_stores import VectorStoreFactory
from src.rag import HuggingFaceLLM, Retriever, RAGPipeline

# Create components
data_manager = DataManager()
chunker = ChunkerFactory.create('semantic')
embedder = EmbedderFactory.create('huggingface')
vector_store = VectorStoreFactory.create('chroma')
llm = HuggingFaceLLM()
retriever = Retriever(embedder, vector_store)

# Assemble pipeline
rag = RAGPipeline(
    data_manager=data_manager,
    chunker=chunker,
    embedder=embedder,
    vector_store=vector_store,
    llm=llm,
    retriever=retriever
)
```

## Configuration

### Complete config/base_config.yaml

```yaml
# Data Loading
data_loader:
  type: pdf
  batch_size: 10

# Chunking
chunking:
  strategy: semantic
  config:
    similarity_threshold: 0.7
    min_chunk_size: 100
    max_chunk_size: 1000
    embedder:
      type: huggingface
      model_name: sentence-transformers/all-mpnet-base-v2
      device: cpu

# Embeddings
embeddings:
  model_name: sentence-transformers/all-mpnet-base-v2
  device: cpu
  cache_folder: ./models/embeddings
  batch_size: 32

# Vector Store
vector_store:
  type: chroma
  config:
    persist_directory: ./data/vector_db/chroma
    collection_name: rag_documents

# LLM
llm:
  model_name: meta-llama/Llama-2-7b-chat-hf
  device: cpu
  cache_folder: ./models/llm
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  load_in_8bit: false

# Retrieval
retrieval:
  k: 10
  score_threshold: 0.7
```

## Best Practices

### 1. Indexing Documents

```python
# Index in batches for large datasets
rag.index_documents(
    source='documents/',
    recursive=True,
    file_extensions=['.pdf', '.txt', '.md']
)
```

### 2. Query Optimization

```python
# Adjust k based on use case
result = rag.query(
    query_text="...",
    k=3,  # Fewer docs for focused answers
    return_context=False  # Reduce response size
)
```

### 3. Performance Tuning

**For Speed**:
- Use smaller models (MiniLM, Phi-2)
- Use GPU (`device: 'cuda'`)
- Reduce batch_size for lower memory
- Use fixed-size chunking

**For Quality**:
- Use larger models (mpnet, Llama-2)
- Use semantic chunking
- Increase retrieval k
- Adjust score_threshold

## Troubleshooting

### Issue: Models downloading slowly
**Solution**: Pre-download models:
```python
from transformers import AutoModel
from sentence_transformers import SentenceTransformer

# Download embedding model
SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Download LLM
AutoModel.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
```

### Issue: Out of memory
**Solutions**:
- Use smaller models
- Enable 8-bit quantization: `load_in_8bit: true`
- Reduce batch_size
- Use CPU instead of GPU for embeddings

### Issue: Poor retrieval quality
**Solutions**:
- Use semantic chunking
- Adjust score_threshold
- Try different embedding models
- Increase chunk overlap (fixed-size)

### Issue: Slow queries
**Solutions**:
- Use GPU for LLM
- Reduce max_new_tokens
- Cache frequently accessed documents
- Use smaller k for retrieval

## API Reference

### RAGPipeline

```python
RAGPipeline(data_manager, chunker, embedder, vector_store, llm, retriever)
RAGPipeline.from_config(config: Dict) -> RAGPipeline
index_documents(source, recursive, file_extensions) -> int
query(query_text, k, return_context) -> Dict
```

### Retriever

```python
Retriever(embedder, vector_store, k, score_threshold)
Retriever.from_config(config: Dict) -> Retriever
retrieve(query, k, filter_dict) -> List[Dict]
```

### HuggingFaceLLM

```python
HuggingFaceLLM(config: Dict)
generate(prompt, max_tokens, temperature) -> str
generate_with_context(query, context, max_tokens) -> str
```

### Vector Stores

```python
VectorStoreFactory.create(store_type, config) -> BaseVectorStore
add_documents(texts, embeddings, metadatas)
search(query_embedding, k, filter_dict) -> List[Dict]
get_count() -> int
delete_collection()
```

## Examples

See `examples/rag_examples.py` for complete working examples.
