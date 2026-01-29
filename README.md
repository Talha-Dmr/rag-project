# Modular RAG Project

A highly modular Retrieval-Augmented Generation (RAG) system built with LangChain and HuggingFace local models.

## Features

- **Modular Dataset Loading**: Support for PDF, Text, JSON, and more
- **Flexible Chunking Strategies**: Semantic, fixed-size, and recursive chunking
- **Vector Store Abstraction**: Easily swap between ChromaDB, FAISS, and others
- **HuggingFace Integration**: Local LLM and embedding models
- **Advanced Reranking**: Cross-encoder and BM25 reranking strategies
- **Configuration-Driven**: Change components via YAML configs without code changes

## Results (Current Snapshot)

- Final MC Dropout gating (50 questions):
  - Energy: abstain 13/50 (0.26), actions none=37, retrieve_more=13
  - Macro: abstain 9/50 (0.18), actions none=41, retrieve_more=9
- Details and ablations: see `docs/future_work_langevin.md`

## Project Structure

```
rag-project/
├── src/
│   ├── core/           # Base classes and utilities
│   ├── dataset/        # Data loading module
│   ├── chunking/       # Text chunking strategies
│   ├── embeddings/     # Embedding models
│   ├── vector_stores/  # Vector database implementations
│   ├── rag/            # RAG pipeline
│   └── reranking/      # Reranking strategies
├── config/             # Configuration files
├── docs/               # Documentation
├── examples/           # Example scripts
└── tests/              # Test suite
```

## Installation

### Using Poetry (Recommended)

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Optional GPU Support

```bash
poetry install --extras gpu
```

## Quick Start

```python
from src.core.config_loader import ConfigLoader
from src.rag.rag_pipeline import RAGPipeline

# Load configuration
config = ConfigLoader().load('base_config')

# Initialize RAG pipeline
rag = RAGPipeline.from_config(config)

# Index documents
rag.index_documents(source="/path/to/documents")

# Query
response = rag.query("What is RAG?", return_sources=True, include_sources_in_answer=True)
print(response)
```

## Configuration

Edit `config/base_config.yaml` to customize:

- Data loaders (PDF, Text, JSON)
- Chunking strategies (Semantic, Fixed-size, Recursive)
- Vector stores (ChromaDB, FAISS)
- Embedding models
- LLM models
- Reranking methods

## Documentation

- [Dataset Implementation](docs/01_dataset_implementation.md)
- [Chunking Implementation](docs/02_chunking_implementation.md)
- [RAG Implementation](docs/03_rag_implementation.md)
- [Reranking Implementation](docs/04_reranking_implementation.md)

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black src/

# Type checking
poetry run mypy src/
```

## Requirements

- Python 3.10+
- 8GB+ RAM (16GB recommended for larger models)
- Optional: CUDA-compatible GPU for faster inference

## License

MIT License
