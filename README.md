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

- Current defaults (epistemic track, latest 50Q rep-vs-logit ablation):
  - Energy (logit-MI): `config/gating_energy_ebcar_logit_mi_sc009.yaml` → abstain 35/50 (0.70)
  - Macro (logit-MI): `config/gating_macro_ebcar_logit_mi_sc009.yaml` → abstain 22/50 (0.44)
- Representation-space track (experimental):
  - Energy (rep-MI): `config/gating_energy_ebcar_rep_mi_sc004.yaml` → abstain 45/50 (0.90)
  - Macro (rep-MI): `config/gating_macro_ebcar_rep_mi_sc004.yaml` → abstain 44/50 (0.88)
- Decision:
  - Keep `logit-MI` as default epistemic signal.
  - Keep `rep-MI` as experimental (currently too conservative on coverage).
- Coverage-first alternative:
  - Energy: `config/gating_energy_ebcar_consistency_only_sc050.yaml` → abstain 14/50 (0.28)
  - Macro: `config/gating_macro_ebcar_consistency_only_sc050.yaml` → abstain 9/50 (0.18)
- Safety variant (MC Dropout + consistency, 50Q):
  - Energy: abstain 31/50 (0.62)
  - Macro: abstain 36/50 (0.72)
- Historical threshold sweep (older calibration run):
  - threshold=0.70 → abstain 46/50 (0.92)
  - threshold=0.85 → abstain 22/50 (0.44)
  - threshold=0.90 → abstain 21/50 (0.42)
- Gating ablation (Energy 20Q, Macro 50Q; conflict-heavy pilot):

| Domain | Setting | abstain | abstain_rate | actions |
| --- | --- | --- | --- | --- |
| Energy | nogate | 0/20 | 0.00 | none=20 |
| Energy | retrieve_more | 11/20 | 0.55 | retrieve_more=11, none=9 |
| Energy | abstain | 15/20 | 0.75 | abstain=15, none=5 |
| Macro | nogate | 0/50 | 0.00 | none=50 |
| Macro | retrieve_more | 11/50 | 0.22 | retrieve_more=11, none=39 |
| Macro | abstain | 35/50 | 0.70 | abstain=35, none=15 |
- Note: Energy set is 20Q (conflict-heavy) so this is a pilot ablation; Macro 50Q is the more stable signal.
- Signal comparison (detector on):
  - Energy (50Q): MC Dropout + consistency `31/50` vs consistency-only `14/50`
  - Macro (50Q): MC Dropout + consistency `36/50` vs consistency-only `9/50`
- Source-consistency only (detector on, no MC Dropout):
  - Energy (50Q): abstain `14/50` (0.28)
  - Macro (50Q): abstain `9/50` (0.18)
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
