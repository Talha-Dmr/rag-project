# Repository Guidelines

## Project Structure & Module Organization
Source code lives in `src/`, where `core/` holds abstractions, `dataset/` loads PDFs/text/JSON, `chunking/` manages splitters, `embeddings/` wraps HuggingFace encoders, `vector_stores/` contains FAISS/Chroma adapters, `rag/` wires the pipeline, and `reranking/` hosts cross-encoder plus BM25 passes. YAML configs sit in `config/`, `.env.example` records required secrets, and `data/` stores ignored raw corpora. Long-form notes and walkthroughs are under `docs/`, while runnable samples are in `examples/` and automation helpers live in `scripts/`. All regression tests live in `tests/` together with domain-specific checks like `test_datasets.py` in the root.

## Build, Test & Development Commands
- `poetry install` - install locked dependencies from `pyproject.toml`.
- `poetry shell` / `poetry run <cmd>` - ensure commands use the managed interpreter.
- `poetry run python demo_ambiguity_rag.py --config config/base_config.yaml` - execute the reference RAG workflow end to end.
- `poetry run pytest` - run the suite; pair with `-k` or `-m` for focused iterations.
- `poetry run pytest --cov=src --cov-report=term-missing` - inspect coverage before merging.
- `poetry run black src/ tests/` and `poetry run flake8 src/ tests/` - enforce formatting and linting.
- `poetry run mypy src/` - gate new contracts with static checks.

## Coding Style & Naming Conventions
Use 4-space indentation, snake_case for modules/functions, and PascalCase for classes. Black is configured for 100-column lines (Python 3.10 target), so let it handle wrapping via `poetry run black`. Keep module boundaries narrow; each subpackage should export clear factories for downstream composition. Keep env vars synchronized with `.env.example`, prefer descriptive config names such as `config/vector/chroma.yaml`, and document public interfaces with concise docstrings, especially when exposing new LangChain or HuggingFace wrappers.

## Testing Guidelines
Pytest is configured to discover `tests/` with filenames `test_*.py`, classes `Test*`, and functions `test_*`. Mirror the source layout (`tests/vector_stores/test_chroma.py`, etc.) so fixtures stay scoped. Favor lightweight synthetic fixtures or assets in `tests/fixtures/` over live corpora in `data/`. Before opening a PR, run `poetry run pytest --cov=src` and include the command output in the PR description. Mark slow GPU-dependent cases with `@pytest.mark.slow` so the default suite stays responsive.

## Commit & Pull Request Guidelines
The history follows Conventional Commits (`feat: optimize training performance`), so keep `type: imperative summary` messages and reference issues when relevant (`fix: resolve faiss reload (#42)`). Squash fixups locally to maintain a clean log. Every PR should explain the motivation, list testing commands executed, flag config changes that require reindexing, and add screenshots or tensorboard links if UX or training curves change. Request reviewers who own the touched module (`vector_stores`, `rag`, etc.) and wait for CI green lights before merging.

## Configuration & Security Notes
Do not commit populated `.env` files or raw datasets; stick to the ignored `data/` directory. Store secrets (API keys, HuggingFace tokens) locally and reference them via `${VARIABLE}` in YAML so configs stay portable. Create new configs in `config/custom/` instead of editing shared defaults, and mention those paths in your PR description so other agents can reproduce runs. When exporting checkpoints or tensorboard logs, scrub PII and verify licensing terms before sharing outside the repo.

