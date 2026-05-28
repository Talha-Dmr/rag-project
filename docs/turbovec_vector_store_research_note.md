# TurboVec Vector Store Research Note

Date: 2026-05-28

## Current Project State

The current FinReg RAG pipeline uses ChromaDB, not FAISS.

Evidence from the repo:

- FinReg configs set `vector_store.type: chroma`.
- Active FinReg index path is `./data/vector_db/domain_finreg`.
- Active collection name is `rag_finreg`.
- Runtime logs show `ChromaDB initialized: collection='rag_finreg', documents=2221`.
- The only implemented vector store backend under `src/vector_stores/stores/` is `chroma_store.py`.

The base config mentions `faiss` as a possible option, but there is no FAISS backend implementation currently wired into the project.

## What TurboVec Is

TurboVec is a Rust vector index with Python bindings. It is built around TurboQuant-style vector quantization and focuses on local vector search with aggressive memory compression.

Claimed properties from the public project pages:

- 2-4 bit vector compression.
- No codebook training step.
- Online insertion without full rebuilds.
- SIMD-accelerated search.
- Python package available as `turbovec`.
- Supports stable external ids through `IdMapIndex`.
- Supports filtered search via allowlists.
- Local-only operation, suitable for private or air-gapped RAG stacks.

Sources:

- https://github.com/RyanCodrai/turbovec
- https://pypi.org/project/turbovec/

## Relevance To This Project

TurboVec is interesting, but it is unlikely to be the first bottleneck in our current setup.

Current FinReg corpus size is about 2,221 indexed documents. At that scale, Chroma search latency and memory usage are probably not the dominant cost. The heavier costs in our recent runs are:

- local LLM generation,
- hallucination detector execution,
- reranking,
- evidence-sampling retry passes,
- larger model VRAM pressure.

Therefore, TurboVec should not be treated as an immediate replacement for Chroma in the main pipeline. It should be treated as a candidate retrieval backend to benchmark.

## Potential Upside

TurboVec may become useful if the corpus grows substantially or if local memory footprint becomes a constraint.

Possible advantages:

- Lower RAM use for large embedding indexes.
- Fast local search without a managed vector database service.
- Good fit for private/local RAG deployment.
- Filtered dense search could help later if we add tenant, source, date, regulation-family, or access-control filters.

## Main Risks

- The package is new and marked alpha on PyPI.
- The current project has a custom vector store interface, so integration requires a new backend implementation.
- Retrieval quality may change because TurboVec uses quantized approximate search.
- Our present corpus is small enough that the performance gain may be negligible.
- Chroma currently stores documents, metadata, and vectors together; TurboVec may require us to persist metadata/documents separately or wrap them carefully.

## Recommended Path

Do not switch the production/default pipeline yet.

Recommended next step is an isolated benchmark:

1. Add an optional `TurboVecStore` backend behind the existing `VectorStoreFactory`.
2. Keep Chroma as the default.
3. Build the same FinReg index in TurboVec.
4. Run retrieval-only comparison against Chroma.
5. Run full RAG comparison only if retrieval quality is acceptable.

## Benchmark Plan

Retrieval-only metrics:

- recall@k against known relevant chunks where labels exist,
- hit@k,
- top-k overlap with Chroma,
- average and p95 retrieval latency,
- index build time,
- index disk size,
- RAM usage during search.

Full RAG metrics:

- hallucination flag rate,
- abstain rate,
- evidence-sampling `retrieve_more` rate,
- answer quality on conflict questions,
- final answer stability when top-k changes.

Suggested test sets:

- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl`
- `benchmarks/finreg/full_rag_questions.jsonl`

## Decision Rule

TurboVec is worth adopting only if it preserves retrieval quality while improving at least one operational constraint.

Minimum acceptance criteria:

- no meaningful drop in recall@k or answer quality,
- no increase in hallucination/abstain rate,
- measurable latency or memory advantage,
- clean persistence and rebuild workflow,
- no instability from the Python/Rust package in our environment.

Until those are proven, Chroma remains the safer default.
