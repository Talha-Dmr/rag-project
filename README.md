# FinReg RAG Hallucination-Gating Project

This repository contains a research RAG pipeline for prudential / supervisory
financial regulation. The current work is focused on making a real FinReg corpus,
retrieval pipeline, hallucination detector, and abstain gate work together in a
reproducible way.

The active path is no longer the early generic ambiguity demo. Treat older
AmbigQA / FEVER / EBCAR / SGLD experiments as historical unless a task explicitly
refers to them.

## Current Active Stack

- Domain: prudential / supervisory FinReg.
- Corpus: official BCBS, EBA, ECB, PRA-BoE, and selected Fed/OCC documents.
- Chunking: `section_aware`.
- Retrieval: dense Chroma retrieval over the real FinReg section index.
- Reranking: cross-encoder reranker for the current local smoke path.
- Generation: local Qwen models, not OpenRouter.
- Detector: `finregbench_modernbert_detector_v3_hardmix`.
- Gate: abstain gate driven by detector/gating statistics.

## Repository Map

- [`src/`](src/) contains the reusable RAG, chunking, embedding, reranking, vector-store, and training code.
- [`scripts/`](scripts/) contains CLI entry points for corpus build, indexing, benchmarks, detector data, and analysis.
- [`config/`](config/) contains YAML configs. See [`config/README.md`](config/README.md) before choosing one.
- [`benchmarks/finreg/`](benchmarks/finreg/) contains the controlled detector and full-RAG benchmark inputs.
- [`data/processed/finreg/`](data/processed/finreg/) contains the processed FinReg corpus text/metadata committed to the repo.
- [`data/domain_finreg/`](data/domain_finreg/) contains FinReg question sets and detector review data.
- [`docs/`](docs/) contains project notes, inventories, methodology, and result reports. Start with [`docs/README.md`](docs/README.md).
- `models/`, `data/vector_db/`, `reports/`, `evaluation_results/`, and detector asset dumps are local/generated artifacts and are not the main source of truth.

## Most Useful Commands

Build or refresh the processed FinReg corpus:

```bash
PYTHONPATH=. venv312/bin/python scripts/build_finreg_phase1_corpus.py
```

Build the section-aware FinReg index:

```bash
PYTHONPATH=. venv312/bin/python scripts/index_domain_corpus.py \
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank_quality \
  --corpus data/processed/finreg/finreg_phase1_corpus.jsonl \
  --reset-collection \
  --index-only
```

Run retrieval audit:

```bash
PYTHONPATH=. venv312/bin/python scripts/audit_finreg_retrieval.py \
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank_quality \
  --questions benchmarks/finreg/full_rag_questions.jsonl \
  --k 24 \
  --run-name local_section_rerank_quality_k24
```

Run a small RTX2070 local smoke without detector/gating:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_CACHE=./models/llm \
TRANSFORMERS_CACHE=./models/llm \
PYTHONPATH=. venv312/bin/python scripts/run_finreg_real_life_benchmark.py \
  --mode full-rag \
  --config gating_finreg_local_qwen15_rtx2070_section_rerank_smoke \
  --k 8 \
  --limit 3 \
  --disable-detector \
  --disable-gating \
  --disable-answer-quality
```

Run a small RTX2070 detector/gating integration smoke:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_CACHE=./models/llm \
TRANSFORMERS_CACHE=./models/llm \
PYTHONPATH=. venv312/bin/python scripts/run_finreg_real_life_benchmark.py \
  --mode full-rag \
  --config gating_finreg_local_qwen15_rtx2070_section_rerank_detector_smoke \
  --k 8 \
  --limit 10 \
  --disable-answer-quality
```

## Current Local Status Notes

- The current section-aware index path is `data/vector_db/domain_finreg_real_section`.
- The current collection name is `rag_finreg_real_section`.
- The current processed corpus file is `data/processed/finreg/finreg_phase1_corpus.jsonl`.
- The RTX2070 smoke configs use `Qwen/Qwen2.5-1.5B-Instruct` with `dtype: bfloat16`; fp16 produced degenerate repeated punctuation on the local machine.
- Full answer-quality configs are heavier than the RTX2070 smoke configs and may need more GPU/RAM.

## Local Artifacts

The repo may contain local-only folders such as `models/`, `reports/`,
`evaluation_results/`, `detector-assets/`, `detector-assets-phase2.2/`, and
temporary corpus dumps. These should not be treated as clean GitHub-facing source
layout. Use manifests and docs for provenance, and only commit small metadata,
configs, scripts, benchmarks, and intentionally curated data.

## Key Docs

- [`docs/current_status.md`](docs/current_status.md)
- [`docs/finreg_phase1_document_inventory.md`](docs/finreg_phase1_document_inventory.md)
- [`docs/finreg_corpus_inventory.md`](docs/finreg_corpus_inventory.md)
- [`docs/finreg_question_set_methodology.md`](docs/finreg_question_set_methodology.md)
- [`docs/finreg_real_life_benchmark_results.md`](docs/finreg_real_life_benchmark_results.md)
- [`docs/finreg_rag_detector_quality_integration_report.md`](docs/finreg_rag_detector_quality_integration_report.md)
- [`docs/phase2_2_artifact_manifest.md`](docs/phase2_2_artifact_manifest.md)
