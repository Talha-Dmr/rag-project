# FinReg RAG Hallucination-Gating Project

This repository contains a research RAG pipeline for prudential and supervisory
financial regulation. The current project is focused on making retrieval,
answer generation, hallucination detection, deterministic abstention, and
stochastic evidence-subset gating work together on a real FinReg corpus.

Older AmbigQA, FEVER, EBCAR, SGLD, and generic modular-RAG experiments are kept
for provenance, but they are not the active report path unless a task
explicitly refers to them.

## Current Research Question

Can a FinReg RAG system reduce unsupported regulatory claims by combining:

- section-aware retrieval from official regulatory documents,
- a ModernBERT hallucination detector,
- abstention when evidence is weak or risky, and
- stochastic evidence sampling over retrieved evidence subsets?

The active contribution is not simply "use a stronger LLM." The LLM can be
local Qwen or an API model, but the project contribution is the detector/gating
layer and the benchmark used to measure unsupported overclaiming.

## Current Stack

- Domain: prudential / supervisory FinReg.
- Corpus: official BCBS, EBA, ECB, PRA-BoE, and selected Fed/OCC documents.
- Chunking: `section_aware`.
- Vector store: Chroma over the real FinReg section index.
- Retrieval: dense retrieval plus optional query expansion.
- Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Generation: local Qwen configs for report runs; DeepSeek API configs are also
  available for lower local VRAM pressure.
- Detector: `finregbench_modernbert_detector_v3_hardmix`.
- Gate: detector-driven abstention plus optional stochastic evidence sampling.

## Final Targeted Benchmark

The current report benchmark is:

`benchmarks/finreg/full_rag_questions_final_targeted160.jsonl`

It contains 160 targeted hallucination stress-test questions:

| Category | Count | What it tests |
|---|---:|---|
| `cross_source_nuanced` | 72 | Unsupported transfer across regulators or documents. |
| `low_evidence_policy` | 40 | Inventing exact operational details from partial evidence. |
| `false_premise` | 32 | Accepting fabricated regulatory requirements. |
| `factual_supported` | 16 | Answering when evidence is directly supported. |

Audited benchmark result:

| System | Expected Behavior | Point Coverage | Answer Rate | Abstain Rate | Forbidden Claim Hit | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| Baseline RAG | 80.00% | 24.34% | 100.00% | 0.00% | 0.00% | 5.32s |
| RAG + Detector | 92.50% | 13.44% | 46.25% | 53.75% | 0.00% | 5.48s |
| RAG + Detector + Stochastic | 93.12% | 13.28% | 45.62% | 54.37% | 0.00% | 5.88s |

The detector provides the main improvement. Stochastic evidence sampling adds a
smaller positive gain while preserving zero benchmark-defined forbidden claim
hits. Point coverage is secondary in the audited benchmark because expected
points are now longer atomic propositions rather than short keyword labels.

## Repository Map

- [`src/`](src/) contains reusable RAG, retrieval, detector, gating, and LLM code.
- [`scripts/`](scripts/) contains CLI entry points for corpus build, indexing,
  benchmark generation, benchmark runs, detector data, and analysis.
- [`config/`](config/) contains YAML configs. Start with
  [`config/README.md`](config/README.md) before choosing one.
- [`benchmarks/finreg/`](benchmarks/finreg/) contains controlled detector and
  full-RAG benchmark inputs. Start with
  [`benchmarks/finreg/README.md`](benchmarks/finreg/README.md).
- [`data/processed/finreg/`](data/processed/finreg/) contains the processed
  FinReg corpus text/metadata committed to the repo.
- [`data/domain_finreg/`](data/domain_finreg/) contains FinReg question sets and
  detector review data.
- [`docs/`](docs/) contains methodology, status notes, audit reports, and poster
  material. Start with [`docs/README.md`](docs/README.md).

Local/generated folders such as `models/`, `data/vector_db/`, `reports/`,
`evaluation_results/`, and detector asset dumps are not the main source of
truth for GitHub-facing documentation.

## Main Configs

- `final_finreg_qwen3b_rag.yaml`: baseline final benchmark RAG.
- `final_finreg_qwen3b_detector.yaml`: final benchmark RAG + detector gate.
- `final_finreg_qwen3b_detector_stochastic.yaml`: final benchmark RAG +
  detector + stochastic evidence sampling.
- `gating_finreg_deepseek_v4_flash_no_evidence_sampling_coverage_quality.yaml`:
  API generation path without stochastic evidence sampling.
- `gating_finreg_deepseek_v4_flash_evidence_vector_v3_coverage_quality.yaml`:
  API generation path with vector stochastic evidence sampling.
- `gating_finreg_local_qwen15_rtx2070_section_rerank_smoke.yaml`: small local
  retrieval/generation smoke config.
- `gating_finreg_local_qwen15_rtx2070_section_rerank_detector_smoke.yaml`: small
  local detector/gating smoke config.

## Useful Commands

Build or refresh the processed FinReg corpus:

```bash
PYTHONPATH=. venv312/bin/python scripts/build_finreg_phase1_corpus.py
```

Build the section-aware FinReg index:

```bash
PYTHONPATH=. venv312/bin/python scripts/index_domain_corpus.py \
  --config final_finreg_qwen3b_detector_stochastic \
  --corpus data/processed/finreg/finreg_phase1_corpus.jsonl \
  --reset-collection \
  --index-only
```

Run the final local stochastic benchmark path:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_CACHE=./models/llm \
TRANSFORMERS_CACHE=./models/llm \
PYTHONPATH=. venv312/bin/python scripts/run_finreg_real_life_benchmark.py \
  --mode full-rag \
  --config final_finreg_qwen3b_detector_stochastic \
  --questions benchmarks/finreg/full_rag_questions_final_targeted160.jsonl \
  --k 16 \
  --run-name final_qwen3b_detector_stochastic
```

Run the DeepSeek API stochastic path:

```bash
DEEPSEEK_API_KEY=... \
PYTHONPATH=. venv312/bin/python scripts/run_finreg_real_life_benchmark.py \
  --mode full-rag \
  --config gating_finreg_deepseek_v4_flash_evidence_vector_v3_coverage_quality \
  --questions benchmarks/finreg/full_rag_questions_final_targeted160.jsonl \
  --k 16 \
  --run-name deepseek_v4_flash_stochastic
```

Run a small local smoke test:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=. venv312/bin/python scripts/run_finreg_real_life_benchmark.py \
  --mode full-rag \
  --config gating_finreg_local_qwen15_rtx2070_section_rerank_detector_smoke \
  --questions benchmarks/finreg/full_rag_questions_final_targeted160.jsonl \
  --k 8 \
  --limit 10 \
  --disable-answer-quality
```

## Key Docs

- [`docs/finreg_final_targeted_benchmark_audit.md`](docs/finreg_final_targeted_benchmark_audit.md)
- [`docs/finreg_final_targeted_benchmark_report.md`](docs/finreg_final_targeted_benchmark_report.md)
- [`docs/poster_right_panel_benchmark_metrics.md`](docs/poster_right_panel_benchmark_metrics.md)
- [`docs/finreg_stochastic_adapter_backlog.md`](docs/finreg_stochastic_adapter_backlog.md)
- [`docs/finreg_stochastic_gating_diagnostic_note.md`](docs/finreg_stochastic_gating_diagnostic_note.md)
- [`docs/finreg_phase1_document_inventory.md`](docs/finreg_phase1_document_inventory.md)
- [`docs/finreg_question_set_methodology.md`](docs/finreg_question_set_methodology.md)
- [`docs/phase2_2_artifact_manifest.md`](docs/phase2_2_artifact_manifest.md)

If an older document conflicts with this README or the final targeted benchmark
audit, prefer the newer audit/report files.
