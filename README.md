# RAG Project

This repository is currently centered on a **prudential / supervisory FinReg RAG baseline**.

The active stack is not a generic multi-domain demo anymore. The canonical working path is:

- real FinReg corpus from official BCBS / EBA / ECB / PRA-BoE / selected Fed sources
- `section_aware` chunking
- dense retrieval + `ebcar` reranking
- local `Qwen/Qwen2.5-1.5B-Instruct`
- FEVER-based hallucination detector
- stochastic gating with `retrieve_more`

## Canonical Baseline

- config: [config/gating_finreg_ebcar_logit_mi_sc009.yaml](/home/talha/projects/rag-project/config/gating_finreg_ebcar_logit_mi_sc009.yaml)
- baseline summary: [docs/current_finreg_baseline.md](/home/talha/projects/rag-project/docs/current_finreg_baseline.md)
- source manifest: [config/finreg_phase1_sources.yaml](/home/talha/projects/rag-project/config/finreg_phase1_sources.yaml)

Current corpus status:

- `31` official source documents
- about `2221` indexed chunks in `rag_finreg`

Primary question sets:

- [data/domain_finreg/questions_finreg_conflict_phase1_refined_v2.jsonl](/home/talha/projects/rag-project/data/domain_finreg/questions_finreg_conflict_phase1_refined_v2.jsonl)
- [data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl](/home/talha/projects/rag-project/data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl)

## Quick Start

Fetch the official source documents:

```bash
PYTHONPATH=. venv312/bin/python scripts/fetch_finreg_phase1_sources.py
```

Build the processed corpus:

```bash
PYTHONPATH=. venv312/bin/python scripts/build_finreg_phase1_corpus.py
```

Rebuild the FinReg index:

```bash
PYTHONPATH=. venv312/bin/python scripts/index_domain_corpus.py \
  --config gating_finreg_ebcar_logit_mi_sc009 \
  --corpus data/processed/finreg/finreg_phase1_corpus.jsonl \
  --reset-collection
```

Run the current 20-question baseline:

```bash
PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy.py \
  --config gating_finreg_ebcar_logit_mi_sc009 \
  --questions data/domain_finreg/questions_finreg_conflict_phase1_refined_v2.jsonl \
  --limit 20 \
  --seed 7 \
  --output evaluation_results/auto_eval/finreg_readme_seed7.json
```

Run the 3-seed confirmation loop:

```bash
PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy_multi_seed.py \
  --config gating_finreg_ebcar_logit_mi_sc009 \
  --questions data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl \
  --limit 50 \
  --seeds 7 11 19 \
  --output-dir evaluation_results/auto_eval
```

## Working Rules

- Treat the FinReg baseline as the default research path unless a task explicitly says otherwise.
- Treat legacy question sets, bootstrap corpus notes, and early stability reports as historical only.
- Prefer the local `Qwen` path. OpenRouter is not part of the active baseline.

## Main References

- baseline summary: [docs/current_finreg_baseline.md](/home/talha/projects/rag-project/docs/current_finreg_baseline.md)
- locked pointer: [docs/baseline_locked.md](/home/talha/projects/rag-project/docs/baseline_locked.md)
- corpus runbook: [docs/finreg_phase1_corpus_runbook.md](/home/talha/projects/rag-project/docs/finreg_phase1_corpus_runbook.md)
- question-set methodology: [docs/finreg_question_set_methodology.md](/home/talha/projects/rag-project/docs/finreg_question_set_methodology.md)
