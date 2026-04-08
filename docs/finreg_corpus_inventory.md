# FinReg Corpus Inventory

## Current State

- Active finreg question sets:
  - `data/domain_finreg/questions_finreg_conflict.jsonl`
  - `data/domain_finreg/questions_finreg_conflict_50.jsonl`
- Active finreg vector store:
  - `data/vector_db/domain_finreg`
- Corpus file currently referenced by docs/scripts:
  - `data/corpora/finreg_corpus.jsonl`

## What The Current Corpus Actually Is

The current `finreg_corpus.jsonl` is not a real regulatory corpus.

- It contains 23 documents.
- All records have `source_type = synthetic_benchmark_note`.
- The records are short synthetic notes with fields like:
  - `source_org`
  - `topic`
  - `stance`
  - `details`
- The current Chroma collection `rag_finreg` also contains 23 embeddings, which matches this synthetic corpus size.

This means the current finreg setup is a bootstrap proxy for uncertainty-aware RAG testing, not a document-grounded financial regulation corpus.

## Provenance

`data/corpora/finreg_corpus.jsonl` is ignored by git via `.gitignore` (`data/*`), so it does not have commit history.

The likely source is:

- `scripts/build_high_stakes_bootstrap_corpora.py`

That script explicitly builds synthetic corpora for health, finreg, and disaster, and writes:

- `data/corpora/finreg_corpus.jsonl`

## Current Question Set Status

### 20Q set

- Path: `data/domain_finreg/questions_finreg_conflict.jsonl`
- Size: 20
- Mix:
  - 5 sanity
  - 15 conflict

### 50Q set

- Path: `data/domain_finreg/questions_finreg_conflict_50.jsonl`
- Size: 50
- Mix:
  - 10 sanity
  - 40 conflict

The 20Q finreg set was added in commit:

- `3fa845edc8fce2fa1c11cd60e508ff531dda417b`
- `Promote balanced detector defaults and add high-stakes eval scaffold`

The 50Q set is an expansion of the 20Q seed set using:

- `scripts/build_high_stakes_questions_50.py`

This means the current finreg question sets are project-authored high-stakes/conflict questions, not an external benchmark.

## Implication For Evaluation

Today the finreg pipeline is testing this combination:

- conflict-oriented finreg questions
- against a tiny synthetic corpus

So current finreg eval results should be interpreted as:

- `synthetic finreg proxy results`

They should not be interpreted as:

- `real financial regulation RAG performance`

## Canonical Corpus Goal

The first real finreg corpus should be narrower than "all finance regulation", but real enough to support retrieval, grounding, and source-conflict evaluation.

Recommended phase-1 scope:

- BCBS / BIS prudential standards and principles
- EBA supervisory guidelines and reports
- PRA / BoE supervisory statements and reporting guidance
- ECB supervisory expectation documents

Optional phase-1 supporting bucket:

- technical reporting manuals or template instructions that directly support answerable operational questions

Out of scope for phase 1:

- broad US multi-agency coverage
- ESG / IFRS side branches unless they directly support current question families
- academic or commentary sources

## Canonical Corpus Requirements

Each included source should be reproducible from raw documents, not manually copied notes.

Minimum artifact chain:

1. raw documents
2. processed text
3. chunk files with stable metadata
4. vector index built from processed artifacts

Each chunk should carry metadata for:

- `source_org`
- `document_id`
- `title`
- `jurisdiction`
- `document_type`
- `year`
- `page_start`
- `page_end`
- `section` if recoverable

## Immediate Next Steps

1. Freeze the current bootstrap setup as synthetic-only.
2. Define the phase-1 document list for BCBS, EBA, PRA/BoE, and ECB.
3. Build a real raw-to-processed finreg corpus pipeline.
4. Rebuild the finreg vector store from processed artifacts.
5. Re-check which existing finreg questions are answerable from the real corpus.
