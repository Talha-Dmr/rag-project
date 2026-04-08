# FinReg Corpus Build Plan

## Objective

Replace the current synthetic finreg bootstrap corpus with a reproducible real-document corpus that supports:

- indexing
- retrieval evaluation
- grounded answering
- conflict-aware gating analysis

## Current Problem

Today the finreg stack depends on:

- `data/corpora/finreg_corpus.jsonl`
- `data/vector_db/domain_finreg`

But the current corpus is:

- synthetic
- small
- git-ignored
- not reproducible from raw sources

## Canonical Artifact Chain

The target artifact chain should be:

1. raw documents
2. processed text artifacts
3. chunk artifacts
4. vector DB built from processed chunks
5. question coverage report

The vector store must never be the only source of truth.

## Proposed Directory Layout

### Raw

```text
data/raw/finreg/
  bcbs/
  eba/
  pra_boe/
  ecb/
  reporting/
```

Each file should remain as close as possible to the original source document.

### Processed

```text
data/processed/finreg/
  bcbs/
  eba/
  pra_boe/
  ecb/
  reporting/
```

Per source document, processed output should include:

- cleaned text
- metadata JSON
- page JSON
- chunk JSONL

### Index

```text
data/vector_db/domain_finreg/
```

This is rebuildable output, not canonical evidence.

## Processing Stages

### Stage 1. Source acquisition

Input:

- PDFs or official text documents

Requirements:

- source URL or provenance recorded
- stable file naming
- document family tagging

### Stage 2. Text extraction

For each source document:

- extract text
- preserve page boundaries when possible
- detect OCR-poor files

Output:

- `.txt`
- `.pages.json`
- `.metadata.json`

### Stage 3. Chunking

Chunk output should be built from processed text, not directly from raw documents.

Requirements:

- stable `chunk_id`
- stable metadata
- page-range traceability
- no silent metadata loss

Output:

- `.chunks.jsonl`

### Stage 4. Index build

Index build should consume only processed chunk artifacts.

Current script that can remain in the flow:

- `scripts/index_domain_corpus.py`

But its input should move from synthetic bootstrap JSONL toward the processed real corpus path.

### Stage 5. Coverage check

Before detector work, run a coverage audit:

- which current questions are answerable from corpus evidence
- which conflict questions map to at least two real source families
- which questions should be rewritten or dropped

## Metadata Standard

Each chunk should expose at least:

- `chunk_id`
- `source_org`
- `document_family`
- `document_id`
- `title`
- `jurisdiction`
- `document_type`
- `year`
- `source_path`
- `page_start`
- `page_end`

Optional but useful:

- `section`
- `topic`
- `effective_date`

## Git / Versioning Policy

### Commit

Should be tracked:

- domain question sets
- corpus build scripts
- processed metadata if small enough
- corpus inventory docs

### Not primary source of truth

Can remain rebuildable artifacts:

- vector DB
- local caches

### Explicitly avoid

- depending on ignored local JSONL as the only corpus source

## Migration Plan

### Phase A. Freeze the synthetic setup

Keep the current bootstrap corpus only as:

- synthetic retrieval/gating proxy

Do not treat it as canonical finreg evidence.

### Phase B. Build real raw corpus

Collect and store phase-1 documents under:

- `data/raw/finreg/...`

### Phase C. Build processed corpus

Generate:

- text
- metadata
- pages
- chunks

under:

- `data/processed/finreg/...`

### Phase D. Rebuild vector DB

Re-index finreg from processed artifacts into:

- `data/vector_db/domain_finreg`

### Phase E. Re-validate question set

Only after the rebuild:

- check answerability
- check conflict grounding
- prune weak or synthetic-only questions

## Immediate Work Items

1. document the phase-1 source list
2. define raw file naming rules
3. decide whether to extend the existing PDF ingestion path or build a finreg-specific one
4. create a small coverage audit for the current 20Q set
5. stop treating `finreg_corpus.jsonl` as production-like evidence
