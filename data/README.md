# Data Guide

This repository tracks only small, curated data needed to understand and
reproduce the FinReg experiments. Large downloaded corpora, model caches, vector
stores, and generated reports are local artifacts.

## Current FinReg Data

- `data/processed/finreg/`: processed official-source FinReg corpus text,
  pages, and metadata.
- `data/processed/finreg/finreg_phase1_corpus.jsonl`: canonical JSONL corpus
  used for section-aware indexing.
- `data/domain_finreg/`: FinReg question sets, detector candidate pools, and
  manual review files.

## Historical Data

- `data/ambiguity_datasets/`: ambiguity datasets used in earlier exploration.
- `data/fever/`: FEVER pair-NLI format notes/data.
- `data/training/`: generated detector training datasets. Some files may be
  local/generated rather than committed source.

## Local Generated Data

- `data/vector_db/`: Chroma/vector indexes. Rebuild with `scripts/index_domain_corpus.py`.
- `data/raw/`: downloaded raw sources. Rebuild/fetch where possible.

Do not assume local generated folders are present on a fresh clone.
