# FinReg Phase-1 Corpus Runbook

## Purpose

This runbook turns the phase-1 finreg source inventory into a real, rebuildable corpus.

It is the first operational step after the planning documents:

- `docs/finreg_phase1_document_inventory.md`
- `docs/finreg_corpus_build_plan.md`

## Tracked Inputs

- source manifest:
  - `config/finreg_phase1_sources.yaml`
- fetch script:
  - `scripts/fetch_finreg_phase1_sources.py`
- corpus build script:
  - `scripts/build_finreg_phase1_corpus.py`

## Output Layout

Raw downloads:

```text
data/raw/finreg/
  bcbs/
  eba/
  pra_boe/
  ecb/
  fetch_manifest.json
```

Processed artifacts:

```text
data/processed/finreg/
  bcbs/
  eba/
  pra_boe/
  ecb/
  finreg_phase1_corpus.jsonl
```

Per document, the processed tree contains:

- `.txt`
- `.pages.json`
- `.metadata.json`

## Phase-1 Commands

Fetch all phase-1 source documents:

```bash
PYTHONPATH=. venv312/bin/python scripts/fetch_finreg_phase1_sources.py
```

Fetch only one or two documents while iterating:

```bash
PYTHONPATH=. venv312/bin/python scripts/fetch_finreg_phase1_sources.py \
  --document-id bcbs239 \
  --document-id ecb_rdarr_guide
```

Build processed corpus from the fetched raw files:

```bash
PYTHONPATH=. venv312/bin/python scripts/build_finreg_phase1_corpus.py
```

Build only a small subset during smoke testing:

```bash
PYTHONPATH=. venv312/bin/python scripts/build_finreg_phase1_corpus.py \
  --document-id bcbs239 \
  --document-id ecb_rdarr_guide
```

Index the processed corpus into the finreg vector DB:

```bash
PYTHONPATH=. venv312/bin/python scripts/index_domain_corpus.py \
  --config gating_finreg_ebcar_logit_mi_sc009 \
  --corpus data/processed/finreg/finreg_phase1_corpus.jsonl \
  --reset-collection
```

## Notes

- The manifest intentionally mixes direct PDFs and official HTML pages.
- Some HTML entries are hub or landing pages. They are useful as provisional source coverage, but lower quality than direct guides or statements.
- The processed JSONL is the current bridge into the existing indexing pipeline. It is not meant to replace richer per-document metadata over time.

## Near-Term Follow-Up

After the first successful phase-1 build:

1. remove weak landing pages where better direct document URLs exist
2. add missing BoE/PRA reporting materials for `fq10`, `fq14`, `fq15`, `fq17`
3. rebuild finreg question coverage against the real corpus
4. rerun retrieval and gating evaluation only after the new index is in place
