# Locked Baseline (FinReg)

This file now only defines the pointer to the canonical active baseline.

For the full current state, use:

- `docs/current_finreg_baseline.md`

## Canonical Locked Config

- `config/gating_finreg_ebcar_logit_mi_sc009.yaml`

## Sanity Check

- `PYTHONPATH=. venv312/bin/python scripts/check_locked_baseline.py`

## Important Note

Older references in this repository that still point to:

- `data/corpora/finreg_corpus.jsonl`
- the synthetic bootstrap corpus
- older 20Q / 50Q files
- early FEVER probe outputs

should be treated as **historical only** unless explicitly regenerated on the current real phase-1 corpus.
