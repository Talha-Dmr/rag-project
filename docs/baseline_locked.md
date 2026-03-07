# Locked Baseline (High-Stakes Track)

This document defines the **canonical locked baseline** for the high-stakes track for the current iteration.

If something conflicts with this doc, treat it as **historical** unless it is regenerated from the same scripts and inputs.

## What Is Locked

- Domains (3-domain mix):
  - Health guidelines: `config/gating_health_ebcar_logit_mi_sc009.yaml` (primary)
  - Disaster/climate risk: `config/gating_disaster_ebcar_logit_mi_sc009.yaml` (primary)
  - Financial regulation/compliance: `config/gating_finreg_ebcar_logit_mi_sc009.yaml` (stress-test)
- Epistemic signal: `logit_mi`
- Gating strategy: `retrieve_more`
- Reranker: `ebcar`
- Hallucination detector checkpoint family: `balanced` (see `hallucination_detector.model_path` in configs)

## Baseline Thresholds (Read From Configs)

Do not duplicate these numbers in multiple docs; the source of truth is the YAML config files:

- `health`: `contradiction_rate_threshold=0.40`
- `finreg`: `contradiction_rate_threshold=0.40`
- `disaster`: `contradiction_rate_threshold=1.01`

Other thresholds are also config-owned (uncertainty, contradiction_prob, source_consistency).

## Evidence (50Q x seeds 7/11/19)

Canonical stability report and JSON summary:

- `docs/stability_report_50_default.md`
- `evaluation_results/auto_eval/seed_stability_summary_50_default.json`

Detector ablation (balanced vs focal), 50Q x seeds 7/11/19:

- `docs/detector_ablation_report_50.md`


## How To Reproduce

0. (Optional) Verify configs still match the locked baseline expectations:
   - `PYTHONPATH=. venv312/bin/python scripts/check_locked_baseline.py`

1. Ensure each domain index exists (or index once):
   - `PYTHONPATH=. venv312/bin/python -u scripts/index_domain_corpus.py --config gating_health_ebcar_logit_mi_sc009 --corpus data/corpora/health_corpus.jsonl`
   - `PYTHONPATH=. venv312/bin/python -u scripts/index_domain_corpus.py --config gating_finreg_ebcar_logit_mi_sc009 --corpus data/corpora/finreg_corpus.jsonl`
   - `PYTHONPATH=. venv312/bin/python -u scripts/index_domain_corpus.py --config gating_disaster_ebcar_logit_mi_sc009 --corpus data/corpora/disaster_corpus.jsonl`

2. Run 50Q x 3 seeds for all 3 domains:
   - `./scripts/run_high_stakes_seed_matrix.sh 50 50 7,11,19 all`

3. Summarize into the canonical report:
   - `PYTHONPATH=. venv312/bin/python scripts/summarize_seed_stability.py --input-glob 'evaluation_results/auto_eval/*_logit_mi_50_seed*.json' --json-out evaluation_results/auto_eval/seed_stability_summary_50_default.json --markdown-out docs/stability_report_50_default.md`

## Non-Defaults

The following are experimental and must not be treated as defaults unless explicitly testing:

- `config/gating_*_ebcar_logit_mi_sc009_focaldet.yaml`
- `config/gating_*_ebcar_logit_mi_sc009_neutralguarddet.yaml`
- `config/gating_*_ebcar_rep_mi_sc009.yaml`
