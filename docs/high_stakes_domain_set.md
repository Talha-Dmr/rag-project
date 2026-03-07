# High-Stakes Domain Set (Current)

This project now uses a high-stakes domain mix designed for stronger uncertainty-aware RAG evaluation:

1. Health guidelines (`domain_health`) - primary
2. Disaster and climate risk (`domain_disaster`) - primary
3. Financial regulation/compliance (`domain_finreg`) - stress-test track

Rationale:
- High decision cost if answers are wrong.
- Natural source conflict and uncertainty in all three domains.
- Better domain-shift coverage than keeping both macro and energy together.
- FinReg is retained as a higher-drift stress-test coverage track.

## Current Assets

Canonical baseline reference:
- `docs/baseline_locked.md`

Locked default domain configs:
- `config/gating_health_ebcar_logit_mi_sc009.yaml`
- `config/gating_disaster_ebcar_logit_mi_sc009.yaml`
- `config/gating_finreg_ebcar_logit_mi_sc009.yaml`

Bootstrap corpus builder:
- `scripts/build_high_stakes_bootstrap_corpora.py`

Question sets:
- Health:
  - 20Q: `data/domain_health/questions_health_conflict.jsonl`
  - 50Q: `data/domain_health/questions_health_conflict_50.jsonl`
- FinReg:
  - 20Q: `data/domain_finreg/questions_finreg_conflict.jsonl`
  - 50Q: `data/domain_finreg/questions_finreg_conflict_50.jsonl`
- Disaster:
  - 20Q: `data/domain_disaster/questions_disaster_conflict.jsonl`
  - 50Q: `data/domain_disaster/questions_disaster_conflict_50.jsonl`

## Recommended Next Execution Order

1. Index one domain corpus at a time into its dedicated vector store directory.
2. Run proxy grounding eval on the 20Q seed set for quick sanity checks.
3. Expand each seed set to 50Q and confirm on `seed=7,11,19` after any config/domain change.
4. Run controlled cross-domain ablation only when change justification exists:
   - `nogate` vs `retrieve_more` vs `abstain`
   - `logit_mi` vs `rep_mi` (same question slices)

## Example Commands

Index first (required):
- `PYTHONPATH=. venv312/bin/python scripts/build_high_stakes_bootstrap_corpora.py`
- `PYTHONPATH=. venv312/bin/python scripts/index_domain_corpus.py --config gating_health_ebcar_logit_mi_sc009 --corpus data/corpora/health_corpus.jsonl --reset-collection`
- `PYTHONPATH=. venv312/bin/python scripts/index_domain_corpus.py --config gating_finreg_ebcar_logit_mi_sc009 --corpus data/corpora/finreg_corpus.jsonl --reset-collection`
- `PYTHONPATH=. venv312/bin/python scripts/index_domain_corpus.py --config gating_disaster_ebcar_logit_mi_sc009 --corpus data/corpora/disaster_corpus.jsonl --reset-collection`

Single-command helper:
- `./scripts/run_high_stakes_seed_eval.sh all 20 7`
- `./scripts/run_high_stakes_seed_matrix.sh 50 50 7,11,19 all`

Note:
- `eval_grounding_proxy.py` now stops early if the target collection is empty.
- Use `--allow-empty-index` only for debugging.

Health:
- `PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy.py --config gating_health_ebcar_logit_mi_sc009 --questions data/domain_health/questions_health_conflict.jsonl --limit 20 --seed 7 --output evaluation_results/auto_eval/health_logit_mi_20.json`

Financial regulation:
- `PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy.py --config gating_finreg_ebcar_logit_mi_sc009 --questions data/domain_finreg/questions_finreg_conflict.jsonl --limit 20 --seed 7 --output evaluation_results/auto_eval/finreg_logit_mi_20.json`

Disaster/climate:
- `PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy.py --config gating_disaster_ebcar_logit_mi_sc009 --questions data/domain_disaster/questions_disaster_conflict.jsonl --limit 20 --seed 7 --output evaluation_results/auto_eval/disaster_logit_mi_20.json`

## Canonical Reports

Do not copy metric tables into this doc (they drift quickly). Use the canonical generated reports:

- Stability (50Q x seeds 7/11/19): `docs/stability_report_50_default.md`
- Calibration policy: `docs/calibration_policy.md`
- Detector ablation (balanced vs focal): `docs/detector_ablation_report_50.md`
