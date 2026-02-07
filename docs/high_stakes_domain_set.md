# High-Stakes 3-Domain Set (Active)

This project now uses a high-stakes domain mix designed for stronger uncertainty-aware RAG evaluation:

1. Health guidelines (`domain_health`)
2. Financial regulation/compliance (`domain_finreg`)
3. Disaster and climate risk (`domain_disaster`)

Rationale:
- High decision cost if answers are wrong.
- Natural source conflict and uncertainty in all three domains.
- Better domain-shift coverage than keeping both macro and energy together.

## Current Assets

- Configs (balanced SGLD detector, domain-tuned where needed):
  - `config/gating_health_ebcar_logit_mi_sc009.yaml`
  - `config/gating_finreg_ebcar_logit_mi_sc009.yaml`
  - `config/gating_disaster_ebcar_logit_mi_sc009.yaml`
  - Note: `disaster` uses `contradiction_rate_threshold: 1.01` (promoted override).
- Bootstrap corpus builder:
  - `scripts/build_high_stakes_bootstrap_corpora.py`
- Question seeds (20Q each):
  - `data/domain_health/questions_health_conflict.jsonl`
  - `data/domain_finreg/questions_finreg_conflict.jsonl`
  - `data/domain_disaster/questions_disaster_conflict.jsonl`

## Recommended Next Execution Order

1. Index one domain corpus at a time into its dedicated vector store directory.
2. Run proxy grounding eval on the 20Q seed set.
3. Tune thresholds per domain (especially contradiction and source-consistency).
4. Expand each seed set to 50Q after first stable pass.
5. Run cross-domain ablation:
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

Note:
- `eval_grounding_proxy.py` now stops early if the target collection is empty.
- Use `--allow-empty-index` only for debugging.

Health:
- `PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy.py --config gating_health_ebcar_logit_mi_sc009 --questions data/domain_health/questions_health_conflict.jsonl --limit 20 --seed 7 --output evaluation_results/auto_eval/health_logit_mi_20.json`

Financial regulation:
- `PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy.py --config gating_finreg_ebcar_logit_mi_sc009 --questions data/domain_finreg/questions_finreg_conflict.jsonl --limit 20 --seed 7 --output evaluation_results/auto_eval/finreg_logit_mi_20.json`

Disaster/climate:
- `PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy.py --config gating_disaster_ebcar_logit_mi_sc009 --questions data/domain_disaster/questions_disaster_conflict.jsonl --limit 20 --seed 7 --output evaluation_results/auto_eval/disaster_logit_mi_20.json`

## Latest Seed Results (50Q each, Feb 7, 2026)

| Domain | Abstain | Source consistency | Contradiction rate | Uncertainty mean |
| --- | --- | --- | --- | --- |
| Health | 4/50 (0.08) | 0.728 | 0.00 | 0.0162 |
| Financial regulation | 1/50 (0.02) | 0.715 | 0.008 | 0.0160 |
| Disaster/climate | 1/50 (0.02) | 0.753 | 0.00 | 0.0162 |

Interpretation:
- The balanced detector + logit-MI path gives low abstain on all three high-stakes domains at 50Q.
- Disaster outlier behavior was fixed by a single domain-specific contradiction-rate override.

Decision:
- Keep `logit_mi` + balanced detector as default for this trio.
- Keep disaster override (`contradiction_rate_threshold: 1.01`) active.
- Next pass: multi-seed confirmation (seed sweep) before any further threshold changes.
