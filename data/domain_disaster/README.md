# Domain: Disaster and Climate Risk (High-Stakes)

This domain targets uncertainty and conflict across hazard outlooks, disaster-risk frameworks, and climate risk scenarios.

Suggested primary sources:
- NOAA outlook products and climate diagnostics
- IPCC synthesis and regional risk assessments
- UNDRR Global Assessment Report
- WMO climate updates and hazard bulletins

Initial question set:
- `data/domain_disaster/questions_disaster_conflict.jsonl`

Run evaluation (after indexing this domain corpus):
- `PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy.py --config gating_disaster_ebcar_logit_mi_sc009 --questions data/domain_disaster/questions_disaster_conflict.jsonl --limit 20 --seed 7 --output evaluation_results/auto_eval/disaster_logit_mi_20.json`
