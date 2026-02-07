# Domain: Health Guidelines (High-Stakes)

This domain targets conflicts and uncertainty across public-health and clinical guidance sources.

Suggested primary sources:
- WHO guidelines and living recommendations
- CDC recommendations and MMWR reports
- NICE guidelines
- ECDC technical guidance

Initial question set:
- `data/domain_health/questions_health_conflict.jsonl`

Run evaluation (after indexing this domain corpus):
- `PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy.py --config gating_health_ebcar_logit_mi_sc009 --questions data/domain_health/questions_health_conflict.jsonl --limit 20 --seed 7 --output evaluation_results/auto_eval/health_logit_mi_20.json`
