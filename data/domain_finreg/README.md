# Domain: Financial Regulation / Compliance (High-Stakes)

This domain targets risk/compliance interpretation conflicts across regulatory and supervisory texts.

Suggested primary sources:
- BIS / BCBS principles and standards (including BCBS239)
- EBA supervisory guidelines
- Federal Reserve / ECB supervisory publications
- IFRS sustainability and risk disclosure standards (where applicable)

Initial question set:
- `data/domain_finreg/questions_finreg_conflict.jsonl`

Run evaluation (after indexing this domain corpus):
- `PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy.py --config gating_finreg_ebcar_logit_mi_sc009 --questions data/domain_finreg/questions_finreg_conflict.jsonl --limit 20 --seed 7 --output evaluation_results/auto_eval/finreg_logit_mi_20.json`
