# Domain: Prudential / Supervisory FinReg

This domain is the active research slice for the repository.

Focus:

- prudential supervision
- supervisory interpretation conflicts
- governance / RDARR / SREP / model-risk / remediation style questions

Current working question sets:

- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2.jsonl`
- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl`

Canonical config:

- `config/gating_finreg_ebcar_logit_mi_sc009.yaml`

Run evaluation:

```bash
PYTHONPATH=. venv312/bin/python scripts/eval_grounding_proxy.py \
  --config gating_finreg_ebcar_logit_mi_sc009 \
  --questions data/domain_finreg/questions_finreg_conflict_phase1_refined_v2.jsonl \
  --limit 20 \
  --seed 7 \
  --output evaluation_results/auto_eval/finreg_domain_seed7.json
```

For the current baseline summary, use:

- `docs/current_finreg_baseline.md`
