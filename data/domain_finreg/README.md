# Domain: Prudential / Supervisory FinReg

This domain is the active research slice for the repository.

Focus:

- prudential supervision
- supervisory interpretation conflicts
- governance / RDARR / SREP / model-risk / remediation style questions

Current working question sets:

- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2.jsonl`
- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl`
- `benchmarks/finreg/full_rag_questions.jsonl`
- `benchmarks/finreg/controlled_candidate_cases.jsonl`

Current local configs:

- `config/gating_finreg_local_qwen15_rtx2070_section_rerank_smoke.yaml`
- `config/gating_finreg_local_qwen15_rtx2070_section_rerank_detector_smoke.yaml`
- `config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank_quality.yaml`

Run a small detector/gating smoke:

```bash
PYTHONPATH=. venv312/bin/python scripts/run_finreg_real_life_benchmark.py \
  --mode full-rag \
  --config gating_finreg_local_qwen15_rtx2070_section_rerank_detector_smoke \
  --k 8 \
  --limit 10 \
  --disable-answer-quality
```

For current orientation, use:

- `README.md`
- `docs/README.md`
- `docs/finreg_question_set_methodology.md`
