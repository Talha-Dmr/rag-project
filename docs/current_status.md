# Current Status

Last updated: 2026-05-20

## Active Goal

Build and evaluate a FinReg RAG pipeline where retrieval, generation,
hallucination detection, and stochastic/abstain gating can be tested separately
and together.

## Corpus

- Scope: prudential / supervisory FinReg.
- Sources: official BCBS, EBA, ECB, PRA-BoE, and selected Fed/OCC documents.
- Current processed corpus: `data/processed/finreg/finreg_phase1_corpus.jsonl`.
- Current section-aware vector index: `data/vector_db/domain_finreg_real_section`.
- Current Chroma collection: `rag_finreg_real_section`.

## Current Local Smoke Results

The latest RTX2070 detector/gating smoke used:

- config: `gating_finreg_local_qwen15_rtx2070_section_rerank_detector_smoke`
- benchmark: `benchmarks/finreg/full_rag_questions.jsonl`
- limit: `10`
- answer-quality rewrite: disabled

Observed summary:

- detector run rate: `1.0`
- abstain rate: `0.4`
- expected behavior match rate: `0.9`
- forbidden claim hit rate: `0.0`
- mean answer-include risk: `0.677`

Report path:

- `reports/finreg_real_life_benchmark/smoke10_qwen15_bf16_rtx2070_section_rerank_detector_gate/summary.json`

## Known Current Issues

- The RTX2070 smoke path uses Qwen 1.5B for local feasibility; it is not the
  final quality model.
- Qwen 1.5B generated degenerate repeated punctuation with fp16 on the local
  machine, so the smoke config uses `dtype: bfloat16`.
- `fullrag_009` failed the automatic expected-behavior check in the 10-question
  smoke; this needs manual review to separate generation weakness from metric
  strictness.
- High-risk answered cases such as `fullrag_007` should be used for gate
  threshold calibration.
- Large local artifacts such as model checkpoints, vector stores, reports, and
  detector asset folders are not expected to exist on a fresh clone.

## Next Work

1. Manually inspect the high-risk answered cases and the single automatic
   failure from the 10-question smoke.
2. Tune detector/gate thresholds on a small calibration set.
3. Run a larger 40-question full-RAG benchmark once local hardware constraints
   are acceptable or a stronger machine is available.
4. Keep corpus changes targeted; the current corpus is sufficient for detector
   and gating research unless a specific hard case exposes a missing document.
