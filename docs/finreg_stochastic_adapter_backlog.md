# FinReg Stochastic Adapter Backlog

Date: 2026-05-28

## Purpose

Track the stochastic gating adapters that should remain active research candidates, so they do not
get forgotten while the main gating work continues.

This backlog is not the current production path. The current production-like baseline remains:

- `logit_mi`
- `guarded_v3`
- evidence subset sampling
- one retry

## Current Shortlist

Primary candidates:

1. `stochastic_sgbd`
   - Best LFM2 no-type replay result on 2026-05-28.
   - Candidate role: more conservative conflict-sensitive scalar adapter.

2. `stochastic_sghmc`
   - Good LFM2 no-type replay result with a less conservative action mix than `stochastic_sgbd`.
   - Candidate role: middle-ground stochastic adapter.

3. `stochastic_mirror_langevin`
   - Best refreshed result on the older 50Q shadow dump.
   - Candidate role: check whether mirror/simplex geometry is more stable across non-LFM2 dumps.

Secondary ablation:

4. `stochastic_wright_fisher`
   - Not the strongest in the latest LFM2 replay.
   - Keep as lower-priority ablation because it represents a different simplex-diffusion idea.

Lower priority unless new evidence appears:

- `stochastic_langevin`
- `stochastic_ou`
- `stochastic_prox_langevin`

## Why They Are Not Promoted Yet

The best adapter changes by evaluation source:

- LFM2 no-type replay: `stochastic_sgbd` was best.
- Older 50Q shadow dump: `stochastic_mirror_langevin` was best.

That means there is signal worth testing, but not enough stability to replace `logit_mi` in the
production-like path.

## When To Revisit

Revisit after the current gating track reaches a stable no-type baseline:

1. `guarded_v3` applied run is accepted as the current baseline.
2. `guarded_v4_lite` is either dropped or redesigned without dataset `type` metadata.
3. The next no-type gating candidate has been compared against `guarded_v3`.

At that point, run the stochastic shortlist as the next ablation block before making another
FullRAG80 claim.

FullRAG80 is now the canonical test dataset:

- `benchmarks/finreg/full_rag_questions.jsonl`

The 50-question current set can be used only as a fast smoke/replay tool. It should not decide
whether a stochastic adapter is promoted.

## Next Test Block

Run these tests in order.

1. Multi-seed replay on existing current50 details
   - seeds: `7`, `11`, `19` if matching dumps exist or can be generated
   - sources: `logit_mi`, `stochastic_sgbd`, `stochastic_sghmc`, `stochastic_mirror_langevin`,
     `stochastic_wright_fisher`
   - report: operating score, action mix, label-aware utility, action agreement with `logit_mi`,
     answer/retrieve-more accuracy

2. FullRAG80 no-type applied run for the top 1-2 adapters
   - same model: LFM2 2.6B
   - dataset: `benchmarks/finreg/full_rag_questions.jsonl`
   - same seed and retry settings as the `guarded_v3` baseline
   - no benchmark metadata at runtime

3. Optional 50Q diagnostic follow-up
   - use only to explain a failure mode or speed up debugging
   - do not use it as the final deciding result

## Acceptance Criteria

Promote a stochastic adapter only if it satisfies all of these:

- improves or matches `logit_mi` on FullRAG80 applied runs,
- does not increase sanity false abstains,
- reduces unsupported/hallucination risk among answered questions or improves retrieve-more
  behavior on conflict questions,
- is stable across at least two seeds,
- does not depend on dataset labels, `type`, expected behavior, or benchmark ids at runtime,
- has a clear implementation path in shared runtime code, not only in replay scripts.

## Current Artifacts

Retest artifacts from 2026-05-28:

- `evaluation_results/auto_eval/lfm2_stochastic_replay_current50_notype_seed7_wide.json`
- `evaluation_results/auto_eval/lfm2_stochastic_adapter_overlap_current50_notype_seed7.json`
- `evaluation_results/auto_eval/finreg_stochastic_replay_current50_seed7_wide_refresh.json`

Related diagnostic note:

- `docs/finreg_stochastic_gating_diagnostic_note.md`
