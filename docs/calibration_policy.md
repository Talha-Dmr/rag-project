# Calibration Policy (FinReg)

This policy defines how gating thresholds should be calibrated for the active FinReg baseline.

## Frozen Inputs

Official calibration runs hold the following fixed:

- config family: `config/gating_finreg_ebcar_logit_mi_sc009.yaml`
- corpus: real phase-1.5 prudential / supervisory FinReg corpus
- chunking: `section_aware`
- retrieval first-pass `k`: `20`
- reranker: `ebcar`
- generator: local `Qwen/Qwen2.5-1.5B-Instruct`
- detector: FEVER `DeBERTa-v3-base`
- gating strategy: `retrieve_more`
- epistemic signal: `logit_mi`

Only gating thresholds should move during a calibration pass.

## Current Reference Runs

Canonical baseline reference:

- `docs/current_finreg_baseline.md`

Current confirmation files:

- `evaluation_results/auto_eval/finreg_phase15_refined_v2_default_seed7.json`
- `evaluation_results/auto_eval/finreg_phase15_refined_v2_default_seed11.json`
- `evaluation_results/auto_eval/finreg_phase15_refined_v2_default_seed19.json`
- `evaluation_results/auto_eval/finreg_phase15_refined_v2_50_default_seed7.json`
- `evaluation_results/auto_eval/finreg_phase15_refined_v2_50_default_seed11.json`
- `evaluation_results/auto_eval/finreg_phase15_refined_v2_50_default_seed19.json`

## Decision Rule

1. Run `20Q refined v2` as a quick probe.
2. If behavior looks plausible, confirm on `50Q refined v2 x 3 seeds`.
3. Change thresholds only when there is a clear safety or collapse issue.
4. Reconfirm on `50Q x 3 seeds` after every threshold change.

## Operating Targets

These are working policy bands, not hard theoretical truths.

- `abstain_rate`: roughly `0.20` to `0.35`
- `retrieve_more`: should remain meaningfully used on the `50Q` set
- `contradiction_rate` (`stats_all`): practical guard `< 0.15`
- `source_consistency`: should stay high and not swing sharply across seeds

## Tuning Order

If tuning is needed, move in this order:

1. `contradiction_prob_threshold`
2. `uncertainty_threshold`
3. `source_consistency_threshold`
4. `contradiction_rate_threshold`

## Current Working Read

On the current phase-1.5 corpus:

- `20Q refined v2` mean abstain is about `0.30`
- `50Q refined v2` mean abstain is about `0.26`
- `retrieve_more` remains active rather than collapsing to always-answer or always-abstain

This means the current baseline is conservative but operational. Threshold tuning should now be
driven by question-level failure analysis rather than broad exploratory sweeps.
