# Calibration Policy (Global-First, Outlier-Override)

This policy defines how to calibrate gating thresholds as domain count grows.

## Why This Exists

A single global threshold set does not always transfer across domains.
In recent runs, the same settings produced:
- health: low abstain
- finreg: very low abstain
- disaster: extreme abstain

So we use a scalable approach: global defaults first, domain overrides only for outliers.

## Decision Rule

1. Start every new domain with global default thresholds.
2. Run a fixed seed evaluation slice (at least 20Q, then 50Q for confirmation).
3. Mark a domain as outlier if at least one condition holds:
- `abstain_rate > 0.70` (too conservative), or
- `abstain_rate < 0.05` with elevated answered risk, or
- answered contradiction metrics violate target bands.
4. Apply domain-specific threshold tuning only to outliers.
5. Keep algorithm and detector fixed; tune only gating thresholds first.

## Target Bands (Current)

These are operational bands, not hard theory limits:
- `abstain_rate`: `0.10` to `0.45` (domain-dependent, but avoid extremes)
- answered `contradiction_rate`: as low as possible; target `< 0.15`
- answered `source_consistency`: should stay meaningfully above abstained bucket
- coverage-risk tradeoff should improve vs global baseline

## Tuning Order (Per Outlier Domain)

Tune in this order to reduce overfitting:
1. `contradiction_rate_threshold`
2. `uncertainty_threshold`
3. `contradiction_prob_threshold`
4. `source_consistency_threshold` (last, because it can be domain-shape sensitive)

Keep `strategy`, detector checkpoint, and retrieval stack unchanged during this step.

## Evaluation Protocol

For each outlier domain:
1. Short sweep on 20Q (`seed=7`) for directional signal.
2. Pick 1-2 candidate settings.
3. Confirm on 50Q with same seed.
4. Promote only if metrics improve without creating a new extreme.

## Artifact Expectations

For each calibration cycle, store:
- baseline JSON result
- candidate JSON results
- final selected config file
- short markdown summary (what changed and why)

## Promotion Criteria

Promote domain-specific override if:
1. it fixes outlier behavior,
2. answered-risk metrics do not regress materially,
3. behavior is stable on rerun with same seed/slice.

Otherwise keep global defaults and mark domain as unresolved.

## Applied Example (Feb 7, 2026)

- Domain: `disaster`
- Baseline (`gating_disaster_ebcar_logit_mi_sc009`): `abstain_rate=0.98` on 50Q
- Tuned override: `contradiction_rate_threshold=1.01`
- Confirmed result: `abstain_rate=0.02` on 50Q with stable detector behavior

This is a valid outlier-fix under this policy: single-threshold override, same detector and retrieval stack.
