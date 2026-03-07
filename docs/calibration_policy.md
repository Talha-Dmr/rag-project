# Calibration Policy (Global-First, Outlier-Override)

This policy defines how we calibrate gating thresholds while domain count grows.

## Why This Exists

Thresholds that look good on one domain can drift on another. We therefore:
- keep one global/default operating point,
- allow small domain-specific overrides only when data shows clear outlier behavior,
- keep model stack fixed during calibration (same detector, same retrieval, same strategy).

## What Is Frozen During Calibration

The following are fixed in official calibration runs:
- detector architecture and checkpoint family (`balanced` track),
- reranker (`ebcar`),
- gating strategy (`retrieve_more`),
- epistemic signal type (`logit_mi`),
- retrieval stack and corpus.

Only gating thresholds are tuned.

## Decision Rule

1. Start with global/default thresholds on a new domain.
2. Run `20Q` quick check, then `50Q x 3 seeds` confirmation.
3. Mark domain as outlier if one holds:
- `abstain_rate > 0.70` (too conservative),
- `abstain_rate < 0.05` with elevated answered risk,
- contradiction metrics leave target bands.
4. Tune thresholds only for outliers, then re-confirm on `50Q x 3 seeds`.

## Target Bands (Operational)

- `abstain_rate`: roughly `0.05` to `0.25` for this high-stakes setup.
- `contradiction_rate` (stats_all): target close to `0.00`; practical guard `< 0.15`.
- `source_consistency`: should remain high and stable across seeds.

## Tuning Order (If Needed)

1. `contradiction_rate_threshold`
2. `uncertainty_threshold`
3. `contradiction_prob_threshold`
4. `source_consistency_threshold`

## Current Status (Feb 14, 2026, 50Q x seeds=7/11/19)

Canonical baseline reference:
- `docs/baseline_locked.md`

Source (generated):
- `docs/stability_report_50_default.md`
- `evaluation_results/auto_eval/seed_stability_summary_50_default.json`

- `health`
  - abstain mean/std: `0.1000 / 0.0432`
  - contradiction mean/std: `0.0160 / 0.0226`
- `finreg`
  - abstain mean/std: `0.0467 / 0.0377`
  - contradiction mean/std: `0.0000 / 0.0000`
- `disaster`
  - abstain mean/std: `0.0800 / 0.0589`
  - contradiction mean/std: `0.0000 / 0.0000`

Interpretation:
- All 3 domains are inside the current target bands on this proxy (no outlier flagged).
- `finreg` abstain mean is slightly below the nominal `0.05` floor, but still close; we monitor rather than override.

Frozen decision for this iteration:
- Keep the 3 domain configs active as-is (see `docs/baseline_locked.md`).
- No additional threshold sweeps unless a regression is observed.

## Health Calibration Update (crt=0.40 confirmed)

We confirmed a health-only override:
- `contradiction_rate_threshold = 0.40`

Confirmation runs (50Q, seeds `7/11/19`):
- `evaluation_results/auto_eval/health_logit_mi_50_seed7_crt040.json`
- `evaluation_results/auto_eval/health_logit_mi_50_seed11_crt040.json`
- `evaluation_results/auto_eval/health_logit_mi_50_seed19_crt040.json`

Observed (directionally):
- `contradiction_rate` dropped from ~`0.43` to near-`0.00` (seed-stable).
- `abstain_rate` increased modestly (still within/near target band).

## FinReg Calibration Update (crt=0.40 confirmed)

We confirmed a finreg-only override:
- `contradiction_rate_threshold = 0.40`

Why:
- With the previous finreg threshold (`1.01`), finreg can silently become unsafe (very high contradiction on a 20Q slice) because contradiction does not trigger `retrieve_more`.
- Lowering `crt` forces adaptive retrieval to spend budget and eliminates contradiction spikes.

Confirmation runs (50Q, seeds `7/11/19`):
- `evaluation_results/auto_eval/finreg_logit_mi_50_seed7_crt040.json`
- `evaluation_results/auto_eval/finreg_logit_mi_50_seed11_crt040.json`
- `evaluation_results/auto_eval/finreg_logit_mi_50_seed19_crt040.json`

Observed:
- `contradiction_rate`: `0.00` across all 3 seeds.
- `abstain_rate`: low-to-moderate (seed-dependent), but inside target band.

## Policy Decision (Current)

- Keep existing domain configs active:
  - `config/gating_health_ebcar_logit_mi_sc009.yaml`
  - `config/gating_finreg_ebcar_logit_mi_sc009.yaml`
  - `config/gating_disaster_ebcar_logit_mi_sc009.yaml`
- Candidate promotions are allowed only if `50Q x seeds(7,11,19)` confirmation stays inside target bands.
- Default reporting domains: `health` + `disaster`; `finreg` stays as stress-test.

## Historical Probes (Non-Canonical)

The sections below are historical probes/experiments from earlier iterations.
They are kept for context, but **must not** override the locked baseline.

### Quick Calibration Check (Historical Probe, Feb 12, 2026, seed=7, limit=20)

Source: `evaluation_results/auto_eval/calibration_quickcheck_seed7_limit20.json`

This was a fast contradiction-rate threshold probe (`0.85 / 0.95 / 1.01`) to verify whether the
current defaults still behave safely after detector updates.

Observed snapshot:

- `health`
  - `crt=0.85`: abstain `0.10`, contradiction `0.15`
  - `crt=0.95`: abstain `0.65`, contradiction `0.84` (degraded)
  - `crt=1.01`: abstain `0.20`, contradiction `0.01` (best risk profile)
- `finreg`
  - `crt=0.85`: abstain `0.05`, contradiction `0.00`
  - `crt=0.95`: abstain `0.10`, contradiction `0.00` (safe)
  - `crt=1.01`: abstain `0.05`, contradiction `0.78` (unsafe)
- `disaster`
  - `crt=0.85`: abstain `1.00`, contradiction `1.00` (unsafe)
  - `crt=0.95`: abstain `0.05`, contradiction `0.04` (best tradeoff)
  - `crt=1.01`: abstain `0.05`, contradiction `0.08`

Interim candidate thresholds (historical probe, to be confirmed before default switch):

- `health`: `1.01`
- `finreg`: `0.95`
- `disaster`: `0.95`

Important:

- This is a **single-seed quick check**. It is directional, not final.
- Final promotion criterion remains: `50Q x seeds(7,11,19)` confirmation per domain.

### 50Q Confirmation Update (Historical Probe, Feb 12, 2026)

Sources:
- `evaluation_results/auto_eval/health_logit_mi_50_seed7_crt101.json`
- `evaluation_results/auto_eval/health_logit_mi_50_seed11_crt101.json`
- `evaluation_results/auto_eval/health_logit_mi_50_seed19_crt101.json`
- `evaluation_results/auto_eval/disaster_logit_mi_50_seed7_crt095.json`
- `evaluation_results/auto_eval/disaster_logit_mi_50_seed11_crt095.json`
- `evaluation_results/auto_eval/disaster_logit_mi_50_seed19_crt095.json`

Observed:

- `health` with `crt=1.01` (50Q, seeds `7/11/19`)
  - abstain mean/std: `0.053 / 0.009`
  - contradiction mean/std: `0.664 / 0.003`
  - decision: **reject** (`crt=1.01` is unsafe for health)

- `disaster` with `crt=0.95` (50Q, seeds `7/11/19`)
  - abstain mean/std: `0.040 / 0.033`
  - contradiction mean/std: `0.000 / 0.000`
  - decision: **passes** confirmation

Implication (historical and explicit to promotion):

- `health: crt=1.01` is unsafe (`contradiction` high) and is **not promoted**.
- `disaster: crt=0.95` was stable in this probe window; we treated it as historical evidence only.
- `finreg: crt=0.95` was rejected as too conservative and high contradiction.

### FinReg 50Q Confirmation (Feb 12, 2026)

Sources:
- `evaluation_results/auto_eval/finreg_logit_mi_50_seed7_crt095.json`
- `evaluation_results/auto_eval/finreg_logit_mi_50_seed11_crt095.json`
- `evaluation_results/auto_eval/finreg_logit_mi_50_seed19_crt095.json`
- `evaluation_results/auto_eval/finreg_crt095_stability_summary.json`
- `docs/finreg_crt095_stability_report.md`

Observed (`crt=0.95`, 50Q, seeds `7/11/19`):

- abstain mean/std: `0.973 / 0.009`
- contradiction mean/std: `0.991 / 0.002`

Decision:

- **Reject** `finreg: crt=0.95` for this cycle (too conservative and still high contradiction signal).

### Frozen Decision (This Iteration)

- Calibration runs are **frozen** for now (no additional threshold sweeps in this cycle).
- Active threshold values are used as a **locked baseline** (not subject to changes in this sprint):
  - `health`: `contradiction_rate_threshold=0.40` (calibrated)
  - `finreg`: `contradiction_rate_threshold=0.40` (calibrated)
  - `disaster`: `contradiction_rate_threshold=1.01`
- Primary reporting domains: `health` + `disaster`.
- `finreg` remains in the project as a **stress-test domain** (monitored for robustness drift, but not primary KPI gate in this cycle).
- Next work focus shifts from threshold sweeps to detector robustness and policy stability.

### Detector Ablation Addendum (Feb 13, 2026)

- Executed: `run_detector_ablation_50.sh 7,11,19 50`
- Compared:
  - `gating_*_ebcar_logit_mi_sc009` (`balanced` checkpoint, copied as `*_balanced_50_seed*.json`)
  - `gating_*_ebcar_logit_mi_sc009_focaldet` (`focal`)
- Result:
  - `focal` is rejected:
    - `health`: `abstain=1.0`, `contradiction=1.0` across all 3 seeds.
    - `finreg`: contradiction ~`0.45` on avg.
  - `balanced` remains stable:
    - Across `health`, `finreg`, `disaster`: `contradiction_rate` stayed `0.0` in 3 seeds.
- Decision update:
  - Keep `balanced` as the default detector variant for this cycle.
  - Continue using logit-MI signal and current fixed gating flow for policy stability.
