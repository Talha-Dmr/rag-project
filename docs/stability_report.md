# Seed Stability Report (High-Stakes 3-Domain)

- Generated: 2026-02-08T16:21:45
- Runs found: 9

## Per-Run Results

| Domain | Seed | Abstain | Retrieve-More | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| disaster | 7 | 0.100 | 0.100 | 0.904 | 0.377 | 0.0167 | 0.753 |
| disaster | 11 | 0.080 | 0.080 | 0.000 | 0.269 | 0.0167 | 0.753 |
| disaster | 19 | 0.040 | 0.040 | 0.732 | 0.352 | 0.0165 | 0.753 |
| finreg | 7 | 0.000 | 0.000 | 0.000 | 0.309 | 0.0160 | 0.715 |
| finreg | 11 | 0.100 | 0.100 | 0.516 | 0.379 | 0.0169 | 0.715 |
| finreg | 19 | 0.980 | 0.980 | 0.996 | 0.411 | 0.0173 | 0.715 |
| health | 7 | 0.040 | 0.040 | 0.004 | 0.297 | 0.0166 | 0.728 |
| health | 11 | 0.000 | 0.000 | 0.000 | 0.324 | 0.0160 | 0.728 |
| health | 19 | 0.180 | 0.180 | 0.556 | 0.366 | 0.0168 | 0.728 |

## Domain Summary (mean ± std, with min/max)

| Domain | Metric | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| health | abstain_rate | 0.0733 | 0.0772 | 0.0000 | 0.1800 |
| health | retrieve_more_rate | 0.0733 | 0.0772 | 0.0000 | 0.1800 |
| health | contradiction_rate | 0.1867 | 0.2612 | 0.0000 | 0.5560 |
| health | contradiction_prob_mean | 0.3291 | 0.0284 | 0.2972 | 0.3661 |
| health | uncertainty_mean | 0.0165 | 0.0003 | 0.0160 | 0.0168 |
| health | source_consistency | 0.7284 | 0.0000 | 0.7284 | 0.7284 |
| finreg | abstain_rate | 0.3600 | 0.4403 | 0.0000 | 0.9800 |
| finreg | retrieve_more_rate | 0.3600 | 0.4403 | 0.0000 | 0.9800 |
| finreg | contradiction_rate | 0.5040 | 0.4067 | 0.0000 | 0.9960 |
| finreg | contradiction_prob_mean | 0.3667 | 0.0425 | 0.3095 | 0.4114 |
| finreg | uncertainty_mean | 0.0167 | 0.0005 | 0.0160 | 0.0173 |
| finreg | source_consistency | 0.7153 | 0.0000 | 0.7153 | 0.7153 |
| disaster | abstain_rate | 0.0733 | 0.0249 | 0.0400 | 0.1000 |
| disaster | retrieve_more_rate | 0.0733 | 0.0249 | 0.0400 | 0.1000 |
| disaster | contradiction_rate | 0.5453 | 0.3920 | 0.0000 | 0.9040 |
| disaster | contradiction_prob_mean | 0.3326 | 0.0464 | 0.2686 | 0.3771 |
| disaster | uncertainty_mean | 0.0167 | 0.0001 | 0.0165 | 0.0167 |
| disaster | source_consistency | 0.7527 | 0.0000 | 0.7527 | 0.7527 |

## Quick Read

- `health`: relatively stable, low abstain on most seeds.
- `finreg`: unstable; one seed can swing to extreme abstention.
- `disaster`: abstain stable-low, contradiction signal can vary by seed.

## Follow-up Calibration (Feb 8, 2026)

- FinReg targeted sweep (`seed=19`, `limit=20`) selected `contradiction_rate_threshold=1.01` as best local fix:
  - `0.85` -> `abstain_rate=0.15`, `contradiction_rate=0.24`
  - `0.95` -> `abstain_rate=0.15`, `contradiction_rate=0.31`
  - `1.01` -> `abstain_rate=0.05`, `contradiction_rate=0.00` (selected)
  - `1.10` -> `abstain_rate=0.05`, `contradiction_rate=0.19`
- 50Q confirm run (`seed=19`) also supports the override:
  - old (`0.85`): `abstain_rate=0.98`, `contradiction_rate=0.996`
  - new (`1.01`): `abstain_rate=0.04`, `contradiction_rate=0.016`
- Quick cross-seed check (`limit=20`) under new threshold:
  - `seed=7`: `abstain_rate=0.05`, `contradiction_rate=0.00`
  - `seed=11`: `abstain_rate=0.00`, `contradiction_rate=0.00`
  - `seed=19`: `abstain_rate=0.05`, `contradiction_rate=0.00`
- Remaining validation: re-run all three seeds (`7/11/19`) on full 50Q with the new finreg threshold.
