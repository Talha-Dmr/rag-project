# Seed Stability Report (High-Stakes 3-Domain)

- Question set size(s): 50
- Generated: 2026-02-13T12:19:39
- Runs found: 9

## Per-Run Results

| Domain | Seed | Set | Tag | Abstain | Retrieve-More | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| disaster | 7 | 50 | - | 0.020 | 0.020 | 0.000 | 0.290 | 0.0166 | 0.753 |
| disaster | 11 | 50 | - | 0.160 | 0.160 | 0.000 | 0.293 | 0.0174 | 0.753 |
| disaster | 19 | 50 | - | 0.060 | 0.060 | 0.000 | 0.291 | 0.0167 | 0.753 |
| finreg | 7 | 50 | - | 0.000 | 0.000 | 0.004 | 0.299 | 0.0159 | 0.715 |
| finreg | 11 | 50 | - | 0.020 | 0.020 | 0.000 | 0.297 | 0.0165 | 0.715 |
| finreg | 19 | 50 | - | 0.060 | 0.060 | 0.000 | 0.215 | 0.0164 | 0.715 |
| health | 7 | 50 | - | 0.100 | 0.100 | 0.428 | 0.361 | 0.0166 | 0.728 |
| health | 11 | 50 | - | 0.080 | 0.080 | 0.408 | 0.360 | 0.0162 | 0.728 |
| health | 19 | 50 | - | 0.120 | 0.120 | 0.456 | 0.362 | 0.0162 | 0.728 |

## Domain Summary (mean ± std, with min/max)

| Domain | Metric | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| health | abstain_rate | 0.1000 | 0.0163 | 0.0800 | 0.1200 |
| health | retrieve_more_rate | 0.1000 | 0.0163 | 0.0800 | 0.1200 |
| health | contradiction_rate | 0.4307 | 0.0197 | 0.4080 | 0.4560 |
| health | contradiction_prob_mean | 0.3609 | 0.0007 | 0.3600 | 0.3617 |
| health | uncertainty_mean | 0.0163 | 0.0002 | 0.0162 | 0.0166 |
| health | source_consistency | 0.7284 | 0.0000 | 0.7284 | 0.7284 |
| finreg | abstain_rate | 0.0267 | 0.0249 | 0.0000 | 0.0600 |
| finreg | retrieve_more_rate | 0.0267 | 0.0249 | 0.0000 | 0.0600 |
| finreg | contradiction_rate | 0.0013 | 0.0019 | 0.0000 | 0.0040 |
| finreg | contradiction_prob_mean | 0.2706 | 0.0393 | 0.2150 | 0.2993 |
| finreg | uncertainty_mean | 0.0163 | 0.0003 | 0.0159 | 0.0165 |
| finreg | source_consistency | 0.7153 | 0.0000 | 0.7153 | 0.7153 |
| disaster | abstain_rate | 0.0800 | 0.0589 | 0.0200 | 0.1600 |
| disaster | retrieve_more_rate | 0.0800 | 0.0589 | 0.0200 | 0.1600 |
| disaster | contradiction_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| disaster | contradiction_prob_mean | 0.2914 | 0.0013 | 0.2905 | 0.2932 |
| disaster | uncertainty_mean | 0.0169 | 0.0004 | 0.0166 | 0.0174 |
| disaster | source_consistency | 0.7527 | 0.0000 | 0.7527 | 0.7527 |

## Quick Read

- `health`: seed-stable; balanced coverage; elevated contradiction risk.
- `finreg`: seed-stable; high coverage (low abstain); low contradiction risk.
- `disaster`: moderately stable; high coverage (low abstain); low contradiction risk.
- Cross-domain: contradiction spikes exist; keep calibration active.
