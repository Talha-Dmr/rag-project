# Seed Stability Report (High-Stakes 3-Domain)

- Question set size(s): 50
- Generated: 2026-02-14T22:38:16
- Runs found: 9

## Per-Run Results

| Domain | Seed | Set | Tag | Abstain | Retrieve-More | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| disaster | 7 | 50 | - | 0.020 | 0.020 | 0.000 | 0.290 | 0.0166 | 0.753 |
| disaster | 11 | 50 | - | 0.160 | 0.160 | 0.000 | 0.293 | 0.0174 | 0.753 |
| disaster | 19 | 50 | - | 0.060 | 0.060 | 0.000 | 0.291 | 0.0167 | 0.753 |
| finreg | 7 | 50 | - | 0.100 | 0.100 | 0.000 | 0.297 | 0.0162 | 0.715 |
| finreg | 11 | 50 | - | 0.020 | 0.020 | 0.000 | 0.299 | 0.0161 | 0.715 |
| finreg | 19 | 50 | - | 0.020 | 0.020 | 0.000 | 0.299 | 0.0159 | 0.715 |
| health | 7 | 50 | - | 0.140 | 0.140 | 0.000 | 0.281 | 0.0173 | 0.728 |
| health | 11 | 50 | - | 0.040 | 0.040 | 0.000 | 0.305 | 0.0158 | 0.728 |
| health | 19 | 50 | - | 0.120 | 0.120 | 0.048 | 0.342 | 0.0169 | 0.728 |

## Domain Summary (mean ± std, with min/max)

| Domain | Metric | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| health | abstain_rate | 0.1000 | 0.0432 | 0.0400 | 0.1400 |
| health | retrieve_more_rate | 0.1000 | 0.0432 | 0.0400 | 0.1400 |
| health | contradiction_rate | 0.0160 | 0.0226 | 0.0000 | 0.0480 |
| health | contradiction_prob_mean | 0.3096 | 0.0253 | 0.2809 | 0.3425 |
| health | uncertainty_mean | 0.0167 | 0.0006 | 0.0158 | 0.0173 |
| health | source_consistency | 0.7284 | 0.0000 | 0.7284 | 0.7284 |
| finreg | abstain_rate | 0.0467 | 0.0377 | 0.0200 | 0.1000 |
| finreg | retrieve_more_rate | 0.0467 | 0.0377 | 0.0200 | 0.1000 |
| finreg | contradiction_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| finreg | contradiction_prob_mean | 0.2983 | 0.0009 | 0.2971 | 0.2991 |
| finreg | uncertainty_mean | 0.0161 | 0.0001 | 0.0159 | 0.0162 |
| finreg | source_consistency | 0.7153 | 0.0000 | 0.7153 | 0.7153 |
| disaster | abstain_rate | 0.0800 | 0.0589 | 0.0200 | 0.1600 |
| disaster | retrieve_more_rate | 0.0800 | 0.0589 | 0.0200 | 0.1600 |
| disaster | contradiction_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| disaster | contradiction_prob_mean | 0.2914 | 0.0013 | 0.2905 | 0.2932 |
| disaster | uncertainty_mean | 0.0169 | 0.0004 | 0.0166 | 0.0174 |
| disaster | source_consistency | 0.7527 | 0.0000 | 0.7527 | 0.7527 |

## Quick Read

- `health`: moderately stable; balanced coverage; low contradiction risk.
- `finreg`: moderately stable; high coverage (low abstain); low contradiction risk.
- `disaster`: moderately stable; high coverage (low abstain); low contradiction risk.
- Cross-domain: no contradiction spike observed across seeds.
