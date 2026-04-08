# Seed Stability Report (High-Stakes 3-Domain)

- Question set size(s): 20
- Generated: 2026-02-13T12:19:31
- Runs found: 9

## Per-Run Results

| Domain | Seed | Set | Tag | Abstain | Retrieve-More | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| disaster | 7 | 20 | - | 0.050 | 0.050 | 0.080 | 0.321 | 0.0164 | 0.760 |
| disaster | 11 | 20 | - | 0.050 | 0.050 | 0.000 | 0.271 | 0.0163 | 0.760 |
| disaster | 19 | 20 | - | 0.000 | 0.000 | 0.020 | 0.324 | 0.0159 | 0.760 |
| finreg | 7 | 20 | - | 0.000 | 0.000 | 0.040 | 0.314 | 0.0157 | 0.712 |
| finreg | 11 | 20 | - | 0.000 | 0.000 | 0.010 | 0.308 | 0.0163 | 0.712 |
| finreg | 19 | 20 | - | 0.000 | 0.000 | 0.980 | 0.411 | 0.0161 | 0.712 |
| health | 7 | 20 | - | 0.300 | 0.300 | 0.670 | 0.344 | 0.0181 | 0.725 |
| health | 11 | 20 | - | 0.150 | 0.150 | 0.000 | 0.311 | 0.0177 | 0.725 |
| health | 19 | 20 | - | 0.050 | 0.050 | 0.000 | 0.246 | 0.0162 | 0.725 |

## Domain Summary (mean ± std, with min/max)

| Domain | Metric | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| health | abstain_rate | 0.1667 | 0.1027 | 0.0500 | 0.3000 |
| health | retrieve_more_rate | 0.1667 | 0.1027 | 0.0500 | 0.3000 |
| health | contradiction_rate | 0.2233 | 0.3158 | 0.0000 | 0.6700 |
| health | contradiction_prob_mean | 0.3004 | 0.0410 | 0.2457 | 0.3444 |
| health | uncertainty_mean | 0.0173 | 0.0008 | 0.0162 | 0.0181 |
| health | source_consistency | 0.7252 | 0.0000 | 0.7252 | 0.7252 |
| finreg | abstain_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| finreg | retrieve_more_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| finreg | contradiction_rate | 0.3433 | 0.4504 | 0.0100 | 0.9800 |
| finreg | contradiction_prob_mean | 0.3444 | 0.0472 | 0.3080 | 0.4110 |
| finreg | uncertainty_mean | 0.0160 | 0.0002 | 0.0157 | 0.0163 |
| finreg | source_consistency | 0.7116 | 0.0000 | 0.7116 | 0.7116 |
| disaster | abstain_rate | 0.0333 | 0.0236 | 0.0000 | 0.0500 |
| disaster | retrieve_more_rate | 0.0333 | 0.0236 | 0.0000 | 0.0500 |
| disaster | contradiction_rate | 0.0333 | 0.0340 | 0.0000 | 0.0800 |
| disaster | contradiction_prob_mean | 0.3053 | 0.0242 | 0.2711 | 0.3240 |
| disaster | uncertainty_mean | 0.0162 | 0.0002 | 0.0159 | 0.0164 |
| disaster | source_consistency | 0.7598 | 0.0000 | 0.7598 | 0.7598 |

## Quick Read

- `health`: seed-sensitive; balanced coverage; elevated contradiction risk.
- `finreg`: seed-sensitive; high coverage (low abstain); elevated contradiction risk.
- `disaster`: moderately stable; high coverage (low abstain); low contradiction risk.
- Cross-domain: contradiction spikes exist; keep calibration active.
