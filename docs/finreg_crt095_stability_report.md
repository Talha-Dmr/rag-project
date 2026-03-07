# Seed Stability Report (High-Stakes 3-Domain)

- Question set size(s): 50
- Generated: 2026-02-12T21:46:17
- Runs found: 3

## Per-Run Results

| Domain | Seed | Set | Tag | Abstain | Retrieve-More | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| finreg | 7 | 50 | crt095 | 0.960 | 0.960 | 0.992 | 0.398 | 0.0201 | 0.715 |
| finreg | 11 | 50 | crt095 | 0.980 | 0.980 | 0.988 | 0.399 | 0.0191 | 0.715 |
| finreg | 19 | 50 | crt095 | 0.980 | 0.980 | 0.992 | 0.397 | 0.0196 | 0.715 |

## Domain Summary (mean ± std, with min/max)

| Domain | Metric | Mean | Std | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: |
| finreg | abstain_rate | 0.9733 | 0.0094 | 0.9600 | 0.9800 |
| finreg | retrieve_more_rate | 0.9733 | 0.0094 | 0.9600 | 0.9800 |
| finreg | contradiction_rate | 0.9907 | 0.0019 | 0.9880 | 0.9920 |
| finreg | contradiction_prob_mean | 0.3976 | 0.0008 | 0.3967 | 0.3986 |
| finreg | uncertainty_mean | 0.0196 | 0.0004 | 0.0191 | 0.0201 |
| finreg | source_consistency | 0.7153 | 0.0000 | 0.7153 | 0.7153 |

## Quick Read

- `finreg`: seed-stable; conservative coverage (high abstain); elevated contradiction risk.
- Cross-domain: contradiction spikes exist; keep calibration active.
