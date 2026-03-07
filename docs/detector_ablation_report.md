# Detector Ablation Report (Balanced vs Focal)

- Generated: 2026-02-13T00:12:14
- Runs found: 18
- Domains: `health`, `finreg`, `disaster`
- Variants: `balanced`, `focal`

## Per-Run Results

| Domain | Variant | Seed | Abstain | Retrieve-More | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| disaster | balanced | 7 | 0.020 | 0.020 | 0.000 | 0.290 | 0.0166 | 0.753 |
| disaster | balanced | 11 | 0.160 | 0.160 | 0.000 | 0.293 | 0.0174 | 0.753 |
| disaster | balanced | 19 | 0.060 | 0.060 | 0.000 | 0.291 | 0.0167 | 0.753 |
| disaster | focal | 7 | 0.060 | 0.060 | 0.000 | 0.305 | 0.0162 | 0.753 |
| disaster | focal | 11 | 0.020 | 0.020 | 0.000 | 0.304 | 0.0161 | 0.753 |
| disaster | focal | 19 | 0.100 | 0.100 | 0.000 | 0.304 | 0.0166 | 0.753 |
| finreg | balanced | 7 | 0.040 | 0.040 | 0.000 | 0.215 | 0.0160 | 0.715 |
| finreg | balanced | 11 | 0.020 | 0.020 | 0.000 | 0.216 | 0.0157 | 0.715 |
| finreg | balanced | 19 | 0.060 | 0.060 | 0.000 | 0.215 | 0.0164 | 0.715 |
| finreg | focal | 7 | 0.020 | 0.020 | 0.396 | 0.345 | 0.0160 | 0.715 |
| finreg | focal | 11 | 0.040 | 0.040 | 0.480 | 0.348 | 0.0160 | 0.715 |
| finreg | focal | 19 | 0.040 | 0.040 | 0.484 | 0.347 | 0.0166 | 0.715 |
| health | balanced | 7 | 0.060 | 0.060 | 0.000 | 0.255 | 0.0160 | 0.728 |
| health | balanced | 11 | 0.080 | 0.080 | 0.000 | 0.256 | 0.0164 | 0.728 |
| health | balanced | 19 | 0.020 | 0.020 | 0.000 | 0.255 | 0.0157 | 0.728 |
| health | focal | 7 | 1.000 | 1.000 | 1.000 | 0.435 | 0.0177 | 0.728 |
| health | focal | 11 | 1.000 | 1.000 | 1.000 | 0.435 | 0.0193 | 0.728 |
| health | focal | 19 | 1.000 | 1.000 | 1.000 | 0.437 | 0.0197 | 0.728 |

## Domain Summary (mean ± std)

| Domain | Variant | Abstain | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |
| --- | --- | --- | --- | --- | --- | --- |
| health | balanced | 0.053 ± 0.025 | 0.000 ± 0.000 | 0.255 ± 0.000 | 0.0160 ± 0.0003 | 0.728 ± 0.000 |
| health | focal | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.435 ± 0.001 | 0.0189 ± 0.0008 | 0.728 ± 0.000 |
| finreg | balanced | 0.040 ± 0.016 | 0.000 ± 0.000 | 0.216 ± 0.001 | 0.0160 ± 0.0003 | 0.715 ± 0.000 |
| finreg | focal | 0.033 ± 0.009 | 0.453 ± 0.041 | 0.347 ± 0.001 | 0.0162 ± 0.0003 | 0.715 ± 0.000 |
| disaster | balanced | 0.080 ± 0.059 | 0.000 ± 0.000 | 0.291 ± 0.001 | 0.0169 ± 0.0004 | 0.753 ± 0.000 |
| disaster | focal | 0.060 ± 0.033 | 0.000 ± 0.000 | 0.304 ± 0.000 | 0.0163 ± 0.0002 | 0.753 ± 0.000 |

## Overall Summary (all domains, all seeds)

| Variant | Abstain | Contradiction | Contradiction Prob | Uncertainty | Source Consistency |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced | 0.058 | 0.000 | 0.254 | 0.0163 | 0.732 |
| focal | 0.364 | 0.484 | 0.362 | 0.0171 | 0.732 |

## Paired Delta (focal - balanced)

| Metric | Mean Delta | Std Delta | Pairs |
| --- | ---: | ---: | ---: |
| abstain_rate | 0.3067 | 0.4556 | 9 |
| contradiction_rate | 0.4844 | 0.4095 | 9 |
| contradiction_prob_mean | 0.1080 | 0.0701 | 9 |
| uncertainty_mean | 0.0008 | 0.0016 | 9 |
| source_consistency | 0.0000 | 0.0000 | 9 |

## Quick Read

- Overall, `focal` increases abstain (+0.307) and increases contradiction (+0.484) vs `balanced`.
- Contradiction probability mean shifts by +0.108 (focal - balanced).
- Risk concentration: focal contradiction rises sharply in `health`, `finreg`.
- Default recommendation for risk control: `balanced`.
