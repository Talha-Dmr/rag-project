# Epistemic Decision Matrix

Canonical stage gates for the shadow epistemic program.

## Stages

1. `Stage 0`: feasibility on `20Q / seed=7`
2. `Stage 1`: fast pass (`answered_contradiction_rate <= baseline`, abstain within band, contradiction guard)
3. `Stage 2`: cost gate (runtime within allowed ratio vs baseline)
4. `Stage 3`: confirmation on `50Q / seed=7`
5. `Stage 4`: robustness on `50Q / seed=11,19`

## Candidate Status

| Candidate | S0 | S1 | S2 | S3 | S4 | Final |
| --- | --- | --- | --- | --- | --- | --- |
| `logit_mi` | fail | fail | fail | pending | pending | freeze |
| `stochastic_langevin` | fail | fail | fail | pending | pending | freeze |

