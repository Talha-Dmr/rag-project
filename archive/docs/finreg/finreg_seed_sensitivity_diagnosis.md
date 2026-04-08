# FinReg Seed Sensitivity Diagnosis

## Purpose

This note diagnoses the current `20Q` finreg seed set after:

- phase-1 real corpus build
- phase-1 re-index
- abstain normalization for explicit "I don't know based on the provided context" answers

It is the first practical step before:

- question rewrites
- retrieval fixes
- gate recalibration

## Current Observation

On the current phase-1 real corpus, the post-fix runs produced:

- `seed7`: 13 abstain / 7 answer
- `seed11`: 13 abstain / 7 answer
- `seed19`: 9 abstain / 11 answer

Relevant outputs:

- `evaluation_results/auto_eval/finreg_phase1_real_postfix_logit_mi_20_seed7.json`
- `evaluation_results/auto_eval/finreg_phase1_real_postfix_logit_mi_20_seed11.json`
- `evaluation_results/auto_eval/finreg_phase1_real_postfix_logit_mi_20_seed19.json`

## Stability Buckets

### Stable answer

These answered in all three seeds:

- `fq04`
- `fq05`
- `fq13`

Interpretation:

- these are the least problematic questions in the current stack
- they are still not automatically "perfect"
- but they do not currently look like the first bottleneck

### Stable abstain

These abstained in all three seeds:

- `fq11`
- `fq14`
- `fq15`
- `fq18`
- `fq20`

Interpretation:

- these are the current hard blockers
- some are probably genuinely under-supported
- some may be recoverable with rewrites or corpus expansion

### Seed-sensitive

These changed behavior across seeds:

- `fq01`
- `fq02`
- `fq03`
- `fq06`
- `fq07`
- `fq08`
- `fq09`
- `fq10`
- `fq12`
- `fq16`
- `fq17`
- `fq19`

Interpretation:

- these are the right first targets for diagnosis
- they likely mix:
  - question wording issues
  - retrieval drift
  - corpus coverage gaps
  - generation / abstain instability

## Per-Question Diagnosis

| ID | Current behavior | Likely issue type | Working diagnosis | Recommended action |
| --- | --- | --- | --- | --- |
| `fq01` | seed-sensitive | generation / retrieval focus | Should be an anchor question, but occasionally drifts or produces an abstaining answer. | Keep, but inspect retrieval priority and shorten expected answer shape. |
| `fq02` | seed-sensitive | wording breadth | "Supervisory review" is generic; source family not explicit enough. | Rewrite narrower around EBA/BCBS supervisory review. |
| `fq03` | seed-sensitive | generation instability | Good anchor topic, but unstable answer behavior suggests prompt or evidence-selection drift. | Keep, but inspect top chunks and reduce answer scope. |
| `fq04` | stable answer | low current risk | Works as a sanity anchor. | Keep. |
| `fq05` | stable answer | moderate wording breadth | Stable enough now, but still broad in concept. | Keep, later rewrite to named source family if needed. |
| `fq06` | seed-sensitive | conflict + policy | Real ambiguity topic; now abstains in many runs after normalization. | Keep, but review whether abstain is correct or retrieval needs richer local comparator evidence. |
| `fq07` | seed-sensitive | conflict wording | Good governance conflict theme, but sensitivity suggests evidence framing still unstable. | Keep, inspect BCBS vs local source balance. |
| `fq08` | seed-sensitive | wording + corpus specificity | "Timelines" is a strong claim and may exceed available direct evidence. | Rewrite narrower to named implementation guidance or supervisory expectation timing. |
| `fq09` | seed-sensitive | conflict realism | Good proportionality topic, but still unstable under current corpus. | Keep, inspect whether actual source snippets mention proportionality clearly enough. |
| `fq10` | seed-sensitive | operational evidence gap | Good topic, but may need sharper wording around controlled manual workarounds. | Keep, rewrite narrower if evidence remains weak. |
| `fq11` | stable abstain | under-supported / too broad | Validation frequency and depth likely needs more specific source framing, maybe also later US materials. | Rewrite narrower now; keep as phase-1 hard case. |
| `fq12` | seed-sensitive | retrieval drift | Good phase-1 topic, but climate questions have shown drift toward RDARR material. | Keep, but fix retrieval/source selection before changing corpus. |
| `fq13` | stable answer | low current risk | Outsourcing controls are currently one of the healthier themes. | Keep. |
| `fq14` | stable abstain | wording overspecification | "Penalties or remediation thresholds" likely implies stronger explicit evidence than current corpus provides. | Rewrite narrower; avoid penalty schedule implication. |
| `fq15` | stable abstain | wording overspecification | Material error definitions may not be directly harmonized in current sources. | Rewrite toward escalation/materiality framing. |
| `fq16` | seed-sensitive | phase-1 under-coverage | This was already flagged as a defer topic; current behavior confirms that. | Defer to later corpus expansion. |
| `fq17` | seed-sensitive | wording / evidence mismatch | Retention "requirements" may be too literal; traceability/control wording may fit corpus better. | Rewrite narrower around auditability and traceability expectations. |
| `fq18` | stable abstain | important but under-supported | Strong target theme, but current source mix may not yet give enough direct comparator evidence. | Keep as priority corpus-expansion target. |
| `fq19` | seed-sensitive | broad cross-border framing | Theme is valid, but wording is broad and can outrun current evidence. | Rewrite narrower around group governance expectations. |
| `fq20` | stable abstain | out of current scope | Already flagged as defer; current results confirm phase-1 does not support it well. | Defer. |

## First Practical Conclusion

The current next step should **not** be a full dataset rewrite.

The current next step should be:

1. keep the stable anchors
2. keep the core conflict themes
3. rewrite the over-broad questions
4. defer the out-of-scope themes

## Immediate Working Sets

### Keep as current anchors

- `fq04`
- `fq05`
- `fq13`

### Rewrite first

- `fq02`
- `fq03`
- `fq06`
- `fq07`
- `fq08`
- `fq09`
- `fq10`
- `fq11`
- `fq12`
- `fq14`
- `fq15`
- `fq17`
- `fq19`

### Defer

- `fq16`
- `fq20`

### Watch closely before changing

- `fq01`

Reason:

- it should be one of the easiest anchor questions
- if it stays unstable, the issue is likely retrieval/generation behavior rather than question choice

## Operational Implication

This diagnosis suggests the correct sequence is:

1. rewrite the most over-broad questions
2. inspect retrieval drift for climate / governance / BCBS anchor themes
3. rerun `20Q`
4. only then consider broader corpus expansion
