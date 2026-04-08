# Refined Phase-1 Hard-Core Retrieval Diagnosis

This note captures the first manual diagnosis pass for the refined phase-1
question set on the new real finreg corpus.

Scope:
- `fq06`
- `fq07`
- `fq10`
- `fq14`
- `fq15`
- `fq19`

These were chosen because they remained abstain-prone or seed-sensitive after
the first rewrite pass on
[`questions_finreg_conflict_phase1_refined.jsonl`](../data/domain_finreg/questions_finreg_conflict_phase1_refined.jsonl).

## Summary

The common pattern is not a single detector failure.

The dominant issues are:
- retrieval imbalance on comparison questions
- residual question breadth on a few prompts
- incomplete output hygiene on one prompt

The corpus is no longer the primary bottleneck for all six questions. For some
of them the evidence exists, but the top retrieved set is skewed toward one
source family, which makes the answer unstable across seeds.

## Question-Level Diagnosis

### `fq06`

Question:
"Do BCBS and ECB guidance differ on expected timeliness for risk data
aggregation during periods of stress?"

Observed behavior:
- `seed7`: abstain
- `seed11`: abstain
- `seed19`: answer

Diagnosis:
- Retrieval is dominated by ECB RDARR chunks.
- BCBS evidence is often missing from the top retrieved set.
- This is a two-sided comparison question; when one side is absent, the model
  correctly becomes uncertain or abstains.

Interpretation:
- Primary issue: retrieval imbalance
- Secondary issue: none

Recommended action:
- Add source-family-aware retrieval diversification for comparison questions.
- Ensure at least one BCBS and one ECB candidate can survive into the final
  context set.

### `fq07`

Question:
"Do BCBS 239 and EBA internal governance guidance assign responsibility for
data quality and reporting integrity in the same way?"

Observed behavior:
- `seed7`: abstain
- `seed11`: answer
- `seed19`: answer

Diagnosis:
- Retrieval again overweights ECB/RDARR material even though the question asks
  for BCBS and EBA.
- The answer becomes unstable when EBA-specific evidence is not present.

Interpretation:
- Primary issue: retrieval imbalance
- Secondary issue: wording could be slightly tightened around governance roles

Recommended action:
- Bias comparison retrieval toward the named source families in the query.
- Consider narrowing "in the same way" to a more concrete comparison target such
  as board responsibility, senior management responsibility, or control
  ownership.

### `fq10`

Question:
"How do BCBS and ECB or PRA materials treat acceptable use of manual
workarounds in risk reporting and aggregation processes?"

Observed behavior:
- `seed7`: answer
- `seed11`: abstain
- `seed19`: answer

Diagnosis:
- Relevant evidence exists in the corpus.
- Top contexts are often ECB-heavy with only one BCBS item.
- The question mixes comparison and policy interpretation, so small context
  changes move the model between answer and abstain.

Interpretation:
- Primary issue: borderline retrieval balance
- Secondary issue: prompt still slightly broad

Recommended action:
- Split this into one regulator pair at a time, for example BCBS vs ECB.
- Anchor the expected output to controls around temporary manual adjustments,
  overrides, or remediation conditions.

### `fq14`

Question:
"How do supervisory materials frame remediation expectations when data-quality
deficiencies persist?"

Observed behavior:
- `seed7`: answer
- `seed11`: abstain
- `seed19`: abstain

Diagnosis:
- Retrieval drifts toward ECB RDARR plus unrelated supervisory material.
- The question does not pin the relevant source family tightly enough.
- The answer target is broad: remediation expectations can mean governance,
  escalation, controls, timing, or supervisory follow-up.

Interpretation:
- Primary issue: question breadth
- Secondary issue: retrieval drift

Recommended action:
- Rewrite around a narrower angle, such as escalation, remediation planning, or
  management action when deficiencies persist.
- Limit the comparison to one or two source families.

### `fq15`

Question:
"How do BCBS and ECB or PRA materials frame materiality and escalation for
risk-reporting errors?"

Observed behavior:
- `seed7`: answer
- `seed11`: abstain
- `seed19`: answer

Diagnosis:
- Evidence is partially present but often implicit rather than directly stated.
- The question asks for two linked concepts, materiality and escalation, which
  are not always expressed together in the retrieved passages.
- This makes the model sensitive to small seed changes.

Interpretation:
- Primary issue: weakly explicit evidence in phase-1 corpus
- Secondary issue: question still asks for too much in one shot

Recommended action:
- Split into separate questions for materiality framing and escalation
  expectations.
- Treat this as phase-1 borderline rather than fully unsupported.

### `fq19`

Question:
"How do BCBS, ECB, and EBA materials frame data-governance responsibilities for
cross-border banking groups?"

Observed behavior:
- `seed7`: answer
- `seed11`: abstain
- `seed19`: abstain

Diagnosis:
- The question spans three source families and a relatively specialized
  sub-topic.
- Retrieval is not balanced enough for a three-way comparison.
- One generated answer also showed residual output contamination with unrelated
  employment text after a plausible opening sentence.

Interpretation:
- Primary issue: too broad for phase-1 retrieval
- Secondary issue: output hygiene still needs one more pass

Recommended action:
- Reduce to two-family comparisons first.
- Add stronger post-generation cleanup for non-regulatory text leakage.
- Revisit three-way supervisory comparison only after retrieval balancing.

## Overall Recommendations

Short term:
- keep the refined set as the active working set
- treat `fq06` and `fq07` as retrieval-balance problems
- treat `fq14` and `fq19` as still too broad for phase-1
- treat `fq10` and `fq15` as borderline wording-plus-retrieval problems

Next implementation step:
- add source-family-aware retrieval balancing for comparison questions that
  explicitly name regulator families

After that:
- rerun the refined 20Q seed matrix
- re-check whether the hard-core questions move from abstain to stable answer
  or stable abstain
