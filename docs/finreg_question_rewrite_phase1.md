# FinReg Phase-1 Question Rewrite

## Purpose

This file records the first refined version of the finreg seed set after:

- the move from synthetic bootstrap corpus to real phase-1 corpus
- seed-sensitivity diagnosis
- question-coverage review

The refined set is stored in:

- `data/domain_finreg/questions_finreg_conflict_phase1_refined.jsonl`

The original seed set remains unchanged in:

- `data/domain_finreg/questions_finreg_conflict.jsonl`

## Why A Separate File

The original seed set should remain available for:

- historical comparison
- old experiment traceability
- understanding how the project evolved

The refined set is a phase-1 candidate, not a retroactive overwrite.

## Rewrite Principles

The refined set follows these rules:

1. prefer named source families over generic "frameworks"
2. avoid wording that implies evidence the corpus does not clearly contain
3. reduce synthetic comparison phrasing when a more natural prudential wording is possible
4. keep the conflict / ambiguity focus
5. stay inside the current phase-1 corpus whenever possible

## Main Changes

### Narrowed anchor questions

- `fq02`
  - old: generic supervisory review
  - new: explicitly tied to EBA SREP

- `fq03`
  - old: generic data lineage question
  - new: tied to BCBS 239 and implementation guidance

- `fq05`
  - old: generic model risk management
  - new: tied to PRA / ECB internal model materials

### Narrowed conflict questions

- `fq06`
  - old: BIS vs regional supervisors, generic wording
  - new: BCBS vs ECB, specifically timeliness during stress

- `fq08`
  - old: broad compliance timelines across jurisdictions
  - new: BCBS 239 vs ECB RDARR implementation urgency / remediation timing

- `fq10`
  - old: supervisors differ on manual workarounds
  - new: focuses on manual workarounds in risk reporting controls

- `fq11`
  - old: independent validation across frameworks
  - new: narrowed to ECB and PRA materials

- `fq12`
  - old: consistency across major regulators for climate-risk ICAAP integration
  - new: BCBS climate principles vs PRA climate supervision, focused on pace of integration

- `fq14`
  - old: penalties or remediation thresholds
  - new: remediation expectations only

- `fq15`
  - old: harmonized definitions of material errors
  - new: materiality and escalation framing

- `fq17`
  - old: retention requirements
  - new: auditability and traceability requirements

- `fq18`
  - old: divergence on intraday liquidity monitoring expectations
  - new: BCBS monitoring tools vs PRA liquidity supervision, focused on what firms monitor intraday

- `fq19`
  - old: consistency across supervisory bodies for cross-border groups
  - new: responsibilities for cross-border banking group data governance

- `fq20`
  - old: compliance finding -> immediate capital planning impact
  - new: escalation in supervisory review

### Replaced defer-theme question

- `fq16`
  - old: AI-assisted credit risk explainability
  - old status: defer under phase-1 corpus
  - new: documentation and governance expectations for internal model changes

This replacement was made because the AI explainability theme was already flagged as out of scope for the current phase-1 corpus.

## Current Interpretation

The refined file should be treated as:

- the first **phase-1-aligned candidate**
- still internal

## Second Rewrite Pass

After moving the finreg default to the section-aware stack and adding targeted EBA PDF sources,
two questions still remained broader than the available phase-1 evidence:

- `fq14`
- `fq19`

A second pass therefore produced:

- [questions_finreg_conflict_phase1_refined_v2.jsonl](/home/talha/projects/rag-project/data/domain_finreg/questions_finreg_conflict_phase1_refined_v2.jsonl)

Changes:

- `fq14`
  - v1: broad supervisory framing for persistent data-quality deficiencies
  - v2: narrowed to `ECB RDARR` and `EBA SREP` remediation follow-up

- `fq19`
  - v1: broad cross-border data-governance framing
  - v2: narrowed to `BCBS 239 implementation guidance`, `ECB RDARR`, and `EBA internal governance`
    on group-wide governance across subsidiaries and consolidated groups

Observed result on the current canonical finreg stack:

- v1 mean abstain rate: `0.2833`
- v2 mean abstain rate: `0.2667`

Interpretation:

- the second rewrite pass slightly improves coverage
- it mostly reduces avoidable breadth rather than changing the task family
- `v2` should be treated as the current working refined set
- still subject to retrieval and gating validation

It is not yet the final canonical seed set.

## Recommended Next Step

Run the same `20Q x seeds 7,11,19` process on:

- `data/domain_finreg/questions_finreg_conflict_phase1_refined.jsonl`

Then compare:

- abstain stability
- stable-answer count
- retrieval drift reduction
- whether anchor questions still fail unexpectedly
