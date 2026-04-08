# FinReg Question Set Methodology

## Purpose

This note explains what the current finreg question sets are, how they appear to have been created, and what they are actually measuring.

It is not a claim that the current question sets are benchmark-quality. It is a clarification of scope and intent.

## Current Question Assets

Main files:

- `data/domain_finreg/questions_finreg_conflict.jsonl`
- `data/domain_finreg/questions_finreg_conflict_50.jsonl`

The `20Q` file is the seed set.

The `50Q` file is an expanded set.

## What We Know About Their Origin

### 20Q seed set

The `20Q` finreg set:

- appears to be **project-authored**
- was introduced directly in the repository
- does not have a dedicated generator script

Relevant commit:

- `3fa845edc8fce2fa1c11cd60e508ff531dda417b`
- message: `Promote balanced detector defaults and add high-stakes eval scaffold`

The strongest current inference is:

- the `20Q` finreg set was **manually curated**
- as a small internal stress-test seed set

### 50Q expanded set

The `50Q` set is generated from the seed set via:

- `scripts/build_high_stakes_questions_50.py`

This script:

- preserves the existing `20Q` seed questions
- adds new sanity and conflict questions
- uses:
  - source lists
  - topic lists
  - question templates

For finreg, the expansion script uses source names like:

- `BCBS`
- `EBA`
- `ECB`
- `Federal Reserve`
- `PRA`

And topics such as:

- risk aggregation timeliness
- board accountability for data quality
- manual adjustments in regulatory reporting
- model validation frequency
- climate risk in ICAAP
- outsourcing controls
- materiality thresholds for reporting errors
- audit trail retention
- intraday liquidity monitoring
- AI model explainability requirements

So the `50Q` set should be understood as:

- **template-expanded**
- not independently hand-curated at the same depth as the seed

## What The Current Sets Are Measuring

The current finreg sets are **not** generic finance QA.

They are primarily measuring:

- multi-source comparison
- prudential / supervisory conflict
- ambiguity under partial disagreement
- whether the system should answer confidently or behave cautiously

In practice, this means the sets are best described as:

- **high-stakes conflict-oriented finreg prompts**

Not:

- standard public benchmark QA
- pure factual regulatory QA
- natural user-query distribution

## Question Types Present In The 20Q Seed

The current seed set contains two visible classes:

### 1. Sanity anchor questions

Examples:

- `fq01` core objective of BCBS 239
- `fq02` supervisory review
- `fq03` data lineage
- `fq04` stress testing
- `fq05` model risk management

These are:

- concept anchors
- domain grounding checks
- useful for confirming that the corpus and retrieval stack have basic coverage

### 2. Conflict / ambiguity questions

Examples:

- near-real-time aggregation expectations
- board accountability differences
- proportionality across jurisdictions
- manual workarounds
- model validation frequency and depth
- climate-risk integration into ICAAP
- outsourcing controls
- audit trail retention
- intraday liquidity monitoring
- AI explainability requirements

These are:

- comparison-oriented
- often multi-regulator
- designed to stress retrieval, grounding, and gating

## What They Are Not

The current finreg sets are **not**:

- a labeled ambiguous QA benchmark in the academic sense
- a representative sample of real production user questions
- a comprehensive prudential regulation exam

They are closer to:

- an internal evaluation slice
- a stress-test harness for uncertainty-aware RAG

## Strengths

The current sets are useful because they intentionally target:

- high decision cost
- regulator disagreement
- partial evidence support
- questions where abstain / retrieve-more can be more appropriate than direct answering

That makes them more suitable for gating research than a purely factual QA set.

## Weaknesses

The current sets have real limitations:

- provenance is weakly documented
- `20Q` was likely hand-curated but without a written selection protocol
- `50Q` is template-expanded, so it may over-repeat the same question family
- some questions are broader than the current phase-1 corpus can support
- some questions are closer to synthetic comparison prompts than natural user queries

## Best Current Interpretation

The right way to read the current finreg question sets is:

- `20Q`: manually curated internal seed set
- `50Q`: template-expanded extension of that seed

And the right way to read their role is:

- **conflict / ambiguity evaluation slice**
- not full-domain finreg evaluation

## Recommended Next Documentation Step

The next improvement should be explicit question tagging for each `20Q` item:

- `sanity_anchor`
- `comparison_conflict`
- `broad_undercovered`
- `phase1_supported`
- `phase1_unsupported`
- `natural_user_like`
- `synthetic_comparison_like`

That would make the dataset much easier to interpret during future corpus, retrieval, and gating changes.
