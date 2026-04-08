# FinReg Question Coverage Matrix

## Purpose

This note evaluates the current finreg question sets against the planned phase-1 real corpus scope.

It does not assume the real corpus already exists.

Instead, it answers:

- which current questions should survive into the real-corpus phase
- which questions need narrower wording
- which questions should be deferred until later source expansion

## Corpus Assumption

This coverage pass assumes the phase-1 corpus from:

- `docs/finreg_corpus_phase1_scope.md`

Phase-1 source families:

- BCBS / BIS
- EBA
- PRA / BoE
- ECB

Conditional later source family:

- Federal Reserve / OCC

## Coverage Labels

- `keep`: should remain in the canonical seed set
- `keep_narrower`: keep the topic, but rewrite the wording to match likely corpus evidence
- `defer`: useful topic, but should wait for broader source coverage

## 20Q Seed Set Review

| ID | Type | Theme | Likely source need | Phase-1 status | Notes |
| --- | --- | --- | --- | --- | --- |
| fq01 | sanity | BCBS 239 objective | BCBS | keep | Strong anchor question for the corpus. |
| fq02 | sanity | supervisory review / governance | EBA or BCBS | keep_narrower | Should be tied to a named source family rather than generic "supervisory review". |
| fq03 | sanity | data lineage / reporting traceability | BCBS | keep | Strong BCBS-style evidence topic. |
| fq04 | sanity | stress testing role | BCBS / ECB / PRA | keep_narrower | Good topic, but wording should point to prudential purpose in a specific supervisory frame. |
| fq05 | sanity | model risk management | EBA / PRA / ECB, maybe OCC/FED later | keep_narrower | Too generic today; keep only if rewritten to the selected source family. |
| fq06 | conflict | near-real-time risk aggregation | BCBS vs EBA/ECB/PRA | keep | Core conflict theme. |
| fq07 | conflict | board accountability for data quality | BCBS vs local supervisors | keep | Core governance conflict theme. |
| fq08 | conflict | compliance timelines for risk-data principles | BCBS vs local implementation | keep_narrower | Needs real timeline evidence; wording should mention specific guidance sets. |
| fq09 | conflict | proportionality for smaller banks | BCBS vs EBA/PRA | keep | Good phase-1 conflict topic. |
| fq10 | conflict | manual workarounds in reporting | BCBS vs ECB/PRA | keep | Good operational conflict topic. |
| fq11 | conflict | independent validation frequency / depth | EBA/PRA/ECB, maybe OCC/FED later | keep_narrower | Keep topic, but narrow to sources actually in phase 1. |
| fq12 | conflict | climate risk in ICAAP | BCBS vs ECB/EBA/PRA | keep | Strong phase-1 comparator theme. |
| fq13 | conflict | outsourcing controls for risk systems | EBA/PRA/BCBS | keep | Good if supported by real ICT / outsourcing guidance. |
| fq14 | conflict | remediation thresholds / penalties for data-quality failures | EBA/PRA/ECB | keep_narrower | Topic is relevant, but wording should avoid implying explicit penalty schedules unless evidence exists. |
| fq15 | conflict | material risk-reporting errors | BCBS-aligned vs local guidance | keep_narrower | Good topic, but likely needs narrower "materiality / escalation" wording. |
| fq16 | conflict | AI explainability for credit risk models | mostly later US + broader AI governance sources | defer | Valuable topic, but probably under-supported in phase-1 corpus. |
| fq17 | conflict | audit trail retention | BCBS / EBA / PRA reporting controls | keep_narrower | Keep topic; rewrite toward traceability / retention control expectations if exact retention periods are weak. |
| fq18 | conflict | intraday liquidity monitoring | BCBS vs local supervisors | keep | Strong phase-1 conflict topic. |
| fq19 | conflict | cross-border group data governance | BCBS / ECB / PRA / EBA | keep_narrower | Keep theme; may need narrower wording around group governance expectations. |
| fq20 | conflict | compliance finding -> capital planning impact | broader supervisory judgment / remediation frameworks | defer | Important theme, but wording is too indirect unless specific documents support it. |

## Recommended Canonical 20Q Outcome

### Keep directly

- `fq01`
- `fq03`
- `fq06`
- `fq07`
- `fq09`
- `fq10`
- `fq12`
- `fq13`
- `fq18`

### Keep, but rewrite more narrowly

- `fq02`
- `fq04`
- `fq05`
- `fq08`
- `fq11`
- `fq14`
- `fq15`
- `fq17`
- `fq19`

### Defer until later source expansion

- `fq16`
- `fq20`

## Implication For The 50Q Set

The current 50Q set is not independently curated.

It is a template expansion of the 20Q seed themes using:

- `scripts/build_high_stakes_questions_50.py`

That means the 50Q set should not be treated as canon until the 20Q seed set is cleaned up first.

### Rule for 50Q regeneration

1. freeze the corrected canonical 20Q seed set
2. remove deferred themes from the generator inputs
3. regenerate 50Q from the cleaned theme list

## Recommended Next Step

Before detector or gating recalibration on real finreg evidence:

1. rewrite the `keep_narrower` questions
2. drop or park the `defer` questions
3. regenerate the 50Q set from the corrected 20Q base
