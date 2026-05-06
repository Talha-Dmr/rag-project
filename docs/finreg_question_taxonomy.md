# FinReg Question Taxonomy

This note defines the working taxonomy for the active FinReg question set.

It is not a public benchmark taxonomy. It is an internal evaluation schema for the
current prudential / supervisory FinReg baseline.

## Scope

Applies to:

- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2.jsonl`
- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl`
- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_labels.jsonl`
- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50_labels.jsonl`

## Why This Exists

The current question sets are useful, but they are not yet benchmark-grade. The main maturity
gap was that questions had `id`, `type`, and `query`, but no explicit task taxonomy.

This taxonomy fixes that by making the intended task visible:

- what kind of question it is
- what ambiguity type it targets
- whether it is expected to be well-supported by the current corpus
- whether it is a natural user-style question or a synthetic comparison stress-test

## Label Dimensions

Each question should be interpretable across the following dimensions.

### 1. `task_family`

- `sanity_anchor`
  - direct grounding / anchor question
- `comparison_conflict`
  - multi-source comparison with possible divergence
- `supervisory_escalation`
  - asks how deficiencies or weaknesses connect to escalation / remediation

### 2. `ambiguity_family`

- `none_or_low`
  - mostly direct grounding
- `cross_regulator_divergence`
  - asks whether regulators differ in tone, speed, threshold, or expectation
- `proportionality_or_scope`
  - asks how expectations vary by firm size, complexity, or scope
- `escalation_or_materiality`
  - asks how problems become material, severe, or supervisory concerns

### 3. `support_level`

- `phase15_supported`
  - expected to be answerable on the current phase-1.5 corpus
- `phase15_hard`
  - answerable in principle, but still brittle or retrieval-sensitive
- `phase2_needed`
  - should not be treated as fully supported without more evidence

### 4. `question_style`

- `natural_user_like`
  - plausible user / analyst formulation
- `synthetic_comparison_like`
  - stress-test wording, comparison-heavy or deliberately evaluation-oriented

### 5. `theme`

Current theme buckets:

- `rdarr`
- `governance`
- `stress_testing`
- `model_risk`
- `climate`
- `outsourcing`
- `remediation`
- `materiality`
- `auditability`
- `liquidity`
- `group_governance`

## Working Interpretation

The active 20Q set should be read as:

- a hand-curated internal evaluation slice
- mostly for `sanity_anchor` and `comparison_conflict`
- focused on prudential / supervisory interpretation rather than broad financial regulation

The active 50Q set should be read as:

- a confirmation set
- partly inherited from the 20Q seed
- partly template-expanded
- useful for stability and regression checks, not for strong benchmark claims

In practice:

- `fq01-fq25` are closer to the seed behavior and should be read as the more meaningful half
- `fq26-fq50` are mostly synthetic comparison stress-tests and should be interpreted accordingly

## Operational Rule

When reading results:

- `sanity_anchor + phase15_supported` should usually answer unless retrieval fails
- `comparison_conflict + phase15_hard` may legitimately `retrieve_more`
- `phase2_needed` questions should not be used to judge detector/gate quality harshly

This taxonomy is meant to reduce interpretation errors before detector / gate ablations.
