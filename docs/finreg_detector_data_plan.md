# FinReg Detector Data Plan

The first FinReg adaptation attempt did not improve the detector. The likely issue is data:
the initial 66-example train set was too small and too template-like.

## Current Direction

Use a larger corpus-derived candidate pool before any further detector training.

Script:

```bash
PYTHONPATH=. venv312/bin/python scripts/build_finreg_detector_candidate_pool.py
```

Output:

- `data/domain_finreg/detector_candidate_pool_v2.jsonl`
- `data/domain_finreg/detector_candidate_pool_v2_summary.json`
- Current active candidate output after stricter filtering:
  - `data/domain_finreg/detector_candidate_pool_v11.jsonl`
  - `data/domain_finreg/detector_candidate_pool_v11_summary.json`
- Current manual review set:
  - `data/domain_finreg/manual_review/detector_v3/review_set.csv`
  - `data/domain_finreg/manual_review/detector_v3/review_set.jsonl`
- First-pass reviewed output:
  - `data/domain_finreg/manual_review/detector_v3/reviewed_set_codex_v1.csv`
  - `data/domain_finreg/manual_review/detector_v3/reviewed_set_codex_v1.jsonl`
- Reviewed training dataset:
  - `data/training/nli_dataset_finreg_detector_reviewed_v1`

## Policy

- Candidate rows are not gold labels.
- Every row starts with `metadata.review_status = pending`.
- Candidate rows should be sampled and reviewed before training.
- Existing `data/domain_finreg/detector_eval_finreg_v1.jsonl` remains held-out test data.

## Why This Is Better Than The First Attempt

- Premises come directly from the current FinReg corpus.
- Neutral examples are cross-source and cross-theme, closer to RAG grounding failures.
- Contradictions are minimal controlled edits of real regulatory sentences.
- The pool is larger and traceable by source, theme, and transformation.

## Note On Candidate Versions

The first generated pool (`detector_candidate_pool_v1`) was rejected after spot-checking because
it still contained navigation, event-registration, and consultation-response noise.

Versions v2-v10 were intermediate tightening passes. They exposed additional issues such as page
headers, consultation-response fragments, URL/citation lines, double-negation transforms, bullets,
ellipsis fragments, broken PDF extraction artifacts, and document-meta sentences.

The current active candidate pool is v11:

- 240 examples total.
- 120 neutral cross-source/cross-theme pairs.
- 60 extractive entailment pairs.
- 60 minimal contradiction pairs.
- All rows remain `metadata.review_status = pending`.

The v11 pool is suitable for manual review / seed training experiments, but it is not a gold dataset.
Do not use it as held-out evaluation data.

## Manual Review Protocol

Use `data/domain_finreg/manual_review/detector_v3/review_set.csv` for the first review pass.

Allowed `review_label` values:

- `entailment`
- `neutral`
- `contradiction`
- `bad`

Use `keep = yes` only when the pair is clear and useful for training. Use `keep = no` for noisy
PDF extraction, document-meta text, unclear neutral relations, unnatural contradictions, or examples
that are technically true but not useful for the detector.

First-pass review result:

- 90 reviewed examples.
- 78 kept.
- 12 dropped.
- Kept label counts: 29 entailment, 22 neutral, 27 contradiction.
- Balanced training selection uses 22 examples per label.

The first reviewed training dataset is intentionally small:

- selected: 66 examples, 22 per label
- train: 54 examples, 18 per label
- validation: 12 examples, 4 per label
- held-out test: `data/domain_finreg/detector_eval_finreg_v1.jsonl`

This dataset is appropriate for a smoke adaptation run, not for a final detector claim.

## RAG-Aware Pseudo-NLI Dataset

The next detector data attempt switched from isolated sentence pairs to detector-call-shaped
examples:

- premise: retrieved-context-like corpus window
- hypothesis: shorter answer-like claim
- labels: entailment, neutral, contradiction
- held-out test: still `data/domain_finreg/detector_eval_finreg_v1.jsonl`

Script:

```bash
PYTHONPATH=. venv312/bin/python scripts/build_finreg_detector_ragaware_dataset.py
```

Generated datasets:

- `data/training/nli_dataset_finreg_detector_ragaware_v2`
- `data/training/nli_dataset_finreg_detector_ragaware_v3`

Current main training dataset:

- `data/training/nli_dataset_finreg_detector_ragaware_v2`
- train: 360 examples, 120 per label
- validation: 72 examples, 24 per label
- test: copied held-out detector eval, 36 examples

`ragaware_v3` adds stricter filtering and a source-balanced selection attempt, but the generated
sample still shows enough PDF extraction noise that it has not replaced v2 as the main training
input.

## Combined Detector Dataset

A mixed dataset was also created to test whether reviewed examples improve the RAG-aware signal:

- `data/training/nli_dataset_finreg_detector_combo_v1`
- source: RAG-aware v2 train/val plus reviewed v1 train/val
- train: 414 examples, 138 per label
- validation: 84 examples, 28 per label
- test: copied held-out detector eval, 36 examples

This combo dataset did not improve held-out detector performance in the FEVER-base run. Keep it as
an experiment artifact, not the preferred detector training set.

## Data Quality Lesson

The current detector experiments show that quantity alone is not enough:

- RAG-aware pseudo data is more relevant than the first small adaptation set, but still noisy.
- EBA/consultation-derived text and PDF extraction artifacts can dominate the generated examples.
- Minimal contradiction templates are useful but too shallow if overused.
- The most promising model so far used the stronger FEVER DeBERTa-v3-base initialization with the RAG-aware v2 data.

Next detector-data improvement should focus on a cleaner eval-like FinReg NLI set, ideally with
human or stronger-LLM review, before generating more pseudo-labeled examples.
