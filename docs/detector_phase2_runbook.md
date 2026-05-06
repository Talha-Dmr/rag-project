# Detector Phase 2 Runbook

This runbook turns the `Detector Optimization Phase 2` plan into concrete execution steps.

## 1. Benchmark curation

Primary location:

- `data/benchmarks/finreg_detector_phase2/`

Start from:

- `evaluation_results/finreg_detector_manualeval_v3/manual_annotation_sheet.csv`
- `evaluation_results/finreg_detector_manualeval_v3/stratified_eval_subset.jsonl`
- `data/benchmarks/finreg_detector_phase2/v1/benchmark_v1_draft.jsonl`

Goal:

- fill the draft rows
- preserve provenance
- ensure each benchmark item has a final `gold_label`

Recommended acceleration:

- run `scripts/prefill_finreg_detector_phase2_annotations.py`
- use its `suggested_label` and `review_priority` fields only as reviewer aids
- never promote `suggested_label` directly into `gold_label` without review
- then run `scripts/export_finreg_phase2_priority_review.py` to isolate `p0/p1` rows first
- optionally run `scripts/export_finreg_phase2_reviewer_packet.py` to create a compact CSV for human review
- optionally run `scripts/draft_finreg_phase2_review_notes.py` to generate concise draft notes for reviewers
- after manual edits, run `scripts/sync_finreg_phase2_review_labels.py` to copy final gold fields back into the other review CSVs
- then run `scripts/build_finreg_phase2_gold_seed.py` to create the first reviewed benchmark seed set for detector-only eval
- if you need to grow the seed, run `scripts/mine_finreg_phase2_expansion_candidates.py` on the full per-question outputs to surface more supported/contradicted review candidates
- for a sharper contradiction-focused pass, run `scripts/mine_finreg_phase2_hard_risk_watchlist.py` to create a high-risk review watchlist

## 2. Benchmark build path

Current reusable builder:

- `scripts/build_finreg_eval_v2.py`
- `scripts/build_finreg_detector_phase2_benchmark.py`

Near-term workflow:

1. derive candidate rows from manual annotations
   or run:
   `python scripts\build_finreg_detector_phase2_benchmark.py --annotations evaluation_results\finreg_detector_manualeval_v3\manual_annotation_sheet.csv`
2. map rows into the Phase 2 benchmark schema
3. author missing contradiction-heavy and ambiguous rows manually
4. save the curated output as the next benchmark v1 file

## 3. Detector-only evaluation

Compare:

- `balanced`
- `targeted_v2`

Required outputs:

- per-record detector scores
- confusion by gold label
- contradiction recall and precision
- supported precision
- calibration report

## 4. Calibration and aggregation sweep

Evaluate these score views on the same benchmark:

- `contradiction_prob_mean`
- `hallucination_prob_topk`
- `detector_conflict`
- `detector_conflict_consensus`
- entailment vs contradiction margin
- neutral vs contradiction margin

Reusable script:

- `scripts/compare_finreg_detector_phase2_scores.py`

Keep a simple decision table:

- supported precision
- contradiction recall
- ECE
- Brier

## 5. Training slice authoring

Primary location:

- `data/training/nli_dataset_finreg_phase2_targeted/`

Author first:

- cross-document contradictions
- wrong-number or wrong-threshold cases
- partial support cases
- cross-chunk mismatch cases
- multiple-QA mixed-support cases

## 6. Retraining gate

Do not retrain yet if:

- benchmark rows are still mostly placeholders
- contradiction-heavy slices are under-filled
- supported control rows are missing
- calibration comparison is not ready

## 7. Return to chunking

Only return to chunking after:

- detector-only benchmark gains are clear
- calibration is stable
- targeted candidate no longer depends on a special-case reading of a few examples
