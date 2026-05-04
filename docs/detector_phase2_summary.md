# Detector Optimization Phase 2 Summary

This note summarizes what was done in `Detector Optimization Phase 2`, why the work
was structured this way, how the benchmark/review pipeline was built, and what the
current findings mean for the next detector iteration.

## 1. Why Phase 2 was opened

The previous detector work had already produced multiple candidate detectors and a
working gating / inference path. The open problem was no longer "how do we plug a
detector into the stack?" The open problem was "does the detector produce a useful,
finreg-grounded contradiction signal that can be trusted independently from chunking?"

For that reason, Phase 2 deliberately separated detector evaluation from chunking:

- `balanced` was locked as the stable production baseline
- `targeted_v2` was kept as the main optimization candidate
- chunking decisions were explicitly removed from the critical path
- the detector was treated as its own product with its own benchmark and promotion gate

## 2. What we built

### 2.1 Benchmark and dataset scaffolding

We created a Phase 2 benchmark layout and schemas for detector-only evaluation:

- `data/benchmarks/finreg_detector_phase2/...`
- `data/training/nli_dataset_finreg_phase2_targeted/...`
- `scripts/build_finreg_detector_phase2_benchmark.py`

The benchmark schema was designed around the labels that matter for grounding rather
than generic NLI only:

- `supported`
- `unsupported`
- `contradicted`
- `partial`
- `ambiguous`

The slice taxonomy was also expanded to reflect the actual failure modes we saw in
manual eval:

- `supported_control`
- `source_mix_up`
- `unsupported_overreach`
- `direct_contradiction`
- `cross_document_conflict`
- `cross_chunk_mismatch`
- `multiple_qa_mixed_support`
- `wrong_number_or_threshold`

### 2.2 Review acceleration pipeline

A large part of Phase 2 was not pure evaluation logic but review throughput. We built
an annotation flow that lets us prefill, prioritize, review, sync, and promote examples
into gold benchmark rows without directly trusting weak auto-labels.

Scripts added for this:

- `scripts/prefill_finreg_detector_phase2_annotations.py`
- `scripts/export_finreg_phase2_priority_review.py`
- `scripts/export_finreg_phase2_reviewer_packet.py`
- `scripts/draft_finreg_phase2_review_notes.py`
- `scripts/sync_finreg_phase2_review_labels.py`
- `scripts/build_finreg_phase2_gold_seed.py`

This pipeline let us:

1. derive suggested labels and review priority
2. isolate `p0/p1` cases first
3. generate compact reviewer packets
4. draft concise review notes
5. sync reviewed gold labels back into the working review files
6. build reviewed seed benchmarks for detector-only eval

### 2.3 Detector-only evaluation and score comparison

We added the two core Phase 2 evaluation scripts:

- `scripts/eval_finreg_detector_phase2.py`
- `scripts/compare_finreg_detector_phase2_scores.py`

These scripts let us evaluate two different things separately:

1. hard detector behavior
   hard predicted contradiction vs gold `contradicted`
2. soft score behavior
   whether score-based separation can distinguish `supported` from `contradicted`

This distinction ended up being critical, because the detector is clearly stronger as a
soft ranking model than as a hard contradiction classifier.

### 2.4 Seed expansion tooling

The first reviewed seed was too narrow, so we added candidate-mining and seed curation
tools:

- `scripts/mine_finreg_phase2_expansion_candidates.py`
- `scripts/mine_finreg_phase2_hard_risk_watchlist.py`
- `scripts/curate_finreg_phase2_seed_v2.py`
- `scripts/curate_finreg_phase2_seed_v3.py`

This let us move from a tiny early seed to a broader, more balanced benchmark that
contains meaningful `supported` and `contradicted` coverage.

## 3. How the benchmark evolved

### 3.1 Initial reviewed seed

The first reviewed seed was heavily skewed toward:

- `unsupported`
- `ambiguous`
- `partial`

That benchmark was useful for pipeline validation, but it could not tell us whether a
detector was good at finding contradictions, because it barely contained any clean
`supported` or `contradicted` rows.

### 3.2 Seed v2

We expanded the seed with conservative shortlist curation and added fallback row
construction from `per_question.jsonl` when benchmark-prefill coverage was incomplete.

This improved total size, but the class mix was still weak for final detector judgment.

### 3.3 Seed v3

We then ran a second expansion focused specifically on:

- clean supported controls
- clear contradicted comparisons
- wrong institution mapping
- false alignment claims
- source-to-answer reversals

Final tracked benchmark:

- `data/benchmarks/finreg_detector_phase2/v1/benchmark_v3_gold_seed.jsonl`

Current composition:

- `46` rows total
- `40` unique question ids
- `supported`: `8`
- `contradicted`: `7`
- `unsupported`: `11`
- `partial`: `9`
- `ambiguous`: `11`

This is not yet a full production-grade benchmark, but it is now strong enough to
produce meaningful detector-only findings.

## 4. What we found

### 4.1 Hard contradiction behavior is still broken

On `benchmark_v3_gold_seed.jsonl`, both `fever_local` and `targeted_v2` fail as hard
contradiction classifiers:

- `F1 Contradiction = 0`
- `Recall = 0`
- `Precision = 0`

Reason:

- neither detector emits hard contradiction predictions on this benchmark
- all gold `contradicted` rows are therefore missed

Operationally, this means neither detector is ready to be trusted as a production hard
contradiction gate.

### 4.2 Targeted still wins as a soft detector

Although hard contradiction behavior is still unusable, `targeted_v2` continues to show
better soft separation than `fever_local`.

Best score-separation results on the current benchmark:

- `targeted_v2 contradiction_prob_mean`: best supported-vs-contradicted F1 `0.875`
- `targeted_v2 hallucination_prob_topk`: best supported-vs-contradicted F1 `0.875`
- `targeted_v2 neutral_contradiction_margin`: best supported-vs-contradicted F1 `0.875`

For the same comparisons:

- `fever_local contradiction_prob_mean`: best F1 `0.857`
- `fever_local hallucination_prob_topk`: best F1 `0.857`
- `fever_local neutral_contradiction_margin`: best F1 `0.636`

Interpretation:

- `targeted_v2` is better at score-based ranking and separation
- `targeted_v2` is not better at firing a hard contradiction label
- the candidate remains promising, but only as a soft detector

### 4.3 Calibration is still weak

Calibration is not strong enough for direct gate promotion.

Current detector-only eval:

- `fever_local`: `ECE ≈ 0.119`, `Brier ≈ 0.143`
- `targeted_v2`: `ECE ≈ 0.155`, `Brier ≈ 0.148`

Interpretation:

- `targeted_v2` separates better
- `fever_local` is slightly less miscalibrated
- neither model currently looks production-ready as a fully trusted contradiction probability

### 4.4 The main failure mode is not "no signal", but "wrong signal geometry"

The benchmark confirms the original Phase 2 hypothesis:

- the detector is not fully blind
- it does produce risk signal
- but it does not map that risk cleanly into contradiction

The most important symptom is that `targeted_v2` often pushes higher contradiction-like
scores into:

- `ambiguous`
- `partial`
- some unsupported-overreach cases

instead of reserving that signal for clean `contradicted` rows.

So the real issue is not only thresholding. The deeper problem is label geometry:

- contradiction vs ambiguity is still blurred
- contradiction vs unsupported overreach is still blurred
- hard contradiction firing remains too conservative

## 5. What Phase 2 changed operationally

Before this branch, detector work was spread across:

- chunking sensitivity questions
- detector experiments
- manual eval artifacts
- training and gating code paths

After this branch, we now have:

- a detector-only Phase 2 plan
- a concrete benchmark schema
- an annotation/review acceleration pipeline
- seed benchmark build scripts
- detector-only eval and score-comparison scripts
- a stronger reviewed benchmark seed that is usable for detector diagnosis

That is the core outcome of the branch: we moved the detector work from loosely coupled
manual experiments into a repeatable evaluation workflow.

## 6. Current conclusion

The current conclusion for Phase 2 is:

- keep `balanced` as the production baseline
- keep `targeted_v2` as the optimization candidate
- do not promote `targeted_v2` as a hard contradiction detector yet
- treat `targeted_v2` as the best available soft detector candidate

## 7. Recommended next step

The next step should not be another threshold-only pass.

The correct follow-up is a training-and-labeling phase focused on:

- contradiction-heavy finreg slices
- ambiguity vs contradiction disambiguation
- wrong-institution and false-alignment cases
- wrong-number / wrong-threshold cases
- post-hoc calibration and score-to-gate mapping

In short:

- Phase 2 proved that the detector stack and benchmark workflow now exist
- Phase 2 also proved that the current candidate still needs data/loss/calibration work
- the next branch should be a `Phase 2.1` style sharpening pass on `targeted_v2`
