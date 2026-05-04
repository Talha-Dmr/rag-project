# Detector Optimization Phase 2

This document defines the next detector workstream after the targeted-detector exploration.
The goal is to improve detector quality for financial regulation grounding before running any
new chunking-driven conclusions.

## Decision Snapshot

- Locked production baseline detector: `balanced`
- Active baseline config: `config/gating_finreg_ebcar_logit_mi_sc009.yaml`
- Phase 2 optimization candidate: `targeted_v2`
- Candidate config: `config/gating_finreg_ebcar_logit_mi_sc009_targetedcontraguarddet_v2.yaml`
- Scope: detector-only benchmark, training data, calibration, and aggregation
- Explicit non-goal for this phase: chunking comparison or chunking promotion decisions

## Why This Phase Exists

The current repo already has detector training, inference, and gating integration. The main
remaining problem is not missing infrastructure. The problem is that contradiction signal is
often too soft for finreg grounding, so chunking comparisons get flattened before detector-side
differences can be trusted.

Phase 2 therefore treats the detector as its own product:

- lock `balanced` as the stable baseline
- evaluate `targeted_v2` independently from chunking
- sharpen finreg contradiction detection through data and calibration
- compare aggregation methods before changing gating policy again

## Working Hypothesis

`targeted_v2` is promising, but it still should not be judged only by end-to-end chunking
behavior. It should first beat or clearly complement `balanced` on a finreg-specific benchmark
that contains supported, unsupported, contradicted, partial, and ambiguous cases.

## Benchmark Plan

Create a detector benchmark that is independent from chunking and final RAG promotion logic.

Recommended label buckets:

- `supported`
- `unsupported`
- `contradicted`
- `partial`
- `ambiguous`

Recommended benchmark slices:

1. Supported answers
   Answer is materially grounded in retrieved context.
2. Unsupported overreach
   Answer adds claims not backed by retrieved evidence.
3. Direct contradiction
   Answer conflicts with retrieved evidence.
4. Partial support
   Answer gets one part right but overgeneralizes, omits caveats, or blends claims.
5. Ambiguous / insufficient evidence
   Retrieval does not support a clean judgment.
6. Cross-chunk or cross-document conflict
   Evidence pieces disagree or become risky only when compared jointly.
7. Multiple-QA / compound answers
   One answer mixes several regulatory subclaims with mixed support quality.

Recommended benchmark size for first stable version:

- 150 to 250 items total
- enough to support per-bucket analysis
- at least 30 contradiction-heavy examples
- at least 30 neutral/ambiguous examples

Existing assets to reuse:

- `docs/finreg_detector_eval_pipeline.md`
- `docs/finreg_manual_eval_v3_protocol.md`
- `evaluation_results/finreg_detector_manualeval_v3/...`
- draft builder flow from `build_finreg_eval_v2.py`

## Metrics

Do not use only `f1_macro`.

Primary metrics:

- `f1_contradiction`
- contradiction recall
- contradiction precision
- contradiction vs neutral confusion
- support precision on `high_entailment` style cases

Calibration metrics:

- ECE
- Brier score
- reliability plot for contradiction probability

Separation metrics:

- contradiction probability distribution by label bucket
- entailment vs contradiction margin by label bucket
- neutral vs contradiction margin by label bucket
- top-k contradiction signal vs mean contradiction signal

Operational diagnostics:

- false-positive rate on supported answers
- false-negative rate on contradicted answers
- partial-to-supported confusion
- ambiguous-to-contradicted confusion

## Data Strategy

The data should be expanded, but not by simply making the corpus larger.

Priority order:

1. Better eval labels
   The highest-value immediate step is a clean, contradiction-aware finreg eval set.
2. Better training examples
   Add examples that represent the exact failure modes the detector currently misses.
3. More corpus coverage
   Only expand the raw corpus when failures are caused by missing domain variation rather than
   detector confusion.

What should be added first to training data:

- contradiction-heavy finreg pairs
- hard negatives where answer sounds plausible but is unsupported
- cross-document conflict examples
- cross-chunk mismatch examples
- multiple-QA examples with mixed support status
- partial-support examples that should not collapse into full entailment
- wrong-number and wrong-threshold cases
- outdated-regulation and source-mix-up cases

What this means in practice:

- expanding the QA dataset is more urgent than blindly expanding the corpus
- targeted authoring of hard cases is more valuable than bulk weakly-labeled additions
- if corpus expansion happens, it should be guided by uncovered policy families and source types

Recommended ratio for the next training slice:

- keep a stable supported / neutral backbone from the current balanced pipeline
- add a contradiction-heavy finreg slice on top
- avoid turning the full dataset into a contradiction-only set

## Calibration And Aggregation Track

Phase 2 should optimize score behavior before changing hard thresholds.

Aggregation candidates to compare:

- `contradiction_prob_mean`
- `hallucination_prob_topk`
- `detector_conflict`
- `detector_conflict_consensus`
- entailment-contradiction margin
- neutral-contradiction margin

Calibration candidates:

- temperature scaling on detector logits
- per-domain calibration for finreg
- bucketed thresholding for supported vs risky zones

Decision rule for this track:

- prefer the method that improves contradiction recall without collapsing supported precision
- reject methods that only improve by globally shifting all answers into high-risk mode

## Work Sequence

1. Lock the baseline
   Keep `balanced` as the production reference.
2. Freeze the candidate
   Use `targeted_v2` as the optimization candidate for this phase.
3. Build benchmark v1
   Reuse manual-eval v3 artifacts and author missing buckets.
4. Run detector-only evaluation
   Compare `balanced` and `targeted_v2` on the same labeled set.
5. Add calibration layer
   Test temperature scaling and aggregation metrics on top of existing outputs.
6. Add training-data slice
   Introduce contradiction-heavy and partial-support finreg examples.
7. Retrain targeted candidate
   Keep architecture fixed while changing data and calibration inputs.
8. Re-evaluate
   Promote only if benchmark gains are clear and stable.
9. Return to chunking later
   Re-run chunking analysis only after detector benchmark results are convincing.

## Promotion Gate

`targeted_v2` should only be promoted beyond R&D if it satisfies most of the following:

- better contradiction recall than `balanced`
- no major supported-case precision collapse
- improved or at least non-regressed calibration
- visibly stronger score separation on contradiction-heavy finreg cases
- stable behavior across manual-review buckets

## Deliverables

- finreg detector benchmark v1
- benchmark documentation and annotation rules
- balanced vs targeted_v2 detector-only report
- calibration comparison report
- training-slice proposal for finreg contradiction sharpening
- promotion or no-promotion decision note
