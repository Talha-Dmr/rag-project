# FinReg Gating Next Steps: Avoid Benchmark-Specific Fixes

Date: 2026-05-28

## Decision

Do not fix FullRAG80 by adding benchmark-specific rules.

FullRAG80 is now the canonical test dataset:

- `benchmarks/finreg/full_rag_questions.jsonl`

It should be used to diagnose failures and measure improvement, not to drive runtime branching
based on benchmark metadata.

## What Would Be Benchmark-Specific And Should Be Avoided

Avoid:

- branching on `fullrag_*` ids,
- using `question_type` from the benchmark file at runtime,
- using `expected_behavior`, `expected_points`, or `forbidden_claims` as runtime inputs,
- adding per-topic answer templates for the 80 benchmark questions,
- tuning prompts to pass known benchmark rows rather than improving general evidence handling.

These changes would overfit the benchmark and would not represent a better RAG system.

## What Is Acceptable

Use benchmark failures to identify general policy weaknesses.

Acceptable general improvements:

- If the evidence does not support a specific requested claim, the answer should say so.
- If the question contains a false premise, the answer should not accept the premise.
- If evidence is partial but relevant, the system can give a cautious partial answer instead of hard abstaining.
- If retrieved context is topic-adjacent but not actually responsive, the gate should avoid confident answers.
- If risk appears reducible by more evidence, the first action should be `retrieve_more`, not final abstain.
- If risk remains high after the evidence budget is exhausted, final abstain is appropriate.

These rules are benchmark-independent because they depend only on the user question, generated answer,
retrieved evidence, detector outputs, and stability under evidence perturbation.

## Current Gating Track

The active gating work is not model selection. Model selection identified LFM2 2.6B as the best
current local generator candidate, but FullRAG80 shows that gating/retrieval policy is now the
dominant issue.

Current relevant gating work:

1. `logit_mi` remains the scalar baseline.
2. Old stochastic proxy adapters are not enough on their own because many collapse to the same
   action sequence after threshold calibration.
3. Evidence subset sampling is the most useful stochastic signal so far.
4. `guarded_v3` is the best current shadow/retry candidate from the earlier 50Q evidence-budget
   validation.
5. `guarded_v4_lite` was a promising applied candidate, but it needs fair one-retry comparison
   against `guarded_v3`.

Stochastic adapter follow-up is tracked separately so it does not get lost:

- `docs/finreg_stochastic_adapter_backlog.md`

That backlog should be revisited after the current no-type `guarded_v3` baseline is accepted and
before making another FullRAG80 claim.

## Dataset Usage Rule

Use FullRAG80 as the main test dataset from this point forward.

The older 50-question FinReg file can still be useful for fast smoke checks, debugging, replay, and
iteration, but it should not be treated as the deciding test dataset.

The practical workflow is:

1. Use 50Q only when a run would otherwise be too slow for quick iteration.
2. Use FullRAG80 for the decision-making run.
3. Report clearly whether a result came from 50Q smoke/replay or the 80Q test dataset.

## Hardware Placement Rule

When VRAM can support it, keep the detector on GPU. Do not move detector work to CPU by default,
because CPU detector runs are materially slower.

CPU detector placement is acceptable only as a fallback for OOM/instability or for an explicit
isolation experiment, and the run notes should say so.

## FullRAG80 Lesson

LFM2 FullRAG80 automatic result:

- total: 80
- expected behavior match: 50.0%
- abstain rate: 36.2%
- forbidden claim hit rate: 0.0%
- mean expected point coverage: 0.302

The mismatch analysis found four broad failure classes:

- false abstain on answerable questions,
- low-evidence over-answering,
- false-premise non-refutation,
- answered but low coverage or topic drift.

These should not become benchmark-specific branches. They should become general evidence-grounded
gate checks.

## 2026-05-28 FullRAG80 Guarded-Answer Update

Completed run:

- run: `fullrag80_lfm2_guarded_v3_guarded_answer_cpu_detector`
- config: `gating_finreg_lfm2_26b_local_rtx2070_evidence_retry_cpu_detector`
- questions: `benchmarks/finreg/full_rag_questions.jsonl`
- total: `80`
- expected behavior match: `63.75%` (`51/80`)
- abstain rate: `65.0%` (`52/80`)
- answer rate: `35.0%` (`28/80`)
- forbidden claim hit rate: `0.0%`
- mean expected point coverage: `0.2226`

This run added guarded answering after evidence sampling and exhausted `retrieve_more` handling. It
improves the previous stable CPU-detector FullRAG80 baseline (`56.25%` match, `50.0%` abstain,
`0.0%` forbidden claim hit rate), but the improvement comes with a higher abstain rate.

Mismatch breakdown:

- total mismatches: `29`
- by question type: `14` factual-supported, `13` multi-source-nuanced, `2` false-premise
- by action: `21` abstained after `retrieve_more`, `8` answered but failed expected behavior
- forbidden claim rows: `0`

Interpretation:

- The main remaining issue is false abstention, not forbidden claims.
- The guard is safer than the earlier baseline, but too conservative on answerable factual and
  multi-source nuanced questions.
- The next change should improve answerability detection from runtime retrieval/evidence signals,
  not by using FullRAG80 ids, types, expected behavior, or expected answer points.

Stability note:

- GPU detector placement was attempted but failed in full runs with CUDA misaligned-address or
  illegal-instruction errors.
- The current stable path uses LFM2 on CUDA and detector/logit sampling on CPU.
- LFM2 CUDA generation can still fail intermittently; `scripts/run_fullrag80_lfm2_stable.sh` uses
  `--resume` so a crash continues from `per_question.partial.jsonl` instead of restarting.

## 2026-05-28 FullRAG80 Relaxed-Answer Update

Completed run:

- run: `fullrag80_lfm2_guarded_v3_relaxed_answer_cpu_detector`
- config: `gating_finreg_lfm2_26b_local_rtx2070_evidence_retry_cpu_detector_relaxed_answer`
- questions: `benchmarks/finreg/full_rag_questions.jsonl`
- total: `80`
- expected behavior match: `77.5%` (`62/80`)
- abstain rate: `45.0%` (`36/80`)
- answer rate: `55.0%` (`44/80`)
- forbidden claim hit rate: `0.0%`
- mean expected point coverage: `0.3222`

This run relaxed only runtime evidence/retrieval guards after evidence sampling:

- final answer allowed after exhausted `retrieve_more` at lower retrieval/source thresholds,
- answer guard allows slightly higher subset answer-include risk,
- no benchmark ids, benchmark question types, expected behavior, expected answer points, or forbidden
  claims are used at runtime.

Result versus the previous guarded-answer run:

- match improved from `63.75%` (`51/80`) to `77.5%` (`62/80`),
- abstain dropped from `65.0%` to `45.0%`,
- forbidden claim hit rate remained `0.0%`,
- remaining mismatches dropped from `29` to `18`.

Mismatch breakdown:

- by type: `8` multi-source-nuanced, `7` factual-supported, `2` false-premise, `1`
  low-evidence-policy
- by action: `13` answered but failed expected behavior, `5` abstained after `retrieve_more`

Interpretation:

- The main bottleneck moved from false abstention to answer quality/coverage.
- The relaxed guard is a better current FullRAG80 candidate than the stricter guarded-answer config.
- The next fix should not broadly relax gating further. It should address answered-but-wrong cases:
  low-evidence over-answering, false-premise non-refutation, and topic drift in retrieved context.

## 2026-05-28 FullRAG80 Answer-Quality Guard Update

Completed run:

- run: `fullrag80_lfm2_guarded_v3_relaxed_answer_quality_guard_cpu_detector`
- config: `gating_finreg_lfm2_26b_local_rtx2070_evidence_retry_cpu_detector_relaxed_answer_quality_guard`
- questions: `benchmarks/finreg/full_rag_questions.jsonl`
- total: `80`
- expected behavior match: `81.25%` (`65/80`)
- abstain rate: `52.5%` (`42/80`)
- answer rate: `47.5%` (`38/80`)
- forbidden claim hit rate: `0.0%`
- mean expected point coverage: `0.3044`
- answer-quality rewrite rate: `57.5%` (`46/80`)

This run added benchmark-independent answer-quality coverage signals to final answer guards:

- low answer completeness can block final answering after exhausted `retrieve_more`,
- low context coverage can block final answering,
- question phrases such as "but not", "not an exact", "not a specific", and "not a fixed" are
  treated as specific unsupported-claim markers by the answer-quality audit.

Result versus the relaxed-answer baseline:

- match improved from `77.5%` (`62/80`) to `81.25%` (`65/80`),
- forbidden claim hit rate remained `0.0%`,
- abstain rose from `45.0%` to `52.5%`,
- answer count dropped from `44` to `38`,
- quality rewrites increased latency and caused one LFM2 CUDA illegal-memory crash; the stable
  runner resumed and completed the run.

Behavior changes versus the relaxed-answer baseline:

- improved: `fullrag_012`, `fullrag_017`, `fullrag_026`, `fullrag_047`, `fullrag_063`
- regressed: `fullrag_061`, `fullrag_072`
- net: `+3` expected-behavior matches

Mismatch breakdown:

- total mismatches: `15`
- by type: `7` factual-supported, `6` multi-source-nuanced, `1` false-premise,
  `1` low-evidence-policy
- by action: `9` false abstains after `retrieve_more`, `6` answered but failed expected behavior

Interpretation:

- The quality guard is the best automatic FullRAG80 result so far, but it is more conservative and
  slower than the relaxed-answer baseline.
- The main remaining issue has shifted back toward false abstention.
- Next work should tune or condition the quality guard using runtime evidence signals so it keeps
  the low-coverage answer improvements without suppressing answerable factual and multi-source
  questions.

## Recommended Next Experiment

Run a fair, benchmark-independent gating comparison:

1. Compare candidate policies on FullRAG80 with the same model, same questions, same seeds, and one
   retry enabled.
2. Report action transitions:
   - `retrieve_more -> none`
   - `retrieve_more -> retrieve_more`
   - `retrieve_more -> abstain`
   - `none -> retrieve_more`
   - `none -> none`
3. Report answer quality proxies among answered questions.
4. Inspect failures by evidence relationship, not by benchmark id.
5. Only promote a policy if it improves selective generation quality without simply memorizing the
   benchmark structure.

## 2026-05-28 FullRAG80 Quality-Guard Relaxed-Escape Update

Completed run:

- run: `fullrag80_lfm2_guarded_v3_quality_guard_relaxed_escape_v2_cpu_detector`
- config: `gating_finreg_lfm2_26b_local_rtx2070_evidence_retry_cpu_detector_quality_guard_relaxed_escape`
- questions: `benchmarks/finreg/full_rag_questions.jsonl`
- total: `80`
- expected behavior match: `82.5%` (`66/80`)
- abstain rate: `47.5%` (`38/80`)
- answer rate: `52.5%` (`42/80`)
- forbidden claim hit rate: `0.0%`
- mean expected point coverage: `0.3590`
- answer-quality rewrite rate: `52.5%` (`42/80`)

This run keeps the answer-quality guard, then adds a runtime-only relaxed escape after exhausted
`retrieve_more` when the final candidate answer has stronger evidence quality:

- answer completeness at least `0.70`,
- context coverage at least `0.70`,
- relaxed retrieval/source/conflict ceilings still pass,
- closed-probe questions are blocked from this relaxed escape unless the answer directly refutes
  the premise.

The closed-probe block is intentionally benchmark-independent. It uses only question wording and
answer text, not FullRAG80 ids, types, expected behavior, expected points, or forbidden claims.

Result versus the answer-quality guard baseline:

- match improved from `81.25%` (`65/80`) to `82.5%` (`66/80`),
- forbidden claim hit rate remained `0.0%`,
- abstain dropped from `52.5%` to `47.5%`,
- answer count rose from `38` to `42`,
- the run hit two LFM2 CUDA generation crashes; the stable runner resumed and completed.

Behavior changes versus the answer-quality guard baseline:

- improved: `fullrag_015`, `fullrag_061`, `fullrag_067`
- regressed: `fullrag_017`, `fullrag_063`
- net: `+1` expected-behavior match

Relaxed escape activations:

- `fullrag_015`, `fullrag_028`, `fullrag_058`, `fullrag_067`, `fullrag_074`
- all five relaxed-escape activations matched expected behavior in this run

Closed-probe relaxed-escape blocks:

- `fullrag_002`, `fullrag_006`, `fullrag_026`, `fullrag_042`, `fullrag_046`
- all five blocks matched expected behavior in this run

Mismatch breakdown:

- total mismatches: `14`
- by type: `7` factual-supported, `5` multi-source-nuanced, `1` false-premise,
  `1` low-evidence-policy
- by action: `8` false abstains after `retrieve_more`, `6` answered but failed expected behavior

Interpretation:

- This is the best automatic FullRAG80 result so far.
- The gain is real but modest, because generation variation and borderline gating still caused two
  regressions versus the quality-guard baseline.
- Treat the relaxed escape as the current best candidate, not as a final policy.
- Next work should stabilize the borderline cases with runtime evidence/answer-quality signals,
  especially rows that have high context coverage but low answer completeness or vice versa.

Canonical test dataset:

- `benchmarks/finreg/full_rag_questions.jsonl`

Fast smoke/debug dataset only:

- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl`

## 2026-05-28 Gating Comparison Update

The LFM2 one-retry comparison was run on the 50-question current FinReg set with:

- config: `gating_finreg_lfm2_26b_local_rtx2070_evidence_retry_smoke`
- questions: `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl`
- seed: `7`
- evidence subset size: `4`
- evidence subsets: `4`
- max retries: `1`

Typed comparison, where the eval script passes the dataset `type` field into the evidence-sampling
policy:

| Policy | Answer | Abstain | Retrieve-more actions | Sanity abstain | Conflict abstain | Hallucination flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `guarded_v3` | 33 | 17 | 17 | 2/10 | 15/40 | 38/50 |
| `guarded_v4_lite` | 26 | 24 | 24 | 0/10 | 24/40 | 36/50 |

Typed result: `guarded_v4_lite` fixes the sanity false-abstain issue and slightly reduces
hallucination flags, but it does that by using the question `type` label and by abstaining much more
on conflict questions.

Production-like no-type comparison, using the same questions with only `id` and `query`:

| Policy | Answer | Abstain | Retrieve-more actions | Sanity abstain | Conflict abstain | Hallucination flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `guarded_v3` | 33 | 17 | 17 | 3/10 | 14/40 | 38/50 |
| `guarded_v4_lite` | 19 | 31 | 31 | 4/10 | 27/40 | 38/50 |

No-type result: the apparent `guarded_v4_lite` improvement is not robust without the dataset
`type` field. It becomes substantially more conservative, increases false abstains on sanity
questions, and no longer improves the hallucination flag count.

Conclusion:

- Do not promote `guarded_v4_lite` as-is.
- Treat the typed result as a useful diagnostic, not as a production-ready policy win.
- The next fix should remove dependency on benchmark metadata by deriving answerability/conflict
  intent from runtime inputs, or by making the evidence-sampling policy robust without a question
  type hint.
- Keep `guarded_v3` as the safer current applied policy until that metadata dependency is removed.

## Acceptance Criteria

A gating change is acceptable only if it improves general behavior:

- fewer unsupported answers among answered responses,
- fewer unnecessary abstains on answerable questions,
- better use of `retrieve_more` as an intermediate action,
- stable behavior across seeds,
- explainable action changes at question/evidence level,
- no use of benchmark-only metadata in runtime policy.
