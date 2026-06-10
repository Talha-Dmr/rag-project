# FinReg Stochastic Adapter Backlog

Date: 2026-05-28

## Purpose

Track the stochastic gating adapters that should remain active research candidates, so they do not
get forgotten while the main gating work continues.

This backlog is not the current production path. The current production-like baseline remains:

- `logit_mi`
- `guarded_v3`
- evidence subset sampling
- one retry

## Current Shortlist

Primary candidates:

1. `stochastic_sgbd`
   - Best LFM2 no-type replay result on 2026-05-28.
   - Candidate role: more conservative conflict-sensitive scalar adapter.

2. `stochastic_sghmc`
   - Good LFM2 no-type replay result with a less conservative action mix than `stochastic_sgbd`.
   - Candidate role: middle-ground stochastic adapter.

3. `stochastic_mirror_langevin`
   - Best refreshed result on the older 50Q shadow dump.
   - Candidate role: check whether mirror/simplex geometry is more stable across non-LFM2 dumps.

Secondary ablation:

4. `stochastic_wright_fisher`
   - Not the strongest in the latest LFM2 replay.
   - Keep as lower-priority ablation because it represents a different simplex-diffusion idea.

Lower priority unless new evidence appears:

- `stochastic_langevin`
- `stochastic_ou`
- `stochastic_prox_langevin`

## Why They Are Not Promoted Yet

The best adapter changes by evaluation source:

- LFM2 no-type replay: `stochastic_sgbd` was best.
- Older 50Q shadow dump: `stochastic_mirror_langevin` was best.

That means there is signal worth testing, but not enough stability to replace `logit_mi` in the
production-like path.

## When To Revisit

Revisit after the current gating track reaches a stable no-type baseline:

1. `guarded_v3` applied run is accepted as the current baseline.
2. `guarded_v4_lite` is either dropped or redesigned without dataset `type` metadata.
3. The next no-type gating candidate has been compared against `guarded_v3`.

At that point, run the stochastic shortlist as the next ablation block before making another
FullRAG80 claim.

FullRAG80 is now the canonical test dataset:

- `benchmarks/finreg/full_rag_questions.jsonl`

The 50-question current set can be used only as a fast smoke/replay tool. It should not decide
whether a stochastic adapter is promoted.

## Next Test Block

Run these tests in order.

1. Multi-seed replay on existing current50 details
   - seeds: `7`, `11`, `19` if matching dumps exist or can be generated
   - sources: `logit_mi`, `stochastic_sgbd`, `stochastic_sghmc`, `stochastic_mirror_langevin`,
     `stochastic_wright_fisher`
   - report: operating score, action mix, label-aware utility, action agreement with `logit_mi`,
     answer/retrieve-more accuracy

2. FullRAG80 no-type applied run for the top 1-2 adapters
   - same model: LFM2 2.6B
   - dataset: `benchmarks/finreg/full_rag_questions.jsonl`
   - same seed and retry settings as the `guarded_v3` baseline
   - no benchmark metadata at runtime

3. Optional 50Q diagnostic follow-up
   - use only to explain a failure mode or speed up debugging
   - do not use it as the final deciding result

## Acceptance Criteria

Promote a stochastic adapter only if it satisfies all of these:

- improves or matches `logit_mi` on FullRAG80 applied runs,
- does not increase sanity false abstains,
- reduces unsupported/hallucination risk among answered questions or improves retrieve-more
  behavior on conflict questions,
- is stable across at least two seeds,
- does not depend on dataset labels, `type`, expected behavior, or benchmark ids at runtime,
- has a clear implementation path in shared runtime code, not only in replay scripts.

## Current Artifacts

Retest artifacts from 2026-05-28:

- `evaluation_results/auto_eval/lfm2_stochastic_replay_current50_notype_seed7_wide.json`
- `evaluation_results/auto_eval/lfm2_stochastic_adapter_overlap_current50_notype_seed7.json`
- `evaluation_results/auto_eval/finreg_stochastic_replay_current50_seed7_wide_refresh.json`

Related diagnostic note:

- `docs/finreg_stochastic_gating_diagnostic_note.md`

## 2026-06-01 Extended Mathematical Adapter Sweep

New shadow/replay-only adapters were added to `scripts/eval_grounding_proxy.py` and included in
the replay/diagnostic defaults:

- `stochastic_sgld`
- `stochastic_adaptive_sgld`
- `dirichlet_simplex`
- `stein_particle`
- `conformal_margin`
- `risk_budgeted_bayesian`

External research checked during this pass:

- C-RAG, "Certified Generation Risks for Retrieval-Augmented Language Models"
  (`https://arxiv.org/abs/2402.03181`): supports treating RAG quality as a risk-bound problem, but
  requires an explicit bounded risk function and calibration procedure.
- CONFLARE, "CONFormal LArge language model REtrieval" (`https://arxiv.org/abs/2404.04287`):
  reinforces retrieval-uncertainty calibration as a first-class RAG problem.
- Conformal-RAG / conditional conformal factuality (`https://arxiv.org/abs/2506.20978`): points
  toward sub-claim quality/factuality guarantees rather than only whole-answer accuracy.
- SVGD uncertainty work (`https://arxiv.org/abs/2106.10760`): supports particle/ensemble-style
  uncertainty, but a scalar replay proxy is only a weak approximation of that idea.
- SGLD/Bayesian uncertainty background (`https://arxiv.org/abs/1409.0578`): supports Langevin-style
  posterior sampling, but the current adapter is not actually sampling model weights or detector
  posterior states.

These are still scalar replay adapters. They do not replace the current runtime evidence-subset
sampling policy. Their purpose is to test whether a more explicitly stochastic mathematical
adapter has enough signal to justify a runtime implementation.

Extended replay artifacts:

- `evaluation_results/auto_eval/deep_dive_stochastic_extended_replay_current50_notype_seed7.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_extended_replay_shadow_current50_seed7.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_extended_replay_shadow_current50_seed11.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_extended_replay_shadow_current50_seed19.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_extended_replay_applied_float32_current50_seed7.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_extended_replay_applied_float32_current50_seed11.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_extended_replay_applied_float32_current50_seed13.json`

Aggregate result across the seven replay sets:

| Source | Wins | Mean rank | Mean score | Score sd | Mean answer | Mean retrieve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `logit_mi` | 4 | 2.71 | 0.7284 | 0.0510 | 0.497 | 0.503 |
| `stochastic_mirror_langevin` | 2 | 3.00 | 0.7214 | 0.0189 | 0.623 | 0.377 |
| `stochastic_ou` | 0 | 4.71 | 0.6967 | 0.0366 | 0.734 | 0.266 |
| `stochastic_adaptive_sgld` | 0 | 5.43 | 0.7025 | 0.0345 | 0.626 | 0.374 |
| `stochastic_sgbd` | 1 | 6.00 | 0.6963 | 0.0389 | 0.694 | 0.306 |
| `stein_particle` | 0 | 8.00 | 0.6978 | 0.0305 | 0.706 | 0.294 |

Interpretation:

- `logit_mi` remains the strongest reference baseline by wins and mean rank.
- `stochastic_mirror_langevin` is the best mathematical stochastic candidate so far. It has a
  close mean score to `logit_mi`, much lower score variance, and wins two applied replay seeds.
- `stochastic_sgbd` is still useful as an LFM2/no-type challenger, but its strength did not hold
  across the wider replay set.
- `stochastic_adaptive_sgld` is a promising second-tier candidate; it repeatedly appears near the
  top but does not beat `stochastic_mirror_langevin` in aggregate.
- `stein_particle` has isolated strong runs, but its aggregate rank is not yet high enough to
  promote.
- `dirichlet_simplex`, `conformal_margin`, and `risk_budgeted_bayesian` did not beat the existing
  shortlist in this sweep.

Updated shortlist for the next adapter block:

1. Keep `logit_mi` as the baseline/reference.
2. Promote `stochastic_mirror_langevin` as the primary mathematical stochastic candidate.
3. Keep `stochastic_adaptive_sgld` as the secondary candidate.
4. Keep `stochastic_sgbd` as an LFM2-specific challenger, not the default choice.
5. Keep `stein_particle` only as an exploratory ablation.

Next action before a FullRAG160 claim:

- Do not run every adapter on DeepSeek FullRAG160.
- First move the top 1-2 candidates into a shared runtime-compatible adapter path or replay them
  from DeepSeek-derived per-question statistics.
- Then compare only `logit_mi`, `stochastic_mirror_langevin`, and optionally
  `stochastic_adaptive_sgld`/`stochastic_sgbd` against the current `vector_v3` evidence-subset
  policy.

## Expanded Candidate Families To Research Next

The current adapter sweep is not exhaustive. The next research block should widen the candidate
space, but it should separate candidates that can be tested with existing per-question statistics
from candidates that require new runtime signals.

### A. Directly Testable With Current Signals

These can be added as replay/shadow adapters without changing the generator:

1. Semantic-entropy-inspired adapter
   - Basis: semantic uncertainty / semantic entropy work.
   - Local approximation: combine detector label entropy, top-2 margin, label disagreement, and
     support score.
   - Runtime risk: low; all ingredients already exist.
   - Caveat: this is not true sampled semantic entropy unless we generate multiple answers.

2. CRAG-style retrieval evaluator adapter
   - Basis: Corrective RAG uses a retrieval evaluator to decide whether documents are good enough.
   - Local approximation: combine retrieval max/mean/spread, source consistency, context coverage,
     answer completeness, and support score.
   - Runtime risk: low; aligns strongly with the project goal because it evaluates evidence quality
     before trusting generation.

3. UAR/active-retrieval multi-criteria adapter
   - Basis: active retrieval methods decide when retrieval is useful instead of always retrieving.
   - Local approximation: separate uncertainty into orthogonal criteria:
     knowledge insufficiency, evidence conflict, source inconsistency, and answer incompleteness.
   - Runtime risk: low to medium; decision surface is more interpretable than a single blended
     scalar.

4. Evidence-instability adapter
   - Basis: our own `vector_v3` evidence-subset policy plus Speculative RAG's multi-subset drafting
     idea.
   - Local approximation: use subset action instability, subset answer rate, subset retrieve-more
     rate, and cross-subset risk spread.
   - Runtime risk: low; this is closest to what is already working.

5. Energy/margin adapter
   - Basis: semantic energy and margin-based hallucination detection.
   - Local approximation: use low top-2 margin, high entropy, low support, and contradiction-neutral
     gap.
   - Runtime risk: low; likely useful as a cheap detector-side signal.

### B. Testable But More Expensive

These require additional calls or more detector passes:

1. SelfCheck-style answer consistency
   - Generate several low-temperature candidate answers or drafts, then check whether they agree.
   - Useful for hallucination risk, but costly with DeepSeek and very costly locally.
   - Good fit as a small 20Q/50Q experiment before any FullRAG160 run.

2. Speculative multi-draft RAG
   - Draft answers from different evidence subsets, then verify/select.
   - Strong conceptual fit with our evidence subset sampling, but it changes the runtime pipeline
     more than a scalar gate.
   - Candidate role: future quality improvement after gating is stable.

3. True conformal/risk-controlled abstention
   - Requires a calibration set and a defined risk function.
   - Strong for thesis/research contribution if implemented carefully.
   - Not equivalent to the current `conformal_margin` replay proxy.

4. Bayesian/stochastic embedding retrieval
   - Sample query/chunk embeddings or perturb embedding space to estimate retrieval confidence.
   - Strong conceptual fit, especially for financial QA, but requires retrieval-layer changes.
   - Candidate role: retrieval-side uncertainty, not just answer-side gating.

### C. Probably Not Worth Prioritizing Now

These are valid research ideas but poorly matched to the current project constraints:

- Weight-space SGLD/HMC over the LLM: too expensive and not feasible with the current local hardware.
- Full hidden-state semantic entropy probes: promising, but require model internals/training and are
  awkward with API-only DeepSeek.
- Heavy RL-trained active retrieval policies: interesting but too large a scope for the current
  benchmark/debugging cycle.

## Expanded Research Sources

Additional sources checked after the initial sweep:

- "To Retrieve or Not to Retrieve? Uncertainty Detection for Dynamic Retrieval Augmented Generation"
  (`https://arxiv.org/abs/2501.09292`): dynamic retrieval can be driven by uncertainty metrics and
  can reduce retrieval calls while preserving much of QA quality.
- FLARE / "Active Retrieval Augmented Generation" (`https://arxiv.org/abs/2305.06983`): supports
  iterative retrieval based on low-confidence future generation.
- "Unified Active Retrieval for Retrieval Augmented Generation" (`https://arxiv.org/abs/2406.12534`):
  motivates multi-criterion retrieval decisions instead of one uncertainty threshold.
- "Corrective Retrieval Augmented Generation" (`https://arxiv.org/abs/2401.15884`): supports adding
  a retrieval/evidence evaluator before trusting generated answers.
- "Speculative RAG" (`https://arxiv.org/abs/2407.08223`): supports using distinct evidence subsets
  to produce diverse drafts and reduce long-context position bias.
- "UncertaintyRAG" (`https://arxiv.org/abs/2410.02719`): supports retrieval-side uncertainty based
  on span/chunk uncertainty, which maps to our retrieval quality problem more than to generation
  model selection.
- "SelfCheckGPT" (`https://arxiv.org/abs/2303.08896`): supports stochastic answer consistency as a
  black-box hallucination signal.
- "Semantic Entropy Probes" (`https://arxiv.org/abs/2406.15927`) and semantic uncertainty work
  (`https://arxiv.org/abs/2302.09664`): supports semantic-level uncertainty instead of raw token or
  label entropy.
- "Semantic Energy" (`https://arxiv.org/abs/2508.14496`): motivates energy/margin-style uncertainty
  beyond plain entropy.
- "Learning Conformal Abstention Policies" (`https://arxiv.org/abs/2502.06884`): supports adaptive
  risk/abstention thresholds, but requires a real calibration procedure.

## Revised Research Plan

Do not stop at the current six new adapters. The next controlled block should add and replay these
families in this order:

1. `retrieval_evaluator_crag`
2. `active_retrieval_multicriteria`
3. `semantic_entropy_proxy`
4. `evidence_instability`
5. `energy_margin`
6. `selfcheck_consistency` on a small sample only
7. `conformal_risk_calibrated` after a calibration split is defined
8. `bayesian_embedding_retrieval` as a retrieval-layer follow-up

Promotion rule:

- A scalar adapter can be promoted only if it changes action decisions in a useful way, not merely
  by shifting thresholds.
- Any expensive stochastic method must first prove value on 20Q/50Q before touching FullRAG160.
- The final contribution should remain project-aligned: better evidence-grounded selective
  answering, not just chasing one benchmark score.

## 2026-06-01 Expanded Replay Results

The first five cheap candidates from the revised research plan were implemented as replay/shadow
adapters:

- `retrieval_evaluator_crag`
- `active_retrieval_multicriteria`
- `semantic_entropy_proxy`
- `evidence_instability`
- `energy_margin`

Replay artifacts:

- `evaluation_results/auto_eval/deep_dive_stochastic_expanded_replay_current50_notype_seed7.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_expanded_replay_shadow_current50_seed7.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_expanded_replay_shadow_current50_seed11.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_expanded_replay_shadow_current50_seed19.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_expanded_replay_applied_float32_current50_seed7.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_expanded_replay_applied_float32_current50_seed11.json`
- `evaluation_results/auto_eval/deep_dive_stochastic_expanded_replay_applied_float32_current50_seed13.json`

Aggregate result across the same seven replay sets:

| Source | Wins | Mean rank | Mean score | Score sd | Mean answer | Mean retrieve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `evidence_instability` | 3 | 2.00 | 0.7537 | 0.0349 | 0.543 | 0.457 |
| `logit_mi` | 1 | 4.43 | 0.7284 | 0.0510 | 0.497 | 0.503 |
| `stochastic_mirror_langevin` | 0 | 4.57 | 0.7214 | 0.0189 | 0.623 | 0.377 |
| `active_retrieval_multicriteria` | 1 | 6.14 | 0.7215 | 0.0197 | 0.526 | 0.474 |
| `stochastic_ou` | 0 | 7.00 | 0.6967 | 0.0366 | 0.734 | 0.266 |
| `stochastic_adaptive_sgld` | 0 | 7.71 | 0.7025 | 0.0345 | 0.626 | 0.374 |
| `semantic_entropy_proxy` | 1 | 10.71 | 0.7008 | 0.0449 | 0.623 | 0.377 |
| `retrieval_evaluator_crag` | 1 | 15.43 | 0.6936 | 0.0438 | 0.757 | 0.243 |
| `energy_margin` | 0 | 15.86 | 0.6903 | 0.0297 | 0.717 | 0.283 |

Per-run highlights:

- `retrieval_evaluator_crag` was best on the LFM2 no-type seed7 replay
  (`0.7847`, `answer=37`, `retrieve_more=13`), but did not generalize across the other replay
  sets.
- `evidence_instability` won three of seven replay sets and was top-3 in every replay set. This is
  the strongest expanded candidate.
- `active_retrieval_multicriteria` won applied float32 seed11 and was strong on applied seed7/13.
- `semantic_entropy_proxy` won shadow seed19 but was not stable enough across all sets.
- `energy_margin` did not materially improve over the older candidates.

Updated interpretation:

- The strongest signal is not another Langevin-style scalar transform. The strongest signal is the
  stability of the evidence-subset decision itself.
- `evidence_instability` is highly project-aligned because it formalizes the stochastic evidence
  perturbation idea already used by `vector_v3`.
- `active_retrieval_multicriteria` is the best non-subset cheap candidate and is more interpretable
  than the old scalar adapters.
- `stochastic_mirror_langevin` remains the best purely mathematical scalar stochastic candidate,
  but it is no longer the top overall research candidate after expanding the search.

Revised next shortlist:

1. `evidence_instability` as the primary stochastic gating direction.
2. `active_retrieval_multicriteria` as the primary cheap non-subset challenger.
3. `stochastic_mirror_langevin` as the best pure mathematical scalar adapter.
4. `logit_mi` as the baseline/reference.
5. `semantic_entropy_proxy` as an exploratory ablation only.

Next implementation direction:

- Move `evidence_instability` and `active_retrieval_multicriteria` toward shared runtime-compatible
  adapter code.
- Compare them against current `vector_v3` rather than treating them as independent of it.
- Do not run `retrieval_evaluator_crag` or `energy_margin` on FullRAG160 unless later reformulated.

## 2026-06-03 Runtime Adapter Bridge

Runtime-compatible adapter code was added after the expanded replay sweep.

Shared implementation:

- `src/rag/stochastic_epistemic_adapter.py`

Runtime integration points:

- `src/rag/rag_pipeline.py`
- `src/rag/evidence_sampling_policy.py`

Replay/diagnostic scripts now delegate to the shared adapter implementation:

- `scripts/eval_grounding_proxy.py`
- `scripts/replay_finreg_stochastic_gate.py`
- `scripts/analyze_finreg_stochastic_gate_diagnostics.py`

This prevents the replay formulas and runtime formulas from drifting apart.

Runtime use cases:

1. Scalar uncertainty adapter for the main gate:

   ```yaml
   gating:
     epistemic_adapter: stochastic_mirror_langevin
     uncertainty_metric: uncertainty_epistemic_adapter
   ```

   If `uncertainty_metric` is not set to `uncertainty_epistemic_adapter`, the adapter score is
   still written into `stats["uncertainty_epistemic_adapter"]` for diagnostics but does not change
   the main gate decision.

2. Evidence-subset adapter policy:

   ```yaml
   gating:
     evidence_sampling:
       enabled: true
       shadow_only: false
       policy: adapter_evidence_instability
       adapter_threshold: 0.42
   ```

   This uses the shared `evidence_instability` score over evidence-subset summary fields. It should
   be compared against `vector_v3`, not treated as unrelated to it.

3. Active retrieval challenger:

   ```yaml
   gating:
     evidence_sampling:
       enabled: true
       shadow_only: false
       policy: adapter_active_retrieval
       adapter_threshold: 0.25
   ```

   This uses the shared `active_retrieval_multicriteria` score over retrieval, support, source, and
   detector-conflict summary fields.

Verification:

- Compile check passed for the shared adapter, runtime pipeline, evidence policy, and replay tools.
- Smoke replay on `finreg_grounding_proxy_evidence_applied_float32_current50_seed7_details.jsonl`
  preserved the expanded result order:
  `evidence_instability` first, `active_retrieval_multicriteria` second,
  `stochastic_mirror_langevin` third.

Important limitation:

- This bridge makes the shortlist runtime-compatible; it does not prove the new runtime policies on
  FullRAG160 yet.
- The next empirical step is a small applied run or a DeepSeek-derived replay comparing:
  `vector_v3`, `adapter_evidence_instability`, `adapter_active_retrieval`, and
  `stochastic_mirror_langevin` as a scalar uncertainty adapter.

## 2026-06-04 Mirror Langevin v2 Research Pass

Question:

- Can a more explicitly simplex-aware `stochastic_mirror_langevin_v2` beat the current
  `stochastic_mirror_langevin` scalar adapter without turning into an evidence-instability clone?

Implementation tested:

- Added `stochastic_mirror_langevin_v2` to `src/rag/stochastic_epistemic_adapter.py`.
- Kept v1 unchanged.
- Fixed a v1 research weakness in the v2 formula only: empty detector probability vectors are no
  longer treated as high-confidence simplex corners. V2 multiplies mirror geometry terms by observed
  detector probability mass.
- V2 blends:
  - baseline epistemic inertia,
  - detector-mass-gated mirror tension,
  - conflict dispersion,
  - boundary/margin risk,
  - support risk,
  - unsupported neutral/contradiction mass.

Replay outputs:

- Coarse-grid replay reports:
  - `evaluation_results/auto_eval/mirror_v2_replay_current50_notype_seed7.json`
  - `evaluation_results/auto_eval/mirror_v2_replay_shadow_current50_seed7.json`
  - `evaluation_results/auto_eval/mirror_v2_replay_shadow_current50_seed11.json`
  - `evaluation_results/auto_eval/mirror_v2_replay_shadow_current50_seed19.json`
  - `evaluation_results/auto_eval/mirror_v2_replay_applied_float32_current50_seed7.json`
  - `evaluation_results/auto_eval/mirror_v2_replay_applied_float32_current50_seed11.json`
  - `evaluation_results/auto_eval/mirror_v2_replay_applied_float32_current50_seed13.json`
- Fine-threshold v2-only replay reports:
  - `evaluation_results/auto_eval/mirror_v2_fine_replay_current50_notype_seed7.json`
  - `evaluation_results/auto_eval/mirror_v2_fine_replay_shadow_current50_seed7.json`
  - `evaluation_results/auto_eval/mirror_v2_fine_replay_shadow_current50_seed11.json`
  - `evaluation_results/auto_eval/mirror_v2_fine_replay_shadow_current50_seed19.json`
  - `evaluation_results/auto_eval/mirror_v2_fine_replay_applied_float32_current50_seed7.json`
  - `evaluation_results/auto_eval/mirror_v2_fine_replay_applied_float32_current50_seed11.json`
  - `evaluation_results/auto_eval/mirror_v2_fine_replay_applied_float32_current50_seed13.json`

Aggregate with fine-threshold v2:

| Adapter | Wins | Mean rank | Mean score | Score sd | Answer rate | Retrieve rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `evidence_instability` | 4 | 1.86 | 0.7537 | 0.0349 | 0.543 | 0.457 |
| `logit_mi` | 1 | 3.29 | 0.7284 | 0.0510 | 0.497 | 0.503 |
| `stochastic_mirror_langevin_v2` | 1 | 4.00 | 0.7149 | 0.0438 | 0.600 | 0.400 |
| `active_retrieval_multicriteria` | 1 | 4.14 | 0.7215 | 0.0197 | 0.526 | 0.474 |
| `stochastic_mirror_langevin` | 0 | 3.57 | 0.7214 | 0.0189 | 0.623 | 0.377 |
| `stochastic_adaptive_sgld` | 0 | 5.57 | 0.7025 | 0.0345 | 0.626 | 0.374 |
| `stochastic_sgbd` | 0 | 5.57 | 0.6963 | 0.0389 | 0.694 | 0.306 |

Interpretation:

- V2 is not the new primary mathematical adapter. It wins one no-type replay set after fine
  calibration, but loses to v1 on mean score and stability.
- V2 is still useful as a research artifact because it isolates the detector-mass issue in the
  mirror geometry: missing detector vectors should not be interpreted as simplex certainty.
- The current decision remains:
  1. `evidence_instability` is the strongest overall stochastic gating direction.
  2. `stochastic_mirror_langevin` remains the best pure mathematical scalar adapter.
  3. `stochastic_mirror_langevin_v2` stays in backlog for future formula work, not promotion.

## 2026-06-04 Detector-Only Gate Ablation

Motivation:

- A project contributor raised the concern that most of the behavior may come from the detector
  itself, and that the gate may be relatively unimportant.
- This is a valid risk. The right answer is an ablation, not an argument.

Detector-only adapter sources added:

- `detector_contradiction`: direct `contradiction_prob_mean`
- `detector_uncertainty`: direct detector `uncertainty_mean`
- `detector_label_disagreement`: direct label disagreement
- `detector_conflict_mass`: direct conflict mass
- `detector_margin_risk`: `1 - top2_margin_mean`
- `detector_support_risk`: `1 - support_score`

These are intentionally simple. They use the same replay decision policy and threshold sweep as the
gate adapters, but they do not blend multiple stochastic/evidence-stability signals.

Replay outputs:

- `evaluation_results/auto_eval/detector_only_ablation_current50_notype_seed7.json`
- `evaluation_results/auto_eval/detector_only_ablation_shadow_current50_seed7.json`
- `evaluation_results/auto_eval/detector_only_ablation_shadow_current50_seed11.json`
- `evaluation_results/auto_eval/detector_only_ablation_shadow_current50_seed19.json`
- `evaluation_results/auto_eval/detector_only_ablation_applied_float32_current50_seed7.json`
- `evaluation_results/auto_eval/detector_only_ablation_applied_float32_current50_seed11.json`
- `evaluation_results/auto_eval/detector_only_ablation_applied_float32_current50_seed13.json`

Aggregate source ranking:

| Source | Wins | Mean rank | Mean score | Score sd | Answer rate | Retrieve rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `evidence_instability` | 3 | 2.57 | 0.7740 | 0.0374 | 0.429 | 0.571 |
| `detector_support_risk` | 1 | 2.86 | 0.7589 | 0.0293 | 0.654 | 0.346 |
| `detector_contradiction` | 1 | 3.00 | 0.7634 | 0.0241 | 0.551 | 0.449 |
| `stochastic_mirror_langevin` | 1 | 5.00 | 0.7342 | 0.0214 | 0.606 | 0.394 |
| `detector_uncertainty` | 1 | 5.14 | 0.7425 | 0.0398 | 0.497 | 0.503 |
| `active_retrieval_multicriteria` | 0 | 4.43 | 0.7461 | 0.0255 | 0.491 | 0.509 |
| `logit_mi` | 0 | 6.14 | 0.7425 | 0.0398 | 0.497 | 0.503 |
| `detector_conflict_mass` | 0 | 7.29 | 0.7092 | 0.0259 | 0.634 | 0.366 |
| `detector_label_disagreement` | 0 | 8.57 | 0.6894 | 0.0355 | 0.700 | 0.300 |
| `detector_margin_risk` | 0 | 10.00 | 0.6450 | 0.0000 | 0.000 | 1.000 |

Group comparison:

| Group | Mean score | Score sd | Wins vs other group |
| --- | ---: | ---: | ---: |
| Best detector-only source per set | 0.7825 | 0.0290 | 2 |
| Best non-detector gate source per set | 0.7863 | 0.0266 | 4 |

Interpretation:

- The criticism is partially correct: simple detector-only thresholds are strong baselines.
- The gate is not irrelevant: the best non-detector gate group still wins more sets and has a
  slightly higher aggregate score.
- The current evidence does not support claiming that stochastic gating dominates detector-only
  behavior by a large margin.
- The defensible project contribution is narrower and stronger:
  detector signals are converted into safer RAG actions through calibrated gating and
  evidence-subset stability, with detector-only thresholding as the required baseline.

Decision:

- Keep detector-only ablation in every future 160Q / DeepSeek replay.
- Do not present gate improvements without a detector-only baseline.
- Treat `evidence_instability` as the main stochastic gating candidate only if it continues to beat
  the best detector-only source on FullRAG160 or improves stability/operating behavior in a way the
  detector-only source does not.

## 2026-06-10 FinReg160-Hard And Posterior Follow-Up

Motivation:

- The latest final 160Q comparison suggests the canonical FinReg set may be too easy for the
  baseline: normal RAG already reaches very high expected-behavior match, so the stochastic layer
  mostly appears as extra abstention.
- The canonical 160Q set should stay fixed as the regression benchmark. Instead of rewriting it,
  use a harder derived benchmark to test whether stochastic gating adds value under low evidence,
  partial support, misattribution, and cross-source uncertainty.

New hard benchmark:

- `benchmarks/finreg/full_rag_questions_hard.jsonl`
- Generated by `scripts/build_finreg_hard_benchmark.py`
- Same row schema as the canonical benchmark, plus `source_id` to trace each hard case back to its
  canonical source question.

Current hard distribution:

| Question type | Count |
| --- | ---: |
| `factual_supported` | 20 |
| `hard_factual_completeness` | 20 |
| `false_premise_misattribution` | 30 |
| `low_evidence_specific_claim` | 40 |
| `cross_source_conflict` | 30 |
| `partial_support_overclaim` | 20 |

How to use it:

- Do not replace the canonical 160Q benchmark.
- Use canonical 160Q for final regression claims and report continuity.
- Use FinReg160-Hard to develop and pressure-test:
  - detector-only baselines,
  - `vector_v3`,
  - `adapter_evidence_instability`,
  - `adapter_active_retrieval`,
  - `stochastic_mirror_langevin`,
  - any future posterior-style method.

Posterior sampling follow-up:

- True posterior sampling is not implemented yet. Current project code has evidence-subset
  sampling, stochastic scalar adapters, and Bayesian/Langevin-inspired proxies, but it does not
  sample LLM weights, detector posterior states, or a calibrated posterior over answers.
- Keep posterior sampling as a separate research branch after the hard benchmark and detector-only
  ablations are stable.
- First practical target: a black-box posterior-style approximation over evidence subsets and
  answer/detector outcomes, not weight-space SGLD/HMC over the LLM.
- Acceptance bar: it must beat or explain the gap against detector-only thresholds on hard cases,
  not merely add randomness or increase abstention.
