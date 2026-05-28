# FinReg Stochastic Gating Diagnostic Note

This note records the current decision on stochastic gating for the FinReg RAG stack.
It should be read together with:

- `docs/current_finreg_baseline.md`
- `docs/calibration_policy.md`
- `docs/future_work_langevin.md`

## Current Conclusion

The current stochastic gate experiments should not be treated as final stochastic gating
methods yet.

The observed high non-answer rates from several stochastic adapters were mostly caused by
scale mismatch:

- `logit_mi` epistemic values are small, roughly in the `0.00-0.04` band on the current
  50Q FinReg dump.
- Several stochastic proxy adapters produce much larger values, often in the `0.07-0.30`
  range.
- Applying the same `epi_threshold` range to all sources makes the stochastic adapters look
  excessively conservative.

When source-specific thresholds are widened, many stochastic adapters converge to almost the
same action mix as `logit_mi`. This is also not a strong result: it suggests that the current
adapters are mostly reweighted scalar transforms of the same underlying detector/retrieval
statistics rather than independent stochastic decision mechanisms.

## Important Interpretation

A high non-answer rate is not automatically bad.

For this project, the gate should:

- answer when the evidence is sufficient
- retrieve more when uncertainty is reducible by more evidence
- abstain when the available corpus/evidence is insufficient or conflict is irreducible
- avoid unsupported answers even when that lowers answer coverage

Therefore, `answer_rate`, `retrieve_more_rate`, `abstain_rate`, or total non-answer rate must
not be used alone as success metrics.

The useful target is selective generation quality:

- high safe-answer coverage on answerable questions
- low hallucination/contradiction rate among answered questions
- low unnecessary non-answer rate on sanity/answerable questions
- high retrieve/abstain rate on hard, underspecified, conflicting, or unsupported questions

## Current Weakness

The current stochastic adapters in `scripts/eval_grounding_proxy.py` are proxy-level adapters.
They mostly reuse the same fixed per-question statistics:

- contradiction probability
- label disagreement
- neutral probability
- uncertainty mean
- retrieval score spread
- source consistency

Because they all collapse those signals into one scalar `u_epi`, calibrating thresholds can
make different adapters produce the same actions. In the latest diagnostic pass, several
calibrated stochastic sources matched `logit_mi` on all `50/50` actions. This indicates limited
independent signal, not a robust stochastic contribution.

## Initial Diagnostic Snapshot

Current diagnostic artifact:

- `evaluation_results/auto_eval/finreg_stochastic_gate_diagnostics_current50_seed7.json`

Input dump:

- `evaluation_results/auto_eval/finreg_shadow_dump_current50_seed7_details.jsonl`

Wide replay grid:

- `evaluation_results/auto_eval/finreg_stochastic_replay_current50_seed7_wide_calibration.json`

Key observation from the first diagnostic pass:

| Source | Calibrated epi threshold | Non-answer rate | Action agreement with `logit_mi` |
| --- | ---: | ---: | ---: |
| `logit_mi` | `0.05` | `48%` | `50/50` |
| `stochastic_ou` | `0.10` | `48%` | `50/50` |
| `stochastic_langevin` | `0.25` | `48%` | `50/50` |
| `stochastic_sghmc` | `0.125` | `48%` | `50/50` |
| `stochastic_sgbd` | `0.125` | `48%` | `50/50` |
| `stochastic_prox_langevin` | `0.125` | `48%` | `50/50` |
| `stochastic_wright_fisher` | `0.45` | `48%` | `50/50` |
| `stochastic_mirror_langevin` | `0.125` | `54%` | `47/50` |

The only source with meaningful action disagreement in this pass is
`stochastic_mirror_langevin`, and it changes three hard/conflict questions from `answer` to
`retrieve_more`. This is worth inspecting, but it is not enough to claim a general stochastic
improvement.

## Evaluation Rule Going Forward

Every stochastic gate comparison must report:

- source-specific score distribution, including min/p25/p50/p75/max
- source-specific calibrated threshold
- action rates: `answer`, `retrieve_more`, `abstain`
- non-answer rate split into `retrieve_more` and `abstain`
- answer contradiction / hallucination proxy among answered questions
- expected-action bucket behavior from FinReg labels
- action agreement/disagreement with the `logit_mi` baseline

If a new method only changes the threshold scale but produces the same actions as `logit_mi`,
it should not be claimed as a meaningful stochastic improvement.

## 2026-05-28 Retest: Stochastic Adapters Are Still Research Candidates

A new replay was run after the LFM2/no-type gating checks to avoid prematurely discarding the
stochastic adapters.

Artifacts:

- LFM2 no-type replay:
  `evaluation_results/auto_eval/lfm2_stochastic_replay_current50_notype_seed7_wide.json`
- LFM2 overlap report:
  `evaluation_results/auto_eval/lfm2_stochastic_adapter_overlap_current50_notype_seed7.json`
- Refreshed older shadow replay:
  `evaluation_results/auto_eval/finreg_stochastic_replay_current50_seed7_wide_refresh.json`

The LFM2 no-type replay used:

- input details:
  `evaluation_results/auto_eval/lfm2_guarded_v3_retry_seed7_current50_notype_details.jsonl`
- labels:
  `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50_labels.jsonl`
- production-like question file without `type` metadata:
  `evaluation_results/auto_eval/tmp/questions_finreg_current50_no_type.jsonl`
- policy: `epi_coupled_v2`
- uncertainty formula: `v2_conflict_aware`
- wide epistemic threshold grid from `0.005` to `0.50`
- aleatoric threshold grid from `0.25` to `0.50`

Best calibrated operating points on the LFM2 no-type details:

| Source | Operating score | Actions | Thresholds | Agreement with best `logit_mi` |
| --- | ---: | --- | --- | ---: |
| `stochastic_sgbd` | `0.7702` | `answer=16`, `retrieve_more=34` | `epi=0.10`, `ale=0.40` | `44/50` |
| `stochastic_sghmc` | `0.7516` | `answer=33`, `retrieve_more=17` | `epi=0.125`, `ale=0.45` | `37/50` |
| `stochastic_ou` | `0.7491` | `answer=24`, `retrieve_more=26` | `epi=0.075`, `ale=0.40` | `46/50` |
| `logit_mi` | `0.7424` | `answer=20`, `retrieve_more=30` | `epi=0.03`, `ale=0.40` | `50/50` |
| `stochastic_langevin` | `0.7421` | `answer=25`, `retrieve_more=25` | `epi=0.25`, `ale=0.40` | `45/50` |
| `stochastic_mirror_langevin` | `0.7421` | `answer=25`, `retrieve_more=25` | `epi=0.10`, `ale=0.40` | `45/50` |
| `stochastic_wright_fisher` | `0.7421` | `answer=25`, `retrieve_more=25` | `epi=0.45`, `ale=0.40` | `45/50` |
| `stochastic_prox_langevin` | `0.7421` | `answer=25`, `retrieve_more=25` | `epi=0.125`, `ale=0.40` | `45/50` |

Best calibrated operating points on the older 50Q shadow dump:

| Source | Operating score | Actions | Thresholds | Agreement with best `logit_mi` |
| --- | ---: | --- | --- | ---: |
| `stochastic_mirror_langevin` | `0.7245` | `answer=23`, `retrieve_more=27` | `epi=0.125`, `ale=0.40` | `24/50` |
| `stochastic_sghmc` | `0.7221` | `answer=36`, `retrieve_more=14` | `epi=0.125`, `ale=0.45` | `37/50` |
| `logit_mi` | `0.7067` | `answer=37`, `retrieve_more=13` | `epi=0.03`, `ale=0.50` | `50/50` |
| `stochastic_wright_fisher` | `0.7053` | `answer=26`, `retrieve_more=24` | `epi=0.45`, `ale=0.40` | `27/50` |
| `stochastic_sgbd` | `0.7053` | `answer=26`, `retrieve_more=24` | `epi=0.125`, `ale=0.40` | `27/50` |
| `stochastic_prox_langevin` | `0.7053` | `answer=26`, `retrieve_more=24` | `epi=0.125`, `ale=0.40` | `27/50` |
| `stochastic_ou` | `0.7053` | `answer=26`, `retrieve_more=24` | `epi=0.10`, `ale=0.40` | `27/50` |
| `stochastic_langevin` | `0.7053` | `answer=26`, `retrieve_more=24` | `epi=0.25`, `ale=0.40` | `27/50` |

Interpretation:

- The earlier "everything collapses to `logit_mi`" conclusion was too broad for the current LFM2
  setup. Some stochastic adapters do produce different calibrated action sets.
- `stochastic_sgbd` is the best LFM2/no-type replay candidate, while
  `stochastic_mirror_langevin` was best on the older shadow dump. This means the signal is not yet
  stable enough to promote directly.
- `stochastic_wright_fisher` and `stochastic_langevin` are not dead, but in this retest they are
  not the strongest LFM2 candidates. They remain useful ablation candidates.
- These results are label-aware replay results, not production results. The labels are used only to
  select operating points and evaluate utility; runtime policy must not use those labels.

Next validation before any promotion:

1. Run the same wide replay on additional seeds and, if available, a holdout dump.
2. Shortlist `stochastic_sgbd`, `stochastic_sghmc`, and `stochastic_mirror_langevin`; keep
   `stochastic_wright_fisher` as a lower-priority ablation.
3. If the shortlist is stable, move the selected stochastic epistemic adapter into shared runtime
   code instead of keeping it only in `scripts/eval_grounding_proxy.py`.
4. Compare the selected adapter against `logit_mi + guarded_v3` in applied no-type runs before
   changing the default production path.

## Next Technical Direction

Keep `logit_mi` as the current baseline. Move stochastic gating from scalar proxy formulas
toward actual stochastic evidence or detector variation:

1. Evidence subset sampling
   - sample different retrieved-context subsets from the same top-k pool
   - recompute detector/gate statistics per subset
   - estimate whether the action is stable under evidence perturbation

2. Retrieval perturbation
   - perturb top-k, reranker order, or source-family balance
   - measure whether the answer decision survives plausible retrieval variation

3. Detector logit perturbation or lightweight posterior approximation
   - perturb detector logits or use sampled detector heads/checkpoints
   - separate detector uncertainty from retrieval/evidence uncertainty

4. Sequential risk-budget gating
   - make `retrieve_more` a budgeted intermediate action
   - answer only when additional retrieval does not materially change risk
   - abstain when risk remains high after the evidence budget is exhausted

## Evidence Subset Sampling Prototype

Prototype script:

- `scripts/eval_finreg_evidence_sampling_shadow.py`

Pipeline shape:

1. run the normal RAG query
2. keep the `pre_gating_answer`
3. keep the final retrieved/reranked context
4. build multiple evidence subsets from that context
5. run the hallucination/support detector on each subset
6. recompute gate stats/action for each subset
7. report action distribution and stability

Initial smoke artifacts:

- `evaluation_results/auto_eval/finreg_evidence_sampling_shadow_smoke_seed7_limit1.json`
- `evaluation_results/auto_eval/finreg_evidence_sampling_shadow_pilot_seed7_limit5.json`
- `evaluation_results/auto_eval/finreg_evidence_sampling_shadow_current50_seed7.json`
- `evaluation_results/auto_eval/finreg_evidence_sampling_shadow_current50_seed11.json`
- `evaluation_results/auto_eval/finreg_evidence_sampling_shadow_current50_seed19.json`

Pilot read (`limit=5`, seed `7`, subset size `4`, `4` subsets/question):

- baseline actions: `none=2`, `retrieve_more=3`
- subset actions: `none=9`, `retrieve_more=11`
- mean subset action instability: `0.25`
- mean baseline action stability under subset perturbation: `0.65`

This is the first useful stochastic signal in this track: the method now measures whether the
same candidate answer remains safe under evidence perturbation. The first five questions are
sanity questions, so any high `retrieve_more` rate there should be inspected carefully; it can
mean either weak subset evidence or an over-sensitive detector/gate.

Full 50Q pass (`seed=7`, subset size `4`, `4` subsets/question):

- baseline actions: `none=28`, `retrieve_more=19`, `abstain=3`
- subset actions: `none=100`, `retrieve_more=88`
- mean subset action instability: `0.277`
- mean baseline action stability under subset perturbation: `0.628`

By question type:

| Type | Count | Baseline actions | Subset actions | Mean instability | Mean baseline stability |
| --- | ---: | --- | --- | ---: | ---: |
| `sanity` | `10` | `none=6`, `retrieve_more=4` | `none=24`, `retrieve_more=16` | `0.250` | `0.700` |
| `conflict` | `40` | `none=22`, `retrieve_more=15`, `abstain=3` | `none=76`, `retrieve_more=72` | `0.284` | `0.608` |

Interpretation:

- Evidence subset sampling adds information beyond the single final context because many
  questions change action under subset perturbation.
- The signal is stronger on conflict questions than sanity questions, but sanity instability is
  still non-trivial. This means the method should not be attached directly as a hard gate yet.
- The next policy should treat subset instability as a weak/secondary risk signal unless paired
  with strong detector risk or hard/conflict query metadata.
- `abstain` cases need special handling because final production answers can be replaced by the
  abstain message; the script currently skips subset checks when a usable pre-gating candidate is
  unavailable.

Evidence-sampling policy replay:

- script: `scripts/replay_finreg_evidence_sampling_policy.py`
- output: `evaluation_results/auto_eval/finreg_evidence_sampling_policy_replay_current50_seed7.json`

Best current shadow candidate: `guarded_v3`.

`guarded_v3` is a first-pass sequential policy:

- preserve answer behavior on sanity questions unless subset risk is consistently high
- use subset non-answer/instability as `retrieve_more` evidence on conflict questions
- do not increase abstain on the first pass; stable high risk should request more evidence first
- leave final abstain to a later evidence-budget stage

Seed `7` 50Q replay:

| Policy | Actions | Balanced label-aware utility | Changes from baseline |
| --- | --- | ---: | ---: |
| baseline | `none=28`, `retrieve_more=19`, `abstain=3` | `0.698` | `0` |
| subset majority | `none=34`, `retrieve_more=13`, `abstain=3` | `0.721` | `14` |
| guarded_v1 | `none=19`, `retrieve_more=25`, `abstain=6` | `0.829` | `15` |
| guarded_v2 | `none=19`, `retrieve_more=25`, `abstain=6` | `0.829` | `15` |
| guarded_v3 | `none=19`, `retrieve_more=28`, `abstain=3` | `0.843` | `13` |

Three-seed replay summary (`seed=7/11/19`, 50Q each):

| Policy | Mean utility | Utility sd | Mean actions | Mean changes | Mean answer accuracy | Mean retrieve-more accuracy |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| baseline | `0.665` | `0.050` | `none=25.3`, `retrieve_more=21.3`, `abstain=3.3` | `0.0` | `0.467` | `0.408` |
| subset majority | `0.679` | `0.078` | `none=32.3`, `retrieve_more=14.3`, `abstain=3.3` | `15.7` | `0.667` | `0.283` |
| guarded_v1 | `0.801` | `0.030` | `none=19.3`, `retrieve_more=25.7`, `abstain=5.0` | `15.3` | `0.733` | `0.583` |
| guarded_v2 | `0.803` | `0.028` | `none=19.3`, `retrieve_more=26.0`, `abstain=4.7` | `15.0` | `0.733` | `0.592` |
| guarded_v3 | `0.809` | `0.034` | `none=19.3`, `retrieve_more=27.3`, `abstain=3.3` | `14.0` | `0.733` | `0.625` |

Interpretation after three seeds:

- `guarded_v3` remains the best current shadow candidate by mean balanced utility.
- `subset_majority` is not safe as a standalone policy. It improves answerable-question behavior
  but loses too much retrieve-more behavior on conflict questions, especially in seed `19`.
- `guarded_v3` is preferable to v1/v2 because it improves conflict retrieve-more behavior while
  avoiding a first-pass abstain increase.

## Applied Evidence-Sampling Validation

The first applied evidence-sampling runs exposed a local generation issue: Qwen
`Qwen/Qwen2.5-1.5B-Instruct` produced degenerate exclamation-mark outputs on this RTX 2070
when loaded with `dtype: float16`. Those runs must not be used for answer-quality conclusions.

The local safe evaluation config now uses:

- Qwen `1.5B` on CUDA with `dtype: float32`
- embeddings, EBCAR reranker, and detector on CPU
- left-side tokenizer truncation to preserve the generation prompt
- no retry budget for first-pass policy comparison

Validated artifacts:

- `config/gating_finreg_ebcar_logit_mi_sc009_shadowfast_rtx2070_safe.yaml`
- `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_applied_float32_current50_seed{7,11,13}.json`
- `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_applied_float32_guarded_v4_current50_seed{7,11,13}.json`
- `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_applied_float32_guarded_v4_lite_current50_seed{7,11,13}.json`

Three-seed applied summary (`seed=7/11/13`, 50Q each):

| Policy | Mean non-answer rate | Aggregated actions | Detector failures | Mean answered contradiction | Mean non-answer contradiction |
| --- | ---: | --- | ---: | ---: | ---: |
| `guarded_v3` | `0.627` | `none=56`, `retrieve_more=76`, `abstain=18` | `0` | `0.123` | `0.197` |
| `guarded_v4` | `0.740` | `none=39`, `retrieve_more=93`, `abstain=18` | `0` | `0.211` | `0.153` |
| `guarded_v4_lite` | `0.693` | `none=46`, `retrieve_more=86`, `abstain=18` | `0` | `0.201` | `0.154` |

Interpretation:

- `guarded_v4` is the strongest conflict-avoidance policy in this pass, but it is materially
  more conservative than `guarded_v3`.
- `guarded_v4_lite` is the current balanced applied candidate. It preserves the `v4` sanity
  guard, reduces unsafe conflict answering versus `v3`, and avoids part of `v4`'s coverage loss.
- The remaining weakness is that `retrieve_more` is still final non-answer in the no-retry eval.
  The next validation stage should test a small evidence-budget retry path before deciding final
  abstain behavior.

Small retry pilot:

- config: `config/gating_finreg_ebcar_logit_mi_sc009_shadowfast_rtx2070_safe_evidence_retry.yaml`
- output: `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_retry_k8_float32_guarded_v4_lite_seed7_limit15.json`
- limit: `15` questions, `seed=7`, `max_retries=1`, retry `k=8`

Pilot result:

- actions: `none=6`, `retrieve_more=8`, `abstain=1`
- non-answer rate: `0.600`
- detector failures: `0`
- retry attempts: `11/15`
- final `k=8`: `11/15`

Interpretation:

- The retry ordering now works: evidence-sampling `retrieve_more` can trigger a second retrieval
  pass instead of being swallowed by the old gate order.
- Some questions become answerable after the second pass, but many remain `retrieve_more`. This
  means the retry path is useful, but it should be evaluated as an evidence-budget mechanism, not
  assumed to be an automatic coverage fix.
- The next run should compare no-retry versus one-retry on the same 50Q set and report action
  transitions, especially `retrieve_more -> none`, `retrieve_more -> retrieve_more`, and
  `retrieve_more -> abstain`.

Full one-retry 50Q validation:

- config: `config/gating_finreg_ebcar_logit_mi_sc009_shadowfast_rtx2070_safe_evidence_retry.yaml`
- output: `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_retry_ordered_k8_float32_current50_seed7.json`
- details: `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_retry_ordered_k8_float32_current50_seed7_details.jsonl`
- policy: `guarded_v3`
- limit: `50` questions, `seed=7`, retry `k=5 -> 8`

Result:

- actions: `none=29`, `retrieve_more=14`, `abstain=7`
- non-answer rate: `0.420`
- retry attempts: `25/50`
- final `k=8`: `25/50`
- detector failures: `0`

Compared with the matching no-retry `guarded_v3` seed `7` run:

| Transition | Count |
| --- | ---: |
| `none -> none` | `16` |
| `retrieve_more -> retrieve_more` | `12` |
| `retrieve_more -> none` | `13` |
| `retrieve_more -> abstain` | `1` |
| `none -> retrieve_more` | `2` |
| `abstain -> abstain` | `6` |

Interpretation:

- The one-retry path materially reduces final `retrieve_more` outcomes on this seed:
  `13/26` no-retry `retrieve_more` decisions become answered after the evidence budget.
- The retry path is not only a relaxation step: `2` no-retry answers become `retrieve_more`, and
  `1` no-retry `retrieve_more` becomes final `abstain`.
- The local Qwen CUDA stack still hit intermittent `illegal memory access` failures. The details
  file was recoverable because per-question rows are flushed, and `--resume` completed the run.
`guarded_v3` is now wired as an optional evidence-sampling shadow policy in the normal RAG
pipeline. It is controlled through `gating.evidence_sampling` or the eval flag
`--evidence-sampling-shadow`.

Runtime behavior:

- default behavior is unchanged
- `shadow_only: true` reports `result["evidence_sampling_gate"]` but does not change the answer
  or the production `gating.action`
- `shadow_only: false` can be used later to apply the policy action, but this should only happen
  after normal-harness validation

Smoke command:

```bash
PYTHONPATH=. HF_HOME=./models/llm TRANSFORMERS_CACHE=./models/llm \
  venv312/bin/python scripts/eval_grounding_proxy.py \
  --config gating_finreg_ebcar_logit_mi_sc009_shadowfast \
  --questions data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl \
  --limit 1 --seed 7 \
  --evidence-sampling-shadow \
  --evidence-sampling-policy guarded_v3
```

Smoke result: production gate stayed at `none`, while evidence sampling shadow proposed
`retrieve_more`. This confirms the shadow path reports a useful alternate decision without
changing production behavior.

Normal-harness 50Q seed `7` artifact:

- `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_shadow_current50_seed7.json`
- `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_shadow_current50_seed7_details.jsonl`
- `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_shadow_current50_seed11.json`
- `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_shadow_current50_seed11_details.jsonl`
- `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_shadow_current50_seed19.json`
- `evaluation_results/auto_eval/finreg_grounding_proxy_evidence_shadow_current50_seed19_details.jsonl`

Normal-harness seed `7` result:

| Channel | Actions |
| --- | --- |
| production gate | `none=27`, `retrieve_more=20`, `abstain=3` |
| evidence shadow | `none=21`, `retrieve_more=26`, missing `3` |

Notes:

- The shadow policy was evaluated on `47/50` questions. The missing `3` were production/model
  abstain cases where no usable candidate answer was passed into evidence sampling.
- `guarded_v3` disagreed with the production gate on `14/47` evaluated questions.
- The disagreement pattern is directionally useful: it relaxes several sanity questions from
  `retrieve_more` to `none`, and tightens several conflict questions from `none` to
  `retrieve_more`.
- The run required resume support because the local CUDA generation stack hit a transient
  `misaligned address` error near the end. `scripts/eval_grounding_proxy.py --resume` now
  resumes from the per-question details file.

Normal-harness three-seed summary:

| Seed | Production actions | Evidence-shadow actions | Shadow evaluated | Shadow disagreements |
| ---: | --- | --- | ---: | ---: |
| `7` | `none=27`, `retrieve_more=20`, `abstain=3` | `none=21`, `retrieve_more=26` | `47/50` | `14` |
| `11` | `none=22`, `retrieve_more=25`, `abstain=3` | `none=21`, `retrieve_more=26` | `47/50` | `15` |
| `19` | `none=15`, `retrieve_more=31`, `abstain=4` | `none=17`, `retrieve_more=29` | `46/50` | `12` |

Average across seeds:

- production actions: `none=21.3`, `retrieve_more=25.3`, `abstain=3.3`
- evidence-shadow actions: `none=19.7`, `retrieve_more=27.0`
- evidence-shadow evaluated: `46.7/50`
- shadow disagreements with production: `13.7/50`
- useful disagreement pattern: about `2.7` sanity questions relaxed from `retrieve_more` to
  `none`, and about `7.7` conflict questions tightened from `none` to `retrieve_more`

Interpretation:

- The normal-harness result is consistent with the standalone replay result: `guarded_v3`
  produces a stable, non-identical signal.
- The shadow policy is not simply more conservative everywhere. It both relaxes answerable
  sanity cases and tightens conflict cases.
- The local CUDA instability is now operationally manageable with `--resume`, but it should be
  called out as an environment issue when reporting runtime experiments.

## Acceptance Criteria

A stochastic gate candidate is only useful if it improves at least one of these without causing
collapse elsewhere:

- lower contradiction/hallucination proxy among answered questions
- better retrieve/abstain decisions on hard or ambiguous FinReg questions
- less unnecessary non-answer on sanity/answerable questions
- meaningful action disagreement from `logit_mi` that is explainable at question level
- stable behavior across seeds, not just one 50Q run
