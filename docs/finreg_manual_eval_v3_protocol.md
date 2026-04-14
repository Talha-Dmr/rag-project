# Finreg Manual Eval V3 Protocol

This note defines the annotation protocol for the `manualeval_v3` run.

Primary artifacts:

- `evaluation_results/finreg_detector_manualeval_v3/detector_comparison_report.md`
- `evaluation_results/finreg_detector_manualeval_v3/per_detector_summary.json`
- `evaluation_results/finreg_detector_manualeval_v3/stratified_eval_subset.jsonl`
- `evaluation_results/finreg_detector_manualeval_v3/manual_annotation_sheet.csv`

Detector roles:

- `fever_local`: reference detector
- `targeted_v2`: candidate detector

Bucket intent:

- `high_entailment`: low-risk answers that the detector sees as well-supported
- `uncertain_zone`: answers where the detector is not clearly confident
- `high_contradiction_signal`: answers with elevated contradiction signal even when contradiction is not the dominant class
- `suspicious_cases`: strongest detector alerts, usually high conflict or high hallucination-topk

Recommended annotation order:

1. Annotate all `high_contradiction_signal` rows first.
2. Annotate all `suspicious_cases` rows second.
3. Annotate a representative sample from `uncertain_zone`.
4. Use `high_entailment` as the control group.

## Label Rules

Use exactly one `label` value:

- `supported`
  - The answer is materially grounded in the retrieved context.
  - Minor wording drift is acceptable if the regulatory claim stays intact.
- `unsupported`
  - The answer introduces claims that are not supported by the retrieved context.
  - Use this when the answer overreaches without directly contradicting the evidence.
- `contradicted`
  - The answer conflicts with the retrieved context.
  - Use this when the answer states the opposite of what the evidence says or attributes a stance to the wrong source.
- `partial`
  - Part of the answer is supported, but important content is missing, blended, or overgeneralized.
- `ambiguous`
  - The evidence itself is too weak, mixed, or off-topic to judge the answer cleanly.

## Error Type Rules

Use one dominant `error_type` when `label` is not `supported`:

- `fabricated_fact`
  - The answer invents a fact, source stance, or operational recommendation.
- `wrong_number_or_threshold`
  - Numeric thresholds, timing, counts, or trigger points are wrong.
- `cross_document_conflict`
  - The answer blends positions from different documents or supervisors incorrectly.
- `outdated_regulation`
  - The answer relies on stale guidance relative to the retrieved context.
- `misinterpretation`
  - The answer misreads what the retrieved context actually implies.
- `incomplete_reasoning`
  - The answer omits key caveats or gives a materially incomplete synthesis.

## Notes Guidance

Use the `notes` column for short, audit-friendly comments:

- Mention the exact failure mode in one sentence.
- If the problem is source mixing, name the sources.
- If the answer should have abstained, say that explicitly.

Preferred note patterns:

- `Should abstain: retrieved evidence does not resolve the cross-source disagreement.`
- `Source mix-up: ECB climate-risk stance inserted into intraday liquidity answer.`
- `Unsupported synthesis: answer generalizes beyond the retrieved chunks.`

## Generation Contamination

Some rows may contain obvious non-finreg text contamination in `generated_answer`.
Examples include unrelated `Project Manager` job-description content.

When this happens:

- set `label=unsupported` unless the contamination directly contradicts the evidence, in which case use `contradicted`
- set `error_type=fabricated_fact`
- write `generation_contamination` in `notes`, plus a short description

Example:

- `generation_contamination: unrelated project-manager content appended to answer`

## Decision Policy By Bucket

`high_contradiction_signal`

- Ask whether the answer should have been flagged as risky even if the detector still predicted `entailment`.
- Be strict about source swaps, blended reasoning, and unjustified synthesis.

`suspicious_cases`

- Treat these as the highest-priority audit cases.
- Confirm whether the detector alert is justified or whether this is a false positive.

`uncertain_zone`

- Focus on whether the answer should have abstained or expressed uncertainty more clearly.

`high_entailment`

- Use these to estimate precision on likely-safe answers.
- If one of these is actually bad, note it clearly because it is a more valuable failure.

## Expected Use In V3

The `manualeval_v3` subset is the recommended annotation set because it finally separates:

- `high_contradiction_signal`
- `suspicious_cases`
- `uncertain_zone`
- `high_entailment`

This makes `targeted_v2` meaningfully comparable against the `fever_local` reference baseline.
