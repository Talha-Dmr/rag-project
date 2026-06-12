# FinReg Final Targeted Benchmark Audit

## Purpose

This audit fixes benchmark-quality issues in the final targeted FinReg evaluation set. The goal is to make the benchmark defensible for the project report: evidence labels must match the cited passage, expected answer points must be atomic propositions, and boilerplate page text must not be used as gold evidence.

## Fixes Applied

1. Strengthened evidence support validation.
   The benchmark builder now checks each `gold_evidence.supports` label against the actual evidence text with stricter concept-specific rules. This prevents broad false positives such as treating "Financial Stability Board" as evidence for board oversight.

2. Removed weak expected-point labels.
   Short tags such as `board oversight`, `not supported`, and `different authority` were replaced with atomic expected propositions, for example: `The cited evidence does not establish that manual senior sign off is sufficient without supporting controls.`

3. Added boilerplate evidence checks.
   Evidence containing navigation/menu/cookie/footer style text is now flagged and rejected by validation.

4. Added duplicate expected-point checks.
   Rows with repeated expected points now fail validation.

5. Added support-label consistency checks.
   Validation now fails if a support label is not actually supported by its evidence passage or if a gold evidence item has no support labels.

## Validation Result

The regenerated `full_rag_questions_final_targeted160.jsonl` passed the strengthened validation:

| Check | Result |
|---|---:|
| Total questions | 160 |
| Duplicate queries | 0 |
| Near duplicate examples | 0 |
| Missing evidence text rows | 0 |
| Forbidden claims present in corpus | 0 |
| Duplicate expected-point rows | 0 |
| Short expected-point rows | 0 |
| Boilerplate evidence rows | 0 |
| Unsupported support labels | 0 |
| Empty support evidence rows | 0 |
| Quality issue count | 0 |

## Audited Benchmark Results

| System | Expected Behavior | Point Coverage | Answer Rate | Abstain Rate | Forbidden Claim Hit | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| Baseline RAG | 80.00% | 24.34% | 100.00% | 0.00% | 0.00% | 5.32s |
| RAG + Detector | 92.50% | 13.44% | 46.25% | 53.75% | 0.00% | 5.48s |
| RAG + Detector + Stochastic | 93.12% | 13.28% | 45.62% | 54.37% | 0.00% | 5.88s |

## Interpretation

The audited benchmark is harder and more defensible than the earlier generated set. Baseline RAG answers every question and reaches 80.00% expected behavior, but it still fails more often on unsupported detail, cross-authority transfer, and cautious-policy cases. Adding the detector raises expected behavior to 92.50% by abstaining on high-risk answers. Adding stochastic evidence sampling gives a smaller additional gain to 93.12%, mainly by preserving detector safety while catching one additional low-evidence/cross-evidence failure.

Point coverage should be treated as a secondary metric after this audit because expected points are now longer atomic propositions rather than short keyword labels. This makes the metric stricter and less directly comparable to earlier benchmark runs.
