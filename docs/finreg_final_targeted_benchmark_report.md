# Final Targeted FinReg Benchmark

## Purpose

This benchmark is a targeted hallucination stress test, not a random general QA benchmark. It was built after analyzing prior baseline RAG failures. The plain RAG system most often failed when a prompt asked it to turn related regulatory evidence into a stronger claim than the source actually supported.

## Benchmark Design

Total questions: 160

| Category | Count | Purpose |
|---|---:|---|
| cross_source_nuanced | 72 | Tests whether the system rejects unsupported transfer of evidence across regulators or documents. |
| low_evidence_policy | 40 | Tests whether the system avoids inventing exact operational details from partial topical evidence. |
| false_premise | 32 | Tests whether the system rejects fabricated regulatory requirements. |
| factual_supported | 16 | Sanity checks that the system can still answer when evidence is directly supported. |

Challenge distribution:

| Challenge type | Count |
|---|---:|
| assertive_cross_authority_transfer | 72 |
| partial_support_misleading_inference | 24 |
| assertive_unsupported_detail_request | 16 |
| fabricated_requirement_acceptance | 32 |
| supported_sanity | 16 |

Validation checks:

| Check | Result |
|---|---|
| total questions | 160 |
| duplicate queries | 0 |
| near duplicate examples | 0 |
| missing evidence text rows | 0 |
| forbidden claims present in corpus | 0 |
| duplicate expected-point rows | 0 |
| short expected-point rows | 0 |
| boilerplate evidence rows | 0 |
| unsupported support labels | 0 |
| empty support evidence rows | 0 |
| quality issue count | 0 |

## Results

| System | Expected Behavior Match | Point Coverage | Answer Rate | Abstain Rate | Forbidden Claim Hit Rate | Mean Latency |
|---|---:|---:|---:|---:|---:|---:|
| Baseline RAG | 80.00% | 24.34% | 100.00% | 0.00% | 0.00% | 5.32s |
| RAG + Detector | 92.50% | 13.44% | 46.25% | 53.75% | 0.00% | 5.48s |
| RAG + Detector + Stochastic | 93.12% | 13.28% | 45.62% | 54.37% | 0.00% | 5.88s |

## Interpretation

Baseline RAG answered every question and reached 80.00% expected behavior. It no longer hit benchmark-defined forbidden claims after the audit, but it still failed more often on unsupported-detail, source-transfer, and cautious-policy prompts.

The detector improved expected behavior to 92.50% by identifying many answers that were not sufficiently supported by the retrieved evidence. The tradeoff is a lower answer rate because the detector abstains on high-risk prompts.

The detector plus stochastic gate produced the best audited expected behavior result at 93.12%. The gain over the deterministic detector is small but positive: it preserves zero forbidden claim hits while catching one additional low-evidence or cross-evidence failure.

Point coverage should be treated as a secondary metric in this audited run. Expected points are now longer atomic propositions rather than short keyword labels, which makes point coverage stricter and less directly comparable to earlier benchmark runs.

## Report Framing

This benchmark should be presented as a targeted stress benchmark derived from observed baseline failure modes. It should not be described as a broad random QA benchmark. The correct claim is:

The proposed detector and stochastic gating layers substantially improve behavior on prompts that pressure the RAG system to over-transfer evidence, accept false premises, or invent unsupported regulatory details.
