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
| unique queries | 160 / 160 |
| duplicate queries | 0 |
| near-duplicate examples | 0 |
| missing evidence paths | 0 |
| missing evidence text | 0 |
| forbidden claims appearing exactly in corpus | 0 |

## Results

| System | Expected Behavior Match | Answer Rate | Abstain Rate | Forbidden Claim Hit Rate | Mean Point Coverage |
|---|---:|---:|---:|---:|---:|
| Baseline RAG | 83.125% | 100.000% | 0.000% | 0.625% | 49.308% |
| RAG + Detector | 95.625% | 49.375% | 50.625% | 0.625% | 33.411% |
| RAG + Detector + Stochastic | 98.125% | 45.000% | 55.000% | 0.000% | 31.737% |

## Interpretation

Baseline RAG answered every question, which helped coverage but caused overclaiming on source-transfer, false-premise, and unsupported-detail prompts.

The detector improved expected behavior by identifying many answers that were not sufficiently supported by the retrieved evidence. This reduced unsafe answering, but still left one forbidden claim hit.

The detector plus stochastic gate produced the strongest safety result. It reached the highest expected behavior match and eliminated forbidden claim hits. Its tradeoff is a lower answer rate because it abstains more often on risk-heavy prompts.

## Report Framing

This benchmark should be presented as a targeted stress benchmark derived from observed baseline failure modes. It should not be described as a broad random QA benchmark. The correct claim is:

The proposed detector and stochastic gating layers substantially improve behavior on prompts that pressure the RAG system to over-transfer evidence, accept false premises, or invent unsupported regulatory details.

