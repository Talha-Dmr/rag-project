# FinReg Full RAG 160 Coverage and Vector-Stochastic Results

Date: 2026-05-30

## Purpose

This note records the coverage-quality baseline and the follow-up vector-stochastic gating work for the 160-question FinReg full RAG benchmark.

The main project configurations are:

- `config/gating_finreg_granite33_2b_3090_evidence_guarded_v6_coverage_quality.yaml`
- `config/gating_finreg_granite33_2b_3090_evidence_vector_v3_coverage_quality.yaml`
- LLM: `ibm-granite/granite-3.3-2b-instruct`
- Retrieval: `k=16`, expanded retrieval up to 24 merged candidates
- Query expansion: enabled, up to 3 expanded queries
- Reranker: cross-encoder, `top_k=8` for guarded v6 and `top_k=12` for vector v3
- Detector: ModernBERT answer-include detector
- Answer quality rewrite: enabled, one rewrite max
- Vector-stochastic evidence sampling: enabled in vector v3, 5 subsets of 4 evidence chunks
- Generation guard: repetition loop detection plus `no_repeat_ngram_size=4`

## Summary Comparison

| Run | Questions | Expected behavior match | Expected point coverage | Answer rate | Abstain rate | Forbidden claim hit rate | Mean answer include risk | Rewrite rate | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `guarded_v6_pre_vector` | 160 | 0.9187 | 0.6343 | 0.8938 | 0.1062 | 0.0125 | 0.6520 | 0.0000 | 11.85s |
| `vector_v2` | 160 | 0.9187 | 0.6339 | 0.8938 | 0.1062 | 0.0063 | 0.6569 | 0.0000 | 10.65s |
| `guarded_v6_coverage_quality_v2` | 160 | 0.9313 | 0.6884 | 0.9563 | 0.0437 | 0.0063 | 0.7251 | 0.4062 | 22.51s |
| `vector_v3_coverage_quality` | 160 | 0.9313 | 0.6884 | 0.9563 | 0.0437 | 0.0063 | 0.7251 | 0.4062 | 26.02s |

## By Question Type

For `guarded_v6_coverage_quality_v2`:

| Question type | Count | Expected behavior match | Expected point coverage | Abstain rate | Forbidden hits |
|---|---:|---:|---:|---:|---:|
| `factual_supported` | 40 | 0.9000 | 0.6827 | 0.0000 | 0 |
| `false_premise` | 40 | 0.9500 | 0.6683 | 0.1250 | 1 |
| `low_evidence_policy` | 40 | 0.9000 | 0.5808 | 0.0250 | 0 |
| `multi_source_nuanced` | 40 | 0.9750 | 0.8217 | 0.0250 | 0 |

## Targeted Failed-Case Tests

After the full 160-question run, follow-up work stopped running the whole benchmark repeatedly and focused only on failed cases from `guarded_v6_coverage_quality_v2`.

| Targeted run | Questions | Purpose | Fixed cases | Remaining failed cases |
|---|---:|---|---|---|
| `failed11_granite33_2b_3090_vector_v3_low_evidence_abstain` | 11 | Test vector-v3 low-evidence abstain behavior | `fullrag_120`, `fullrag_128`, `fullrag_144` | 8 |
| `failed8_granite33_2b_3090_vector_v3_tuned` | 8 | Retest remaining failed cases after vector threshold and rubric fixes | `fullrag_116`, `fullrag_142`, `fullrag_150` | 5 |
| `failed5_granite33_2b_3090_vector_v3_top12` | 5 | Test wider reranked evidence context | `fullrag_131` | 4 |
| `failed4_granite33_2b_3090_vector_v3_domain_prompt` | 4 | Test domain checklist prompting for remaining factual cases | none | 4 |

The final remaining cases are:

- `fullrag_017`
- `fullrag_025`
- `fullrag_049`
- `fullrag_101`

These four are factual coverage failures. They are not mainly detector/stochastic failures: the system answers, but the answer omits enough expected regulatory dimensions to fail the benchmark coverage rubric.

## Interpretation

The coverage-focused configuration improved both answer coverage and overall expected behavior. The main improvement came from retrieving a broader candidate set, reranking more context, and allowing one answer-quality rewrite when required concepts were missing.

The vector-stochastic policy did not improve the full 160-question aggregate score in its first full run, but targeted failed-case testing showed useful behavior on low-evidence and false-premise failures. The policy is most useful when the detector probability vector remains consistently "not included" across sampled evidence subsets. In those cases, the system can abstain instead of returning a broad but unsupported answer.

The wider `top_k=12` context helped the multi-source stress-testing case that previously degenerated into abstention. The remaining factual failures suggest a separate answer-completeness/retrieval-prompting issue rather than a detector decision issue.

The trade-off is speed. The new run is about twice as slow because each question may perform expanded retrieval, reranking, detector scoring, answer-quality auditing, and sometimes a rewrite. This is acceptable for evaluation, but a production/demo config may need a faster preset.

The generation guard was necessary. During the focus test, Granite occasionally produced repeated malformed text. The new guard detects repetition loops and prevents such output from being returned as a normal answer.

## Artifacts

- Guarded full run: `reports/finreg_real_life_benchmark/fullrag160_granite33_2b_3090_guarded_v6_coverage_quality_v2`
- Vector full run: `reports/finreg_real_life_benchmark/fullrag160_granite33_2b_3090_vector_v3_coverage_quality`
- Targeted failed-11 run: `reports/finreg_real_life_benchmark/failed11_granite33_2b_3090_vector_v3_low_evidence_abstain`
- Targeted failed-8 run: `reports/finreg_real_life_benchmark/failed8_granite33_2b_3090_vector_v3_tuned`
- Targeted failed-5 run: `reports/finreg_real_life_benchmark/failed5_granite33_2b_3090_vector_v3_top12`
- Targeted failed-4 run: `reports/finreg_real_life_benchmark/failed4_granite33_2b_3090_vector_v3_domain_prompt`
