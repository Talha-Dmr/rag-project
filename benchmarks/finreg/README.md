# FinReg Real-Life Evaluation Benchmarks

This folder contains report-ready benchmark inputs for the FinReg RAG detector
project.

## Controlled Candidate Benchmark

`controlled_candidate_cases.jsonl` isolates the detector. It contains 80 fixed
candidate-answer cases: 20 topics x 4 labels. Each row contains:

- `query`
- `candidate_answer`
- `expected`: `included` or `not_included`
- `label_detail`: `included`, `not_included`, `contradicted`, or `partial`

The evaluation retrieves evidence, checks the fixed candidate answer, and
computes detector metrics such as not-included recall and false include rate.

## Full RAG Benchmark

`full_rag_questions.jsonl` evaluates the end-to-end system. It contains 160
questions across 4 question types (`factual_supported`, `false_premise`,
`multi_source_nuanced`, and `low_evidence_policy`). Each row contains a question,
manual review guidance, expected answer points, and forbidden claims. The system
retrieves evidence, generates an answer, runs detector/gating, and exports a
manual review sheet.

This 160-question set is the canonical regression benchmark. It should remain
stable so detector-only, detector+stochastic, and future model/API changes can be
compared against the same target.

## Full RAG Hard Benchmark

`full_rag_questions_hard.jsonl` is a harder derived benchmark built from the
canonical 160Q set by `scripts/build_finreg_hard_benchmark.py`. It keeps the
same row schema and adds `source_id` so each hard row can be traced back to the
canonical question it was derived from.

The hard set is designed to stress the part of the project that matters for
stochastic gating: selective answering under low evidence, partial support,
misattribution, cross-source synthesis, and completeness without invented
regulatory details. Its current distribution is:

- 20 `factual_supported`
- 20 `hard_factual_completeness`
- 30 `false_premise_misattribution`
- 40 `low_evidence_specific_claim`
- 30 `cross_source_conflict`
- 20 `partial_support_overclaim`

Use the canonical 160Q set for final regression claims. Use the hard 160Q set to
develop and pressure-test stochastic gating policies that otherwise look
unimportant because the canonical set is already easy for the baseline.

The benchmark script reports pre-review metrics such as answer rate, abstain
rate, answer-include risk, expected-point coverage, forbidden-claim hit rate,
and expected behavior match rate. These are useful for comparison tables, but
final answer quality should still be manually reviewed because generation
errors can come from retrieval, the LLM, the detector, or the gating policy.

Current benchmark results and reproduction commands are documented in
`docs/finreg_real_life_benchmark_results.md`.
