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

`full_rag_questions.jsonl` evaluates the end-to-end system. It contains 40
questions: 10 topics x 4 question types (`factual_supported`, `false_premise`,
`multi_source_nuanced`, and `low_evidence_policy`). Each row contains a question,
manual review guidance, expected answer points, and forbidden claims. The system
retrieves evidence, generates an answer, runs detector/gating, and exports a
manual review sheet.

The benchmark script reports pre-review metrics such as answer rate, abstain
rate, answer-include risk, expected-point coverage, forbidden-claim hit rate,
and expected behavior match rate. These are useful for comparison tables, but
final answer quality should still be manually reviewed because generation
errors can come from retrieval, the LLM, the detector, or the gating policy.

Current benchmark results and reproduction commands are documented in
`docs/finreg_real_life_benchmark_results.md`.
