# FinReg Evaluation Benchmarks

This folder contains benchmark inputs for the FinReg RAG hallucination-gating
project.

## Current Final Benchmark

`full_rag_questions_final_targeted160.jsonl` is the current report benchmark.
It contains 160 targeted hallucination stress-test questions built from observed
baseline RAG failure modes.

Current distribution:

- 72 `cross_source_nuanced`
- 40 `low_evidence_policy`
- 32 `false_premise`
- 16 `factual_supported`

Use this file for final system comparisons across:

- baseline RAG,
- RAG + detector,
- RAG + detector + stochastic evidence sampling,
- API/model swaps that keep the same retrieval/detector/gating setup.

The matching validation summary is:

`full_rag_questions_final_targeted160_validation.json`

The matching report docs are:

- `docs/finreg_final_targeted_benchmark_audit.md`
- `docs/finreg_final_targeted_benchmark_report.md`

## What The Final Benchmark Tests

The benchmark is not a random general QA set. It pressures the model to:

- transfer evidence from one regulator or document to another when that
  transfer is unsupported,
- invent exact operational requirements from broad topical evidence,
- accept false premises embedded in the question, or
- answer normally when the supporting evidence is direct.

The main metric is expected behavior match: the system should answer when
evidence is sufficient, qualify or abstain when evidence is weak, and reject
false premises.

## Older Full-RAG Benchmarks

- `full_rag_questions.jsonl`: earlier canonical 160-question set.
- `final_holdout_80_questions.jsonl`: earlier fixed 80-question holdout.
- `full_rag_questions_hard.jsonl`: earlier hard derived benchmark.
- `full_rag_questions_hard_v2.jsonl`: later hard-v2 benchmark and review set.

These are still useful for regression and stress testing, but they should not
override final claims from `full_rag_questions_final_targeted160.jsonl`.

## Controlled Candidate Benchmark

`controlled_candidate_cases.jsonl` isolates the detector. It contains fixed
candidate-answer cases where the system retrieves evidence and evaluates a
given candidate answer instead of generating a fresh full-RAG answer.

Use controlled cases when the question is specifically about detector behavior.
Use the final full-RAG benchmark when the question is about the end-to-end
system.

## Evaluation Notes

The benchmark script reports answer rate, abstain rate, detector risk,
expected-point coverage, forbidden-claim hit rate, latency, and expected
behavior match. Point coverage is useful but secondary for the final audited
set because expected points are longer atomic propositions rather than short
keyword labels.
