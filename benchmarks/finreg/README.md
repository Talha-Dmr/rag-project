# FinReg Real-Life Evaluation Benchmarks

This folder contains small, report-ready benchmark inputs for the FinReg RAG
detector project.

## Controlled Candidate Benchmark

`controlled_candidate_cases.jsonl` isolates the detector. Each row contains:

- `query`
- `candidate_answer`
- `expected`: `supported` or `unsupported`
- `label_detail`: `supported`, `unsupported`, `contradicted`, or `partial`

The evaluation retrieves evidence, checks the fixed candidate answer, and
computes detector metrics such as unsupported recall and false accept rate.

## Full RAG Benchmark

`full_rag_questions.jsonl` evaluates the end-to-end system. Each row contains a
question and manual review guidance. The system retrieves evidence, generates an
answer, runs detector/gating, and exports a manual review sheet.

Full RAG outputs are not automatically scored unless a human annotates the
generated answers, because generation errors can come from retrieval, the LLM,
the detector, or the gating policy.
