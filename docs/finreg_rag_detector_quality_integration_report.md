# FinReg RAG Detector and Answer Quality Integration Report

Date: 2026-05-12

## Scope

This report summarizes the detector/RAG integration work after the FinReg real-life benchmark analysis. The goal was not to retrain the detector per test question. The goal was to identify why the trained detector had little visible effect in full RAG runs, then add general pipeline mechanisms that make detector output useful in realistic answers.

## Main Finding

The detector model was not the main bottleneck in many full-RAG failures.

The detector is trained to decide whether a generated answer is included/supported by retrieved evidence. In full RAG, many failures were different: the answer was often not contradicted and not obviously fabricated, but it was incomplete, overly broad, or based on weak retrieval coverage. In those cases, an answer-include detector can correctly avoid flagging a hallucination while the end-to-end benchmark still marks the answer as incomplete.

Practical implication:

- Detector helps with unsafe or not-included claims.
- Retrieval and answer completeness controls are needed for partial answers.
- A separate answer-quality audit is useful before the detector gate.

## Implemented Changes

### 1. Answer Quality Audit

New module:

- `src/rag/answer_quality.py`

It adds a deterministic audit layer that estimates whether an answer covers the important concepts implied by the question and retrieved context.

Main outputs:

- `required_concepts`
- `hit_concepts`
- `missing_concepts`
- `answer_completeness_score`
- `answer_completeness_risk`
- `asks_specific_unsupported`
- extracted answer claims

This is not a replacement for the detector. It catches a different problem: incomplete or evasive answers that are still not hallucinations.

### 2. Quality-Guided Rewrite

Updated:

- `src/rag/rag_pipeline.py`
- `src/core/base_classes.py`
- `src/rag/llm_wrapper.py`
- `src/rag/openrouter_llm.py`

The RAG flow now supports:

1. Retrieve context.
2. Generate an initial answer.
3. Audit answer completeness.
4. If the answer is incomplete, rewrite once with targeted feedback.
5. Run the answer-include detector on the final answer.
6. Apply gating/abstain logic.

This directly addresses the observed issue where the detector was being asked to solve incomplete-answer problems that were outside its original scope.

### 3. Specific Unsupported Requirement Policy

The prompt now explicitly handles questions such as:

- exact number not established
- deadline not established
- approval requirement not established
- portal/template/threshold not established

Expected behavior:

- State clearly that the specific item is not established by the retrieved context.
- Do not replace the answer with a broad regulatory discussion.
- Do not invent exact values.

This was added after seeing low-evidence policy questions where the system answered generally instead of marking the exact missing requirement.

### 4. Stable Local Runtime Configuration

New configs:

- `config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank_quality.yaml`
- `config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank_stochastic_quality.yaml`

The stable quality config keeps Qwen on CUDA but moves support models to CPU:

- Qwen LLM: CUDA
- detector: CPU
- embedder: CPU
- cross-encoder reranker: CPU

Reason:

Long multi-question runs on the RTX 3090 hit intermittent CUDA illegal-memory errors when Qwen generation, ModernBERT detection, and reranking all shared the CUDA process. The stable config is slower but completed the 40-question full-RAG run.

LLM stability options added:

- `attn_implementation: eager`
- `use_cache: false`
- `clear_cuda_cache_after_generate: true`
- configurable `max_prompt_tokens`

### 5. Full-RAG Benchmark Reporting

Updated:

- `scripts/run_finreg_real_life_benchmark.py`

New metrics/report fields:

- `mean_answer_completeness_score`
- `answer_quality_rewrite_count`
- `answer_quality_rewrite_rate`
- `answer_quality_missing_concepts`
- partial result files during full-RAG runs

Partial output files are useful because long local LLM benchmarks may be interrupted:

- `per_question.partial.jsonl`
- `summary.partial.json`

### 6. Query Expansion Prototype

General FinReg query expansion was added in `src/rag/rag_pipeline.py`.

It expands retrieval queries for broad regulatory topics such as:

- stress testing / SREP
- ICT and cyber risk
- outsourcing / cloud
- climate / ESG
- model risk
- intraday liquidity
- risk culture / remuneration

Important note:

This was added after analyzing benchmark failures, so it should be treated as a validation-driven improvement. For final project claims, evaluate it on a fresh holdout test set that was not used while developing the feature.

## Completed Test Result

Completed full-RAG run:

- `reports/finreg_real_life_benchmark/fullrag40_quality_detector_section_rerank_stable_gpu`

Summary:

| Metric | Value |
| --- | ---: |
| Total questions | 40 |
| Expected behavior match rate | 0.85 |
| Abstain rate | 0.25 |
| Answer rate | 0.75 |
| Detector run rate | 0.80 |
| Mean expected point coverage | 0.498 |
| Forbidden claim hit rate | 0.00 |
| Mean answer completeness score | 0.511 |
| Answer quality rewrite rate | 0.375 |

Interpretation:

- The system avoided forbidden claims in this run.
- The detector still matters for unsafe/not-included answers.
- The remaining failures are mostly coverage/retrieval/completeness issues, not pure contradiction issues.
- The stable runtime config completed the 40-question run without CUDA crash.

## What Was Not Done

- The detector model was not retrained during this round.
- No per-question fine-tuning was performed.
- The latest policy fix for exact-number / missing-requirement answers was added but not re-benchmarked, because repeated tests were intentionally stopped.
- Claim-level detector checking exists in code but is disabled by default due local CUDA/runtime instability and latency concerns.

## Recommended Roadmap

### Short Term

1. Keep the current detector as the answer-include safety layer.
2. Keep answer-quality audit for completeness and missing-concept detection.
3. Use the stable quality config for demonstrations on the RTX 3090 machine.
4. Create a fresh holdout full-RAG test set before reporting final gains.

### Medium Term

1. Improve retrieval with a cleaner hybrid approach:
   - dense retrieval
   - BM25/lexical retrieval
   - cross-encoder reranking
   - source-family balancing
2. Split broad questions into subqueries before retrieval.
3. Add manual review labels for generated answers:
   - included
   - not_included
   - partial
   - contradicted
   - ambiguous

### Long Term

1. Train a separate answer-completeness or claim-coverage model.
2. Add claim-level verification in a separate process to avoid CUDA collisions.
3. Compare against at least these baselines in the final report:
   - base RAG without detector
   - RAG with detector only
   - RAG with detector + answer quality
   - RAG with detector + answer quality + stochastic uncertainty

## Google Drive Model Package

Do not commit model files to GitHub. They are ignored by `.gitignore`.

Upload this exact folder structure to Google Drive:

```text
Google Drive/
└── finreg-rag-models/
    └── finregbench_modernbert_detector_v3_hardmix/
        └── best_model/
            ├── model.pt
            └── training_state.pt
```

Local source files to upload:

```text
E:\github_proje\rag-project\models\checkpoints\finregbench_modernbert_detector_v3_hardmix\best_model\model.pt
E:\github_proje\rag-project\models\checkpoints\finregbench_modernbert_detector_v3_hardmix\best_model\training_state.pt
```

After teammates download from Drive, they should place the files back at:

```text
models/checkpoints/finregbench_modernbert_detector_v3_hardmix/best_model/model.pt
models/checkpoints/finregbench_modernbert_detector_v3_hardmix/best_model/training_state.pt
```

The config expects this path:

```yaml
hallucination_detector:
  model_path: models/checkpoints/finregbench_modernbert_detector_v3_hardmix/best_model
  base_model: tasksource/ModernBERT-base-nli
```

The base model `tasksource/ModernBERT-base-nli`, Qwen LLM, embedding model, and reranker are downloaded from Hugging Face/cache and should not be uploaded unless an offline demo package is explicitly needed.
