# FinReg RAG Detector Integration Change Report

Date: 2026-05-08

Branch: `finreg-detector-integration`

This report summarizes the practical engineering work completed for the FinReg
RAG detector integration, benchmark setup, detector training workflow, and
retrieval improvements. It is written as a commit-facing and graduation-report
reference: what changed, why it changed, how it was tested, and what remains.

## 1. Goal

The project goal is to build and evaluate a financial-regulation RAG system
that can answer supported questions, reject false premises, and avoid producing
answers that are not included in the retrieved evidence.

The core detector task was renamed conceptually from an "unsupported answer"
framing to an **answer include** framing:

- `included`: the retrieved context includes/supports the candidate answer;
- `not_included`: the answer is not established by the retrieved context;
- contradiction remains a more specific failure mode where the answer conflicts
  with the evidence.

This distinction matters because many real RAG failures are not direct
contradictions. They are unsupported additions, invented thresholds, invented
deadlines, or over-specific answers when evidence is incomplete.

## 2. Environment And Local Model Notes

The work was developed on Windows with a local Python virtual environment.
The user's machine constraints were treated as part of the design:

- Windows OS;
- 16 GB DDR4 RAM;
- NVIDIA RTX 3090 GPU;
- PyTorch/CUDA installation kept under user control;
- model/checkpoint files kept out of Git due to size.

Large generated/local artifacts are intentionally ignored:

- `models/`
- `data/vector_db/`
- `reports/`
- `logs/`

The detector checkpoint used by the latest configs is local-only:

```text
models/checkpoints/finregbench_modernbert_detector_v3_hardmix/best_model/
```

For sharing the project with teammates, send only the model directory needed
for inference/training separately through Drive or another large-file channel.
Do not commit model `.pt`, `.safetensors`, vector DB, or report run artifacts.

## 3. Dataset And Benchmark Work

Two report-facing benchmark sets were prepared under `benchmarks/finreg/`.

### Controlled Candidate Benchmark

File:

```text
benchmarks/finreg/controlled_candidate_cases.jsonl
```

Purpose:

- isolate the detector;
- keep the candidate answer fixed;
- measure whether the detector marks answers as `included` or `not_included`
  correctly.

Composition:

- 80 cases total;
- 20 included;
- 20 not included;
- 20 contradicted;
- 20 partial.

Why this exists:

Full RAG combines retrieval, generation, detector, and gating. If full RAG fails,
it is hard to know which layer caused the failure. The controlled benchmark
removes generation from the equation and tests the detector directly.

### Full RAG Benchmark

File:

```text
benchmarks/finreg/full_rag_questions.jsonl
```

Purpose:

- test retrieval + generation + detector + gating together;
- cover realistic question types;
- provide automatic metrics plus manual review sheets.

Composition:

- 40 questions total;
- 10 factual supported;
- 10 false premise;
- 10 multi-source nuanced;
- 10 low-evidence policy.

The full RAG benchmark is intentionally stricter than a normal demo. Some
questions ask about invented deadlines, portals, thresholds, public dashboards,
or approval requirements. The correct behavior is often to refute or abstain,
not to sound confident.

## 4. Metric Naming Cleanup

The previous `unsupported` naming was too vague for the actual detector role.
Project-facing naming was changed toward **answer include**:

- `answer_include_risk`
- `answer_include_score`
- `answer_include_detected`
- `Expected Behavior`

Important interpretation:

- **Answer rate** means how often the system produced an answer instead of
  abstaining. Higher is not automatically better.
- **Abstain rate** means how often the system refused due to insufficient
  confidence/evidence. Moderate abstention is useful when evidence is weak.
- **Expected behavior match** is the main automatic behavior metric for full
  RAG. It checks whether the system did the expected type of thing for the
  question: answer, refute, synthesize cautiously, or abstain.
- **Expected-point coverage** checks whether important answer concepts appear
  in the generated answer. It is useful but imperfect, because wording can vary.
- **Forbidden-claim hit** checks whether known bad/invented claims appear. Lower
  is better.
- **Answer include risk** estimates the detector's risk that the answer is not
  included in the provided context.

The benchmark script now treats forbidden claims carefully: if a forbidden
phrase appears in a negative/refuting context, it is not automatically counted
as a forbidden claim.

## 5. Detector Calibration And Training

The detector was evaluated in controlled mode first.

### Initial Finding

The raw threshold `0.50` was too conservative. It was safe on not-included
answers but rejected too many genuinely included answers.

### Calibrated Threshold

A threshold of `0.97` preserved not-included recall while reducing false
exclusion of included answers.

Controlled detector results:

| Method | Threshold | Accuracy | Not-Included Recall | False Include | False Exclude |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline deterministic | 0.50 | 0.7875 | 1.0000 | 0.0000 | 0.8500 |
| Calibrated deterministic | 0.97 | 0.9625 | 1.0000 | 0.0000 | 0.1500 |
| Stochastic calibrated | 0.97 | 0.9625 | 1.0000 | 0.0000 | 0.1500 |
| V3 hardmix calibrated | 0.97 | 0.9750 | 1.0000 | 0.0000 | 0.1000 |

### V3 Hardmix Training

Files:

```text
scripts/build_finreg_detector_hardmix_dataset.py
config/finregbench_modernbert_detector_v3_hardmix.yaml
config/gating_finreg_modernbert_detector_v3_hardmix_calibrated.yaml
```

Purpose:

- fine-tune the detector with reviewed hard cases closer to real RAG behavior;
- improve false-exclusion behavior without sacrificing safety;
- keep controlled benchmark cases out of training.

Result:

V3 hardmix became the best detector-only checkpoint:

- accuracy: `0.9750`;
- not-included recall: `1.0000`;
- false include: `0.0000`;
- false exclude: `0.1000`.

Important conclusion:

The detector is not the only bottleneck. V3 improved detector-only performance,
but early full-RAG results still depended heavily on retrieval and generation.

## 6. Stochastic Uncertainty Work

Stochastic detector configs were added to evaluate uncertainty variants:

```text
config/gating_finreg_modernbert_detector_stochastic_calibrated.yaml
config/gating_finreg_local_qwen3_modernbert_detector_stochastic.yaml
config/gating_finreg_local_qwen3_modernbert_detector_stochastic_calibrated.yaml
config/gating_finreg_local_qwen3_8b_modernbert_detector_stochastic.yaml
config/gating_finreg_local_qwen3_8b_modernbert_detector_stochastic_calibrated.yaml
```

Purpose:

- provide an ablation for stochastic equations/uncertainty;
- compare deterministic gating vs stochastic uncertainty-aware gating.

Current conclusion:

Stochastic uncertainty was implemented and evaluated, but it did not improve the
selected benchmark metrics in the current runs. It should be reported as an
experimental ablation rather than the default production path.

## 7. Local LLM Configs

Local Qwen configs were prepared for 3B and 8B class models.

3B path:

```text
config/gating_finreg_local_qwen3_modernbert_detector.yaml
config/gating_finreg_local_qwen3_modernbert_detector_calibrated.yaml
config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated.yaml
```

8B path:

```text
config/gating_finreg_local_qwen3_8b_modernbert_detector.yaml
config/gating_finreg_local_qwen3_8b_modernbert_detector_calibrated.yaml
```

Observed behavior:

- Qwen2.5-3B was practical on the RTX 3090 and became the main local benchmark
  generator.
- Qwen3-8B was tested as a stronger local model. It reduced some forbidden-hit
  risk but abstained more and did not clearly improve expected behavior.
- Larger generation alone did not solve the RAG problem. Retrieval quality was
  the bigger lever.

## 8. Real-Corpus Retrieval Diagnosis

A major issue was found in the old full-RAG setup:

```text
data/vector_db/domain_finreg
```

contained only 23 synthetic benchmark notes, not the full official corpus.

The real official-document corpus exists at:

```text
data/processed/finreg/finreg_phase1_corpus.jsonl
```

It contains 31 official FinReg documents from sources such as BCBS, EBA, ECB,
PRA/BoE, and Fed/OCC.

The real-corpus config was added:

```text
config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus.yaml
```

Indexing the real corpus created:

- 1,788 fixed-size chunks;
- a real official-document Chroma collection.

Result:

The real corpus improved source authenticity, but raw fixed-size retrieval still
performed poorly in full RAG:

| Setup | Answer Rate | Expected Behavior | Coverage | Forbidden Hit |
| --- | ---: | ---: | ---: | ---: |
| Real corpus fixed-size | 0.9000 | 0.4000 | 0.2700 | 0.1000 |

Interpretation:

The model answered more often because it had real evidence, but the context was
still noisy. Many chunks began mid-sentence or mid-table, and dense retrieval
returned broad related material instead of exact evidence.

## 9. Retrieval Audit Tool

File:

```text
scripts/audit_finreg_retrieval.py
```

Purpose:

Evaluate retrieval without generation or detector logic.

It reports:

- collection size;
- retrieved source families;
- expected-point retrieval coverage;
- forbidden concept retrieval hits;
- unique source diversity;
- top retrieved previews;
- chunk start quality.

Why it matters:

Without a retrieval-only audit, every full-RAG failure looks like a possible LLM
or detector problem. The audit showed the retrieval/context construction layer
was the main current bottleneck.

## 10. Section-Aware Chunking

File changed:

```text
src/chunking/strategies/section_aware.py
```

New config:

```text
config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section.yaml
```

Change:

The section-aware chunker was improved so overlap no longer blindly starts from
the middle of a word/sentence. It now prefers sentence-aware overlap and drops
bad overlap when it would create malformed chunk starts.

Retrieval audit result:

| Retrieval Setup | Chunks | Expected-Point Retrieval Coverage | Any Expected-Point Hit | Bad Chunk Start Rate |
| --- | ---: | ---: | ---: | ---: |
| Real corpus fixed-size | 1,788 | 0.6899 | 0.9000 | 0.8563 |
| Real corpus section-aware | 1,996 | 0.6228 | 0.9000 | 0.0781 |

Interpretation:

Section-aware chunking greatly improved readability/cleanliness, but slightly
reduced lexical recall when used alone. That led to the reranking step.

## 11. Cross-Encoder Reranking

New config:

```text
config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank.yaml
```

Design:

- dense retrieval gets 24 candidates;
- cross-encoder reranker selects the top 8 final evidence chunks;
- the LLM receives cleaner, reranked context.

Retrieval audit result:

| Retrieval Setup | Chunks | Candidate Pool | Final Context | Expected-Point Retrieval Coverage | Any Expected-Point Hit | Bad Chunk Start Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real corpus section-aware + cross-encoder | 1,996 | 24 | 8 | 0.6851 | 0.9500 | 0.0813 |

Interpretation:

This kept the clean chunk boundaries from section-aware chunking and recovered
most of the recall lost by sectioning alone.

## 12. Prompt Improvements

File changed:

```text
src/rag/llm_wrapper.py
```

Prompt behavior was tightened:

- use only provided context;
- if answer is not in context, abstain;
- if asked for a specific rule, deadline, threshold, portal, template, or
  approval requirement, state it only when explicit in evidence;
- if evidence is incomplete or mixed, state what is supported and what is not
  established.

Why:

The old prompt still allowed the model to fill gaps in low-evidence questions.
The new prompt is more aligned with the detector's role and the full-RAG test
design.

## 13. Benchmark Script Improvements

File changed:

```text
scripts/run_finreg_real_life_benchmark.py
```

Key changes:

- answer-include naming;
- expected behavior metric;
- concept-aware expected-point matching;
- forbidden-claim matching that respects negation/refutation;
- full-RAG report output;
- manual review sheet fields;
- backward-compatible aliases for old metric names.

Why:

The earlier automatic metrics were too brittle. For example, a correct answer
that says "there is no explicit public dashboard requirement" should not be
penalized merely because the words "public dashboard" appear.

## 14. Full RAG Result Summary

The latest best full-RAG configuration is:

```text
gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank
```

Run command:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank `
  --k 24 `
  --run-name fullrag40_qwen25_3b_section_crossencoder_pool24_prompt_v2_v3
```

Comparison:

| Setup | Answer Rate | Abstain Rate | Expected Behavior | Coverage | Forbidden Hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old synthetic-note calibrated RAG | 0.5750 | 0.4250 | 0.5500 | 0.2421 | 0.1250 |
| Real corpus fixed-size | 0.9000 | 0.1000 | 0.4000 | 0.2700 | 0.1000 |
| Real corpus section-aware + rerank | 0.7250 | 0.2750 | 0.8500 | 0.4659 | 0.0000 |

Main conclusion:

The best improvement came from retrieval engineering, not simply from a stronger
LLM or detector retraining. The correct evidence base, cleaner chunking, wider
candidate retrieval, and reranking together produced the strongest full-RAG
behavior.

## 15. Files Added

Important new files:

```text
scripts/audit_finreg_retrieval.py
scripts/build_finreg_detector_hardmix_dataset.py
config/finregbench_modernbert_detector_v3_hardmix.yaml
config/gating_finreg_modernbert_detector_calibrated.yaml
config/gating_finreg_modernbert_detector_stochastic_calibrated.yaml
config/gating_finreg_modernbert_detector_v3_hardmix_calibrated.yaml
config/gating_finreg_local_qwen3_modernbert_detector_calibrated.yaml
config/gating_finreg_local_qwen3_modernbert_detector_stochastic.yaml
config/gating_finreg_local_qwen3_modernbert_detector_stochastic_calibrated.yaml
config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated.yaml
config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus.yaml
config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section.yaml
config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank.yaml
config/gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank_stochastic.yaml
config/gating_finreg_local_qwen3_8b_modernbert_detector.yaml
config/gating_finreg_local_qwen3_8b_modernbert_detector_calibrated.yaml
config/gating_finreg_local_qwen3_8b_modernbert_detector_stochastic.yaml
config/gating_finreg_local_qwen3_8b_modernbert_detector_stochastic_calibrated.yaml
docs/finreg_real_life_benchmark_results.md
docs/finreg_project_change_report.md
```

## 16. Cleanup Pass

After the main integration work, a cleanup pass removed files that were either
archival, generated locally, failed experimental checkpoints, or not required by
the current report-facing runs.

Removed from Git:

```text
archive/
CHECKPOINT_RECOVERY.md
CHUNKING_PAIR_EVAL_REPORT.md
IMPLEMENTATION_COMPLETE.md
TRAINING_IMPLEMENTATION_STATUS.md
data/vector_db/domain_finreg/
```

Removed only from the local machine because they are ignored/generated artifacts:

```text
models.zip
reports/
logs/
evaluation_results/
models/llm/models--Qwen--Qwen3-8B/
models/llm/models--Qwen--Qwen2.5-1.5B-Instruct/
models/llm/models--Qwen--Qwen3.5-9B/
models/llm/models--sshleifer--tiny-gpt2/
models/checkpoints/finregbench_modernbert_detector_v2_regularized/
models/checkpoints/finregbench_modernbert_detector/checkpoint-step-400/
models/checkpoints/finregbench_modernbert_detector/best_model/optimizer.pt
models/checkpoints/finregbench_modernbert_detector_v3_hardmix/best_model/optimizer.pt
data/vector_db/domain_finreg_real/
```

Kept intentionally:

- `Qwen/Qwen2.5-3B-Instruct` local cache, because it is the current main local
  generator;
- `sentence-transformers/all-MiniLM-L6-v2` embeddings cache;
- V3 hardmix detector `model.pt`;
- original detector `model.pt`, because baseline/stochastic comparisons still
  reference it;
- `models/training/`, because detector loading can still need the base
  ModernBERT tokenizer/model cache;
- section-aware real-corpus vector DB, because it is required for current full
  RAG tests;
- stochastic configs and stochastic sampling code.

Approximate local model/cache size after cleanup:

- `models/llm`: about 5.76 GB, down from about 23.92 GB;
- `models/checkpoints`: about 1.11 GB, down from about 8.36 GB;
- `models/embeddings`: about 87 MB;
- `data/vector_db/domain_finreg_real_section`: about 34 MB.

## 17. Post-Cleanup Full RAG Verification

The post-cleanup verification uses the same section-aware real-corpus index and
cross-encoder reranking path:

Base RAG without detector/gating:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank `
  --k 24 `
  --disable-detector `
  --disable-gating `
  --run-name cleanup_fullrag40_base_no_detector_section_rerank
```

Detector RAG:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank `
  --k 24 `
  --run-name cleanup_fullrag40_detector_section_rerank
```

Detector plus stochastic logit uncertainty RAG:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank_stochastic `
  --k 24 `
  --run-name cleanup_fullrag40_detector_stochastic_section_rerank
```

Post-cleanup results:

| Setup | Answer Rate | Abstain Rate | Expected Behavior | Coverage | Forbidden Hit | Detector Run Rate | Mean Risk | Mean Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base RAG, no detector/gating | 0.7750 | 0.2250 | 0.8500 | 0.4701 | 0.0000 | 0.0000 | n/a | 4.6385s |
| Detector RAG | 0.7250 | 0.2750 | 0.8500 | 0.4659 | 0.0000 | 0.7750 | 0.5761 | 4.6474s |
| Detector + stochastic RAG | 0.7250 | 0.2750 | 0.8500 | 0.4659 | 0.0000 | 0.7750 | 0.5761 | 4.7948s |

Interpretation:

- cleanup did not break the current best full-RAG path;
- stochastic equations/configuration were preserved and runnable;
- in this 40-question run, stochastic logit uncertainty did not change the
  final automatic decisions versus deterministic detector gating;
- the detector path abstained slightly more than the base path, but kept the
  same expected behavior score and exposed answer-include risk for audit.

## 18. Files Modified

Important modified files:

```text
benchmarks/finreg/README.md
benchmarks/finreg/controlled_candidate_cases.jsonl
benchmarks/finreg/full_rag_questions.jsonl
config/gating_finreg_local_qwen15_modernbert_detector.yaml
config/gating_finreg_local_qwen3_modernbert_detector.yaml
docs/finreg_real_life_evaluation_plan.md
scripts/run_finreg_real_life_benchmark.py
src/chunking/strategies/section_aware.py
src/rag/llm_wrapper.py
```

## 19. Verification Performed

Code syntax checks:

```powershell
.\.venv\Scripts\python.exe -m py_compile `
  scripts\run_finreg_real_life_benchmark.py `
  scripts\audit_finreg_retrieval.py `
  scripts\build_finreg_detector_hardmix_dataset.py `
  src\chunking\strategies\section_aware.py `
  src\rag\llm_wrapper.py
```

Repository whitespace check:

```powershell
git diff --check
```

Result:

- no blocking whitespace errors;
- only Windows LF/CRLF warnings from Git.

Functional runs completed:

- controlled detector benchmark;
- real-corpus index build;
- retrieval audit for fixed-size, section-aware, and section-aware+rerank;
- 8-question full-RAG smoke test;
- 40-question full-RAG benchmark for the current best configuration;
- post-cleanup 40-question base RAG run with detector/gating disabled;
- post-cleanup 40-question detector RAG run;
- post-cleanup 40-question detector + stochastic RAG run.

## 20. Remaining Work

Recommended next steps:

1. Manually review the latest full-RAG `manual_review_sheet.csv` before claiming
   final answer-quality metrics.
2. Inspect remaining low-coverage questions individually:
   - stress testing/SREP;
   - manual workaround/data lineage;
   - climate/ESG;
   - outsourcing/cloud;
   - ICT security.
3. Try query rewriting or query expansion for questions whose concepts are too
   broad for dense retrieval alone.
4. Evaluate mGTE reranking as a stronger reranker ablation.
5. Consider source-family-aware retrieval balancing for multi-source questions.
6. Keep detector retraining separate from retrieval work in the report, because
   the experiments show these are different bottlenecks.

## 21. Recommended Report Claim

A fair graduation-report claim is:

> The detector became reliable in controlled answer-include classification after
> calibration and hardmix fine-tuning, but full RAG quality was primarily limited
> by retrieval and context construction. Moving from a synthetic note index to a
> real official-document corpus exposed chunking noise; section-aware chunking
> and cross-encoder reranking improved expected full-RAG behavior from 0.4000 to
> 0.8500 and reduced forbidden-claim hits to 0.0000 in the automatic benchmark.

This claim is stronger and cleaner than saying only "the model improved",
because it explains which subsystem improved and why.
