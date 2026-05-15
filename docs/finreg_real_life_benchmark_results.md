# FinReg Real-Life Benchmark Results

Run date: 2026-05-08

This document records the report-facing benchmark results for the FinReg RAG
detector integration. The full-RAG metrics are pre-review metrics: they are
useful for comparisons, but final answer quality still requires manual review
of `manual_review_sheet.csv`.

## Benchmark Inputs

- Controlled candidate benchmark: `benchmarks/finreg/controlled_candidate_cases.jsonl`
- Full RAG benchmark: `benchmarks/finreg/full_rag_questions.jsonl`

The controlled benchmark has 80 fixed candidate-answer cases:

- 20 `included`
- 20 `not_included`
- 20 `contradicted`
- 20 `partial`

The full RAG benchmark has 40 questions:

- 10 `factual_supported`
- 10 `false_premise`
- 10 `multi_source_nuanced`
- 10 `low_evidence_policy`

## Controlled Detector Results

These runs isolate the ModernBERT detector by keeping the candidate answer
fixed. Higher accuracy is better. Lower false include and false exclude rates
are better.

| Method | Threshold | Accuracy | Not-Included Recall | False Include Rate | False Exclude Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Deterministic baseline | 0.50 | 0.7875 | 1.0000 | 0.0000 | 0.8500 |
| Deterministic calibrated | 0.97 | 0.9625 | 1.0000 | 0.0000 | 0.1500 |
| Stochastic calibrated | 0.97 | 0.9625 | 1.0000 | 0.0000 | 0.1500 |
| V3 hardmix calibrated | 0.97 | 0.9750 | 1.0000 | 0.0000 | 0.1000 |

Interpretation:

- The detector checkpoint was not fundamentally broken.
- The original `0.50` threshold was too conservative for this benchmark: it
  rejected 17 of 20 genuinely included answers.
- Calibration to `0.97` preserved safety on not-included cases while reducing
  unnecessary rejection.
- Stochastic logit uncertainty did not change the final controlled decisions in
  this run; keep it as an ablation unless future uncertainty metrics separate
  errors more clearly.
- A short V3 hardmix fine-tune from the calibrated checkpoint added reviewed
  real-RAG-like detector examples to training. It improved controlled accuracy
  from `0.9625` to `0.9750` and reduced false excludes from `0.1500` to
  `0.1000` without increasing false includes.

## Detector Retraining Notes

The first retraining attempt, `finregbench_modernbert_detector_v2_regularized`,
did not improve the controlled benchmark. Its best threshold sweep reached
`0.9500` accuracy and introduced false includes.

The accepted retraining attempt is `finregbench_modernbert_detector_v3_hardmix`.
It starts from `models/checkpoints/finregbench_modernbert_detector/best_model`
and fine-tunes on:

- the original FinRegBench detector train split;
- 60 reviewed real-RAG-like hard examples oversampled 6x;
- 18 reviewed hard examples added to validation;
- no controlled benchmark cases.

Use `gating_finreg_modernbert_detector_v3_hardmix_calibrated` for detector-only
controlled tests and
`gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated` for the
local Qwen2.5-3B full-RAG setup.

## Full RAG Results

These runs evaluate retrieval, generation, detector, and gating together.

| Method | Detector | Gating | Answer Rate | Abstain Rate | Expected Behavior | Expected-Point Coverage | Forbidden-Claim Hit | Mean Answer-Include Risk |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-3B no detector | off | off | 0.6000 | 0.4000 | 0.5250 | 0.3250 | 0.1250 | n/a |
| Qwen2.5-3B detector baseline | on | threshold 0.50 | 0.2000 | 0.8000 | 0.3500 | 0.0700 | 0.0750 | 0.3874 |
| Qwen2.5-3B detector calibrated | on | threshold 0.97 | 0.5750 | 0.4250 | 0.5500 | 0.2421 | 0.1250 | 0.3874 |
| Qwen2.5-3B V3 hardmix detector calibrated | on | threshold 0.97 | 0.5750 | 0.4250 | 0.5500 | 0.2421 | 0.1250 | 0.4055 |
| Qwen2.5-3B V3 hardmix real-corpus index | on | threshold 0.97 | 0.9000 | 0.1000 | 0.4000 | 0.2700 | 0.1000 | 0.7427 |
| Qwen2.5-3B section+rerank no detector | off | off | 0.7750 | 0.2250 | 0.8500 | 0.4701 | 0.0000 | n/a |
| Qwen2.5-3B V3 real-corpus section+rerank | on | threshold 0.97 | 0.7250 | 0.2750 | 0.8500 | 0.4659 | 0.0000 | 0.5761 |
| Qwen2.5-3B V3 section+rerank stochastic | on | threshold 0.97 + logit MI | 0.7250 | 0.2750 | 0.8500 | 0.4659 | 0.0000 | 0.5761 |
| Qwen2.5-3B stochastic calibrated | on | threshold 0.97 + logit MI | 0.5750 | 0.4250 | 0.5500 | 0.2421 | 0.1250 | 0.3874 |
| Qwen3-8B detector calibrated | on | threshold 0.97 | 0.4750 | 0.5250 | 0.5500 | 0.2454 | 0.1000 | 0.3820 |

Interpretation:

- The uncalibrated detector/gating path over-abstained and hurt end-to-end
  behavior.
- Calibrated detector gating improved expected behavior versus the no-detector
  baseline, but expected-point coverage stayed lower than the no-detector run.
- Qwen3-8B reduced forbidden-claim hits, but did not improve the overall
  expected behavior rate on this benchmark. It abstained more often than
  Qwen2.5-3B.
- V3 hardmix improved the isolated detector benchmark, but did not change the
  full-RAG automatic metrics in this 40-question run. This separates detector
  quality from retrieval/generation quality.
- Rebuilding retrieval from the real official-document corpus changed behavior
  substantially: the index grew from 23 synthetic notes to 1,788 real chunks,
  answer rate rose to `0.9000`, and forbidden-claim hits fell to `0.1000`.
  However, expected behavior dropped to `0.4000` because the raw fixed-size
  chunks often start mid-sentence or retrieve broad nearby material. This shows
  that the next bottleneck is retrieval quality and context construction, not
  only corpus size.
- The section-aware index plus cross-encoder reranking is the best current
  full-RAG configuration. It uses a 24-document dense candidate pool, reranks
  to 8 evidence chunks, and improves expected behavior from `0.4000` to
  `0.8500` while eliminating automatic forbidden-claim hits in this run.
- A cleanup-time no-detector run over the same section-aware/reranked retrieval
  path also reached `0.8500` expected behavior. This means retrieval and prompt
  construction are currently carrying most of the end-to-end gain; detector
  gating mainly adds auditability and controlled abstention rather than a higher
  automatic score on this specific benchmark.
- The cleanup-time stochastic section+rerank run matched the deterministic
  detector run exactly on the automatic metrics. It remains useful as a
  stochastic-equation ablation, not as the default best-performing path.
- The remaining bottleneck is not only the detector. Retrieval coverage,
  answer synthesis, and the benchmark's strict expected-point matching all
  affect the full-RAG score.

## Retrieval Diagnosis

The earlier full-RAG runs used `data/vector_db/domain_finreg`, which contained
only 23 Chroma records. Retrieved contexts were synthetic benchmark notes such
as "BCBS benchmark note on BCBS 239 objective", not the official source corpus.

The real-corpus test uses
`gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus`
and indexes `data/processed/finreg/finreg_phase1_corpus.jsonl` into
`data/vector_db/domain_finreg_real` as 1,788 chunks.

This improved source authenticity, but exposed three remaining retrieval issues:

- fixed-size chunks can begin in the middle of a sentence or table row;
- dense retrieval sometimes returns broad related sections instead of the exact
  clause needed by the question;
- no reranker is applied after dense retrieval, so the generator receives noisy
  context.

Recommended next retrieval roadmap:

- keep the section-aware real-corpus index as the default official-corpus path;
- keep cross-encoder reranking as the default short-term reranker;
- evaluate a stronger long-context reranker such as mGTE as a later ablation;
- manually review the remaining low-coverage questions before claiming final
  answer-quality metrics.

## Retrieval Audit Results

The retrieval audit runs the same 40 full-RAG questions without generation or
detector logic. It checks whether the retrieved evidence contains expected
answer concepts, whether forbidden concepts appear in the retrieved evidence,
and whether chunks start in the middle of sentences.

| Retrieval Setup | Chunks | Candidate Pool | Final Context | Expected-Point Retrieval Coverage | Any Expected-Point Hit | Bad Chunk Start Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real corpus fixed-size | 1,788 | 8 | 8 | 0.6899 | 0.9000 | 0.8563 |
| Real corpus section-aware | 1,996 | 8 | 8 | 0.6228 | 0.9000 | 0.0781 |
| Real corpus section-aware + cross-encoder | 1,996 | 24 | 8 | 0.6851 | 0.9500 | 0.0813 |

Interpretation:

- Fixed-size chunking had decent lexical recall, but most retrieved chunks
  started mid-sentence or mid-table, which produced poor generation behavior.
- Section-aware chunking fixed context readability but slightly reduced recall
  when used alone.
- A wider candidate pool plus cross-encoder reranking recovered recall while
  preserving clean chunk boundaries. This explains the full-RAG jump from
  `0.4000` to `0.8500` expected behavior.

## Full RAG Breakdown By Question Type

| Method | Question Type | Expected Behavior | Abstain Rate | Coverage | Forbidden Hit Rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B no detector | factual_supported | 0.50 | 0.40 | 0.305 | 0.00 |
| Qwen2.5-3B no detector | false_premise | 0.60 | 0.50 | 0.380 | 0.40 |
| Qwen2.5-3B no detector | low_evidence_policy | 0.50 | 0.20 | 0.025 | 0.10 |
| Qwen2.5-3B no detector | multi_source_nuanced | 0.50 | 0.50 | 0.590 | 0.00 |
| Qwen2.5-3B calibrated | factual_supported | 0.50 | 0.40 | 0.305 | 0.00 |
| Qwen2.5-3B calibrated | false_premise | 0.60 | 0.50 | 0.355 | 0.40 |
| Qwen2.5-3B calibrated | low_evidence_policy | 0.60 | 0.30 | 0.025 | 0.10 |
| Qwen2.5-3B calibrated | multi_source_nuanced | 0.50 | 0.50 | 0.283 | 0.00 |
| Qwen3-8B calibrated | factual_supported | 0.50 | 0.40 | 0.305 | 0.00 |
| Qwen3-8B calibrated | false_premise | 0.70 | 0.70 | 0.335 | 0.30 |
| Qwen3-8B calibrated | low_evidence_policy | 0.50 | 0.50 | 0.025 | 0.10 |
| Qwen3-8B calibrated | multi_source_nuanced | 0.50 | 0.50 | 0.317 | 0.00 |

The low-evidence policy slice has intentionally strict expected-point matching.
It should be manually reviewed before making strong quality claims.

## Qwen 8B / Qwen 3.5 Note

`Qwen/Qwen3-8B` was used for the heavier local LLM comparison because it is a
text causal language model and can run on the RTX 3090 after enabling
`device_map: auto` and `low_cpu_mem_usage`.

`Qwen/Qwen3.5-9B` exists, but its Hugging Face config reports
`Qwen3_5ForConditionalGeneration`, not the text-only causal LM path used by this
project's wrapper. It should be treated as a future integration task rather than
mixed into the current text-only benchmark.

## Report Guidance

Recommended comparison groups for the graduation report:

- No detector baseline: shows raw RAG behavior.
- Deterministic detector baseline: shows the over-abstention failure mode.
- Calibrated deterministic detector: shows the practical detector improvement.
- V3 hardmix detector: shows whether retraining improves the detector-only and
  full-RAG results.
- Calibrated stochastic detector: shows whether stochastic equations add
  decision value.
- Qwen3-8B calibrated: shows whether a stronger local generator helps.

Recommended claims:

- Controlled detector performance improved from 0.7875 to 0.9625 accuracy after
  calibration, and to 0.9750 after V3 hardmix retraining, without increasing
  false include rate.
- Stochastic uncertainty was implemented and evaluated, but did not improve the
  selected decision metrics on this benchmark.
- Full RAG performance improved after calibration, but generation/retrieval
  quality remains a major bottleneck.
- Detector retraining helped the controlled detector benchmark, but did not
  improve the current full-RAG automatic metrics; the next gains likely require
  retrieval and answer-synthesis improvements.

## Comprehensive Report Table

This table is designed to be copied into the graduation report as the main
comparison matrix.

| Experiment | Dataset / Scope | Generator | Detector / Gating | Stochastic | Threshold | Main Purpose | Accuracy | Not-Included Recall | False Include | False Exclude | Answer Rate | Abstain Rate | Expected Behavior | Coverage | Forbidden Hit | Mean Risk | Practical Reading |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Controlled baseline | 80 fixed candidate answers | none | ModernBERT answer-include detector | no | 0.50 | Isolate raw detector behavior | 0.7875 | 1.0000 | 0.0000 | 0.8500 | n/a | n/a | n/a | n/a | n/a | n/a | Safe but far too conservative; rejects most included answers. |
| Controlled calibrated | 80 fixed candidate answers | none | ModernBERT answer-include detector | no | 0.97 | Test calibrated detector threshold | 0.9625 | 1.0000 | 0.0000 | 0.1500 | n/a | n/a | n/a | n/a | n/a | n/a | Best detector-only setting: safety preserved, false exclusions much lower. |
| Controlled stochastic calibrated | 80 fixed candidate answers | none | ModernBERT answer-include detector | yes | 0.97 | Test stochastic uncertainty as detector ablation | 0.9625 | 1.0000 | 0.0000 | 0.1500 | n/a | n/a | n/a | n/a | n/a | n/a | Same decisions as calibrated deterministic; useful as ablation, not as current default. |
| Controlled V3 hardmix calibrated | 80 fixed candidate answers | none | ModernBERT answer-include detector fine-tuned with reviewed hardmix data | no | 0.97 | Test retrained detector checkpoint | 0.9750 | 1.0000 | 0.0000 | 0.1000 | n/a | n/a | n/a | n/a | n/a | n/a | Best controlled detector result so far; retraining reduced false exclusions without reducing safety. |
| Full RAG no detector | 40 generated answers | Qwen2.5-3B-Instruct | detector off, gating off | no | n/a | Baseline end-to-end RAG behavior | n/a | n/a | n/a | n/a | 0.6000 | 0.4000 | 0.5250 | 0.3250 | 0.1250 | n/a | Raw RAG answers more often, but still has unsupported/forbidden-claim risk. |
| Full RAG uncalibrated detector | 40 generated answers | Qwen2.5-3B-Instruct | ModernBERT gating | no | 0.50 | Show failure mode of overly strict gating | n/a | n/a | n/a | n/a | 0.2000 | 0.8000 | 0.3500 | 0.0700 | 0.0750 | 0.3874 | Over-abstains heavily; safer but impractical and lower answer usefulness. |
| Full RAG calibrated detector | 40 generated answers | Qwen2.5-3B-Instruct | ModernBERT gating | no | 0.97 | Practical detector-enabled RAG setup | n/a | n/a | n/a | n/a | 0.5750 | 0.4250 | 0.5500 | 0.2421 | 0.1250 | 0.3874 | Best balanced 3B setup; improves expected behavior over no-detector baseline with similar coverage tradeoff. |
| Full RAG V3 hardmix detector | 40 generated answers | Qwen2.5-3B-Instruct | V3 hardmix ModernBERT gating | no | 0.97 | Test whether detector retraining improves end-to-end RAG | n/a | n/a | n/a | n/a | 0.5750 | 0.4250 | 0.5500 | 0.2421 | 0.1250 | 0.4055 | Same automatic full-RAG outcome as calibrated detector; detector-only gains did not fix generation/retrieval bottlenecks. |
| Full RAG V3 real-corpus retrieval | 40 generated answers | Qwen2.5-3B-Instruct | V3 hardmix ModernBERT gating over 1,788 official-document chunks | no | 0.97 | Test real official corpus retrieval instead of synthetic note retrieval | n/a | n/a | n/a | n/a | 0.9000 | 0.1000 | 0.4000 | 0.2700 | 0.1000 | 0.7427 | Confirms retrieval was using the wrong evidence base; real corpus increases answering but needs section-aware chunking/reranking. |
| Full RAG section+rerank no detector | 40 generated answers | Qwen2.5-3B-Instruct | detector off, gating off over section-aware official chunks with cross-encoder reranking | no | n/a | Cleanup-time base RAG over the best retrieval path | n/a | n/a | n/a | n/a | 0.7750 | 0.2250 | 0.8500 | 0.4701 | 0.0000 | n/a | Shows the retrieval/prompt fix is the main end-to-end driver; no detector risk score is produced. |
| Full RAG V3 section+rerank retrieval | 40 generated answers | Qwen2.5-3B-Instruct | V3 hardmix ModernBERT gating over section-aware official chunks with cross-encoder reranking | no | 0.97 | Test retrieval-quality fix after real-corpus diagnosis | n/a | n/a | n/a | n/a | 0.7250 | 0.2750 | 0.8500 | 0.4659 | 0.0000 | 0.5761 | Best current full-RAG setup: cleaner context, stronger expected behavior, and no automatic forbidden-claim hits. |
| Full RAG V3 section+rerank stochastic | 40 generated answers | Qwen2.5-3B-Instruct | V3 hardmix ModernBERT gating over section-aware official chunks with cross-encoder reranking | yes | 0.97 | Cleanup-time stochastic-equation ablation over the best retrieval path | n/a | n/a | n/a | n/a | 0.7250 | 0.2750 | 0.8500 | 0.4659 | 0.0000 | 0.5761 | Same automatic outcome as deterministic section+rerank; stochastic path is preserved and runnable. |
| Full RAG stochastic calibrated | 40 generated answers | Qwen2.5-3B-Instruct | ModernBERT gating | yes | 0.97 | Test stochastic equations in full RAG | n/a | n/a | n/a | n/a | 0.5750 | 0.4250 | 0.5500 | 0.2421 | 0.1250 | 0.3874 | No measurable gain over deterministic calibrated on this benchmark. |
| Full RAG larger local model | 40 generated answers | Qwen3-8B | ModernBERT gating | no | 0.97 | Test stronger local LLM on RTX 3090 | n/a | n/a | n/a | n/a | 0.4750 | 0.5250 | 0.5500 | 0.2454 | 0.1000 | 0.3820 | Similar expected behavior to 3B, slightly fewer forbidden hits, but more abstention. |

Suggested report conclusion from this table:

- The detector itself is viable after calibration.
- The original threshold caused over-abstention, not model failure.
- A small hardmix retrain improved detector-only performance, but not the
  current full-RAG metrics.
- The old full-RAG retrieval baseline was not report-grade because it used a
  23-record synthetic note index. The real-corpus index is the correct starting
  point, but it needs better chunking and reranking.
- Stochastic uncertainty is implemented and experimentally valid, but did not
  improve the selected metrics in this run.
- Larger local generation alone did not solve the problem; retrieval quality,
  answer synthesis, and gating calibration remain the key engineering levers.
- The V3 hardmix detector is a modest but real detector-only improvement; full
  RAG should still be reported separately because detector gains do not
  automatically fix retrieval or generation mistakes.

## Reproduction Commands

Build the V3 hardmix training split:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\build_finreg_detector_hardmix_dataset.py `
  --output-dir data\training\finregbench_detector_v3_hardmix
```

Train the V3 hardmix detector:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\train_hallucination_model.py `
  --config finregbench_modernbert_detector_v3_hardmix `
  --data-dir data\training\finregbench_detector_v3_hardmix `
  --output-dir models\checkpoints\finregbench_modernbert_detector_v3_hardmix `
  --init-from models\checkpoints\finregbench_modernbert_detector\best_model `
  --tensorboard
```

Controlled calibrated:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode controlled `
  --config gating_finreg_modernbert_detector_calibrated `
  --k 5 `
  --run-name controlled80_deterministic_calibrated
```

Controlled stochastic calibrated:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode controlled `
  --config gating_finreg_modernbert_detector_stochastic_calibrated `
  --k 5 `
  --run-name controlled80_stochastic_calibrated
```

Full RAG Qwen2.5-3B calibrated:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector_calibrated `
  --k 5 `
  --run-name fullrag40_qwen25_3b_detector_calibrated
```

Full RAG Qwen2.5-3B V3 hardmix calibrated:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated `
  --k 5 `
  --run-name fullrag40_qwen25_3b_detector_v3_hardmix_calibrated
```

Build the real official-document FinReg index:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
$env:INDEX_ONLY='1'
.\.venv\Scripts\python.exe scripts\index_domain_corpus.py `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus `
  --corpus data\processed\finreg\finreg_phase1_corpus.jsonl `
  --reset-collection `
  --index-only
```

Full RAG Qwen2.5-3B V3 hardmix with real-corpus retrieval:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus `
  --k 8 `
  --run-name fullrag40_qwen25_3b_detector_v3_hardmix_real_corpus_calibrated
```

Build the section-aware real official-document FinReg index:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
$env:INDEX_ONLY='1'
.\.venv\Scripts\python.exe scripts\index_domain_corpus.py `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section `
  --corpus data\processed\finreg\finreg_phase1_corpus.jsonl `
  --reset-collection `
  --index-only
```

Audit section-aware retrieval with cross-encoder reranking:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\audit_finreg_retrieval.py `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank `
  --k 24 `
  --run-name real_corpus_section_crossencoder_pool24_top8
```

Full RAG Qwen2.5-3B V3 hardmix with section-aware retrieval and reranking:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank `
  --k 24 `
  --run-name fullrag40_qwen25_3b_section_crossencoder_pool24_prompt_v2_v3
```

Full RAG Qwen2.5-3B section-aware retrieval and reranking without detector:

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

Full RAG Qwen2.5-3B V3 hardmix with section-aware retrieval, reranking, and
stochastic uncertainty:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='.'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank_stochastic `
  --k 24 `
  --run-name cleanup_fullrag40_detector_stochastic_section_rerank
```

Full RAG Qwen2.5-3B stochastic calibrated:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector_stochastic_calibrated `
  --k 5 `
  --run-name fullrag40_qwen25_3b_detector_stochastic_calibrated
```

Full RAG Qwen3-8B calibrated:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_8b_modernbert_detector_calibrated `
  --k 5 `
  --run-name fullrag40_qwen3_8b_detector_calibrated
```

Controlled V3 hardmix calibrated:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode controlled `
  --config gating_finreg_modernbert_detector_v3_hardmix_calibrated `
  --k 5 `
  --run-name controlled80_v3_hardmix_calibrated
```
