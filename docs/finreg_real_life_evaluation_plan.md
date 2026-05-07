# FinReg Real-Life Evaluation Plan

This plan defines the report-facing evaluation protocol for the FinReg RAG
project.

## Evaluation Questions

1. Does the detector identify answers that are not supported by retrieved
   financial-regulation evidence?
2. Does detector-based gating reduce unsafe answer acceptance in an end-to-end
   RAG setting?
3. Does stochastic uncertainty provide useful extra signal compared with the
   deterministic detector path?

## Test 1: Controlled Candidate Benchmark

Purpose: isolate the detector.

Input:

- user question
- fixed candidate answer
- expected label: `supported` or `unsupported`
- detailed label: `supported`, `unsupported`, `contradicted`, or `partial`

Pipeline:

```text
question -> retrieval -> fixed candidate answer -> detector -> supported/unsupported
```

Why this is useful:

- removes LLM generation randomness
- produces clean confusion-matrix metrics
- directly measures detector behavior

Main metrics:

- accuracy
- unsupported precision
- unsupported recall
- unsupported F1
- false accept rate: unsupported answer predicted as supported
- false reject rate: supported answer predicted as unsupported
- mean `unsupported_risk`
- mean `support_score`

Command:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode controlled `
  --config gating_finreg_modernbert_detector
```

Stochastic variant:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode controlled `
  --config gating_finreg_modernbert_detector_stochastic
```

## Test 2: Full RAG Benchmark

Purpose: evaluate the whole system.

Input:

- user question only
- manual review guidance

Pipeline:

```text
question -> retrieval -> LLM generation -> detector -> gating -> final answer
```

Why this is useful:

- closest to real usage
- captures retrieval, generation, detector, and gating interactions
- produces manual review sheet for report-grade evaluation

Automatic metrics:

- gating action counts
- abstain rate
- mean `unsupported_risk`
- mean `support_score`
- mean latency

Manual labels to add after generation:

- `supported`
- `unsupported`
- `contradicted`
- `partial`
- `ambiguous`

Recommended error types:

- `fabricated_fact`
- `wrong_number_or_threshold`
- `cross_document_conflict`
- `misinterpretation`
- `retrieval_failure`
- `over_abstain`
- `incomplete_answer`

Command:

```powershell
$env:PYTHONUTF8='1'
$env:OPENROUTER_API_KEY='your_key_here'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_openrouter_modernbert_detector
```

No-detector baseline:

```powershell
$env:PYTHONUTF8='1'
$env:OPENROUTER_API_KEY='your_key_here'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_openrouter_modernbert_detector `
  --disable-detector `
  --disable-gating `
  --run-name fullrag_no_detector_baseline
```

Stochastic variant:

```powershell
$env:PYTHONUTF8='1'
$env:OPENROUTER_API_KEY='your_key_here'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_openrouter_modernbert_detector_stochastic
```

Local Qwen 1.5B variant:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen15_modernbert_detector `
  --run-name fullrag_local_qwen15_modernbert
```

Local Qwen 3B variant:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector `
  --run-name fullrag_local_qwen3_modernbert
```

Local Qwen 3B no-detector baseline:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --config gating_finreg_local_qwen3_modernbert_detector `
  --disable-detector `
  --disable-gating `
  --run-name fullrag_local_qwen3_no_detector_baseline
```

## Deterministic vs Stochastic Comparison

Deterministic detector:

```text
p = softmax(f_theta(context, answer))
unsupported_risk = p(neutral) + p(contradiction)
support_score = max_context p(entailment)
```

Stochastic detector:

```text
p_t = softmax(f_theta,t(context, answer)), t = 1...T
mean_prediction = average_t(p_t)
uncertainty = MI(p_t) or variance(p_t)
```

Interpretation:

- deterministic path asks whether the answer is supported
- stochastic path also asks whether the detector's decision is stable
- if uncertainty rises on wrong or partial cases, it is useful for gating
- if uncertainty does not separate errors from correct cases, keep it as an
  experimental ablation rather than a default method

## Report Tables

Controlled benchmark table:

```text
Method              Accuracy  Unsupported Recall  False Accept  False Reject  Runtime
Deterministic       ...
Stochastic Logit-MI ...
```

Full RAG table:

```text
Method                 Answer Rate  Abstain Rate  Unsupported Risk  Manual Error Rate  Runtime
No detector baseline   ...
Detector gating        ...
Stochastic gating      ...
```

Local model comparison table:

```text
Method                         Answer Rate  Abstain Rate  Mean Risk  Mean Latency
Local Qwen 1.5B + detector      ...
Local Qwen 3B + detector        ...
Local Qwen 3B no detector       ...
```

Qualitative table:

```text
Question | Method | Answer | Detector/Gating Decision | Manual Label | Error Type
```

## Repository Recommendation

Use `Talha-Dmr/rag-project` for:

- RAG pipeline code
- detector integration
- benchmark scripts
- small benchmark inputs
- report methodology docs

Use the FinRegBench repository only for:

- dataset generation
- dataset documentation
- training/evaluation data construction

Do not commit trained model weights to either repository. Share model artifacts
through Drive or a release artifact. For normal inference, the required model
files are:

```text
models/checkpoints/finregbench_modernbert_detector/best_model/model.pt
models/checkpoints/finregbench_modernbert_detector/best_model/training_state.pt
```
