# FinReg Unsupported-Answer Detector

This project can run a retrieval-backed unsupported-answer detector trained on
FinRegBench-style examples. The detector is integrated into the RAG pipeline and
can also be tested without an LLM.

## What This Change Commits

- FinReg detector training and evaluation configuration.
- Data preparation support for FinRegBench detector examples.
- ModernBERT-based detector training and evaluation updates.
- RAG pipeline integration for `unsupported_risk` and `support_score`.
- Detector-only real-life smoke test script.
- OpenRouter and detector-only runtime configs.

## What Stays Outside Git

These paths are intentionally local artifacts and should not be committed:

- `models/`
- `logs/`
- `reports/`
- `evaluation_results/`
- `data/vector_db/`
- generated training/test data under `data/`

If GitHub Desktop shows changes under `data/vector_db/`, leave them unchecked.
Those files are runtime database state, not source code.

## Model Artifact Restore Path

The runtime configs expect the trained checkpoint here:

```text
models/checkpoints/finregbench_modernbert_detector/best_model/
```

For inference and demo usage, the minimum files to restore are:

```text
models/checkpoints/finregbench_modernbert_detector/best_model/model.pt
models/checkpoints/finregbench_modernbert_detector/best_model/training_state.pt
```

`optimizer.pt` is only needed if training should resume from the checkpoint. It
is not needed for normal inference.

## Detector-Only Smoke Test

Run this from the repository root after restoring the model artifact:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_detector_test.py --config gating_finreg_modernbert_detector --k 5
```

This test retrieves evidence from the local FinReg vector database, checks
handwritten candidate answers, and writes local-only results under `reports/`.

## Full RAG Usage

For a full RAG run with an OpenRouter LLM, set the API key and use the OpenRouter
config:

```powershell
$env:OPENROUTER_API_KEY='your_key_here'
```

Use `config/gating_finreg_openrouter_modernbert_detector.yaml` for the full
retrieval, generation, and detector-gating path.

## Detector Decision Semantics

The detector uses the training-time input format:

```text
Question: <user question>
Candidate answer: <answer being checked>
```

Each retrieved context is scored separately. The pipeline then aggregates support
with `unsupported_best`:

- `support_score` is the best entailment score found across retrieved contexts.
- `unsupported_risk` is `1 - support_score`.
- Answers above the configured unsupported threshold are treated as unsupported.
