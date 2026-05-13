# FinReg Final Holdout Evaluation Plan

Date: 2026-05-13

## Holdout Set

Final holdout questions:

```text
benchmarks/finreg/final_holdout_80_questions.jsonl
```

Composition:

- 80 questions
- 20 topics
- 4 question types per topic
- 20 factual supported questions
- 20 false-premise questions
- 20 multi-source nuanced questions
- 20 low-evidence policy questions

This set should be treated as final evaluation data. Do not tune prompts, retrieval expansion, detector thresholds, or answer-quality rules by inspecting individual failures from this file. If tuning is needed, use the older 40-question set as development data and keep this file for final reporting.

## Final Comparison Table

Recommended systems:

| System | Purpose |
| --- | --- |
| Base RAG | Measures generation/retrieval without detector or answer-quality controls |
| RAG + Detector | Measures the trained answer-include detector effect |
| RAG + Detector + Answer Quality | Main proposed system |
| RAG + Detector + Stochastic | Optional uncertainty-enhanced variant |

Recommended metrics:

- expected behavior match rate
- answer rate
- abstain rate
- detector run rate
- forbidden claim hit rate
- mean expected point coverage
- mean answer completeness score
- answer quality rewrite rate
- mean latency

## Why Local Tests Were Slow

Qwen 2.5 3B is lightweight compared with 7B/14B/70B models, but the full benchmark is not just one Qwen call per question.

The stable local configuration is slow because:

1. The RTX 3090 run hit CUDA instability when Qwen, ModernBERT detector, embedding, and reranker shared CUDA in one long process.
2. To make long runs complete reliably, the stable config keeps Qwen on CUDA but moves detector, embedder, and cross-encoder reranker to CPU.
3. Query expansion can run retrieval 2-3 times per question.
4. Cross-encoder reranking scores many candidate chunks per question.
5. The answer-quality layer may trigger a second Qwen generation for incomplete answers.
6. The detector checks the final answer against multiple retrieved contexts.
7. `use_cache: false` and `attn_implementation: eager` improve stability but slow generation.
8. Long prompts with 8-10 retrieved chunks are much slower than short chat prompts.

So the runtime is dominated by full RAG orchestration, not by Qwen alone.

## Practical Run Strategy

For a report-ready comparison, run fewer expensive variants first:

1. Base RAG on all 80 questions.
2. RAG + Detector on all 80 questions.
3. RAG + Detector + Answer Quality on all 80 questions.
4. Stochastic only if time permits, or on a smaller diagnostic subset.

Use the holdout file with:

```powershell
.\.venv\Scripts\python.exe scripts\run_finreg_real_life_benchmark.py `
  --mode full-rag `
  --questions benchmarks\finreg\final_holdout_80_questions.jsonl `
  --config gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank_quality `
  --k 24 `
  --run-name final_holdout80_quality
```

For faster smoke checks, use `--limit 8` or `--limit 20`, but do not report those as final results.
