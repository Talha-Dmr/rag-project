# Config Guide

There are many YAML files here because the repository contains several research
phases. Do not pick a config by filename search alone.

## Current FinReg Configs

- `gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank_quality.yaml`
  - Main quality-oriented local FinReg RAG config.
  - Uses the real section-aware FinReg corpus/index.
  - Uses local Qwen 3B generation, cross-encoder reranking, ModernBERT v3 hardmix detector, and answer-quality logic.
  - Heavier than the RTX2070 smoke configs.

- `gating_finreg_local_qwen15_rtx2070_section_rerank_smoke.yaml`
  - Small RTX2070-safe full-RAG smoke.
  - Detector/gating disabled in config.
  - Useful for checking retrieval + reranking + local Qwen generation.

- `gating_finreg_local_qwen15_rtx2070_section_rerank_detector_smoke.yaml`
  - Small RTX2070-safe integration smoke.
  - Detector and abstain gate enabled.
  - Answer-quality rewrite disabled to keep GPU/RAM use controlled.

- `gating_finreg_modernbert_detector_v3_hardmix_calibrated.yaml`
  - Controlled detector benchmark config.
  - Useful when evaluating fixed candidate answers rather than full RAG generation.

## Historical / Experimental Groups

- `gating_finreg_ebcar_*`: earlier EBCAR and uncertainty-gating experiments.
- `gating_finreg_openrouter_*`: OpenRouter experiments; not part of the current local path.
- `sgld_*`: SGLD / Langevin detector experiments.
- `adamw_*`: detector training configs.
- `finregbench_*`: ModernBERT detector training/evaluation configs.

## Rule Of Thumb

For current local FinReg work:

1. Use the RTX2070 smoke configs for quick local checks.
2. Use the `qwen3 ... section_rerank_quality` config for report-quality retrieval/full-RAG checks on a stronger machine.
3. Use controlled detector configs only when the benchmark fixes the candidate answer.
