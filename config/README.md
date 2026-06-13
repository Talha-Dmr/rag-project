# Config Guide

There are many YAML files because the repository contains several research
phases. Do not pick a config by filename search alone.

## Final Targeted Benchmark Configs

Use these for the current 160-question final benchmark:

- `final_finreg_qwen3b_rag.yaml`
  - Baseline final full-RAG run.
  - Uses local `Qwen/Qwen2.5-3B-Instruct`.
  - Detector and stochastic evidence sampling are not the main comparison
    layer in this variant.

- `final_finreg_qwen3b_detector.yaml`
  - Final full-RAG run with the ModernBERT hardmix detector and abstain gate.
  - This is the main deterministic safety comparison against baseline RAG.

- `final_finreg_qwen3b_detector_stochastic.yaml`
  - Final full-RAG run with detector, abstain gate, and stochastic evidence
    subset sampling.
  - This is the current best audited system variant.

The matching benchmark input is:

`benchmarks/finreg/full_rag_questions_final_targeted160.jsonl`

## DeepSeek API Configs

These keep the FinReg retrieval/gating setup but move generation to the
DeepSeek API. They are useful when local GPU memory is the bottleneck.

- `gating_finreg_deepseek_v4_flash_no_evidence_sampling_coverage_quality.yaml`
  - DeepSeek V4 Flash generation without stochastic evidence sampling.

- `gating_finreg_deepseek_v4_flash_evidence_vector_v3_coverage_quality.yaml`
  - DeepSeek V4 Flash generation with vector stochastic evidence sampling.

Both expect `DEEPSEEK_API_KEY` in the environment.

## Local Smoke Configs

- `gating_finreg_local_qwen15_rtx2070_section_rerank_smoke.yaml`
  - Small RTX2070-safe retrieval + generation smoke.
  - Detector/gating disabled in config.

- `gating_finreg_local_qwen15_rtx2070_section_rerank_detector_smoke.yaml`
  - Small RTX2070-safe detector/gating integration smoke.
  - Answer-quality rewrite is disabled to keep GPU/RAM use controlled.

## Older Quality / Development Configs

- `gating_finreg_local_qwen3_modernbert_detector_v3_hardmix_calibrated_real_corpus_section_rerank_quality.yaml`
  - Earlier quality-oriented local FinReg RAG config.
  - Still useful for comparison, but the final report path now uses the
    `final_finreg_qwen3b_*` configs.

- `gating_finreg_modernbert_detector_v3_hardmix_calibrated.yaml`
  - Controlled detector benchmark config.
  - Useful when evaluating fixed candidate answers rather than full RAG
    generation.

## Historical / Experimental Groups

- `gating_finreg_ebcar_*`: earlier EBCAR and uncertainty-gating experiments.
- `gating_finreg_openrouter_*`: OpenRouter experiments.
- `sgld_*`: SGLD / Langevin detector experiments.
- `adamw_*`: detector training configs.
- `finregbench_*`: ModernBERT detector training/evaluation configs.
- `gating_finreg_lfm2_*`, `gating_finreg_granite33_*`,
  `gating_finreg_smollm3_*`, and `gating_finreg_phi4_*`: model exploration and
  hardware-specific experiments.

## Rule Of Thumb

1. For final benchmark claims, use `full_rag_questions_final_targeted160.jsonl`
   with the `final_finreg_qwen3b_*` configs.
2. For local quick checks, use the RTX2070 smoke configs with `--limit`.
3. For lower local VRAM pressure, use the DeepSeek configs.
4. For detector-only questions, use controlled detector configs instead of
   full-RAG configs.
