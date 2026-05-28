# Current Status

Last updated: 2026-05-29

## Active Goal

Build and evaluate a FinReg RAG pipeline where retrieval, generation,
hallucination detection, and stochastic/abstain gating can be tested separately
and together.

## Corpus

- Scope: prudential / supervisory FinReg.
- Sources: official BCBS, EBA, ECB, PRA-BoE, and selected Fed/OCC documents.
- Current processed corpus: `data/processed/finreg/finreg_phase1_corpus.jsonl`.
- Current section-aware vector index: `data/vector_db/domain_finreg_real_section`.
- Current Chroma collection: `rag_finreg_real_section`.

## Current Test Dataset

The canonical test dataset is now:

- `benchmarks/finreg/full_rag_questions.jsonl`

This is the 80-question FullRAG test set. Smaller 50-question files may still be used for quick
debugging, replay, or smoke checks, but they are no longer the primary test dataset.

Runtime code must not use benchmark-only metadata from the 80-question file, including ids,
`expected_behavior`, expected answer points, forbidden claims, or question-type labels.

## Hardware Placement Rule

Prefer GPU execution for the LLM and detector when VRAM can support it. Do not move the detector to
CPU merely for convenience, because CPU detector runs are much slower and distort iteration time.

CPU is acceptable only as an explicit fallback when GPU memory is insufficient, unstable, or needed
for a specific isolation test. Any CPU fallback should be called out in the run notes.

## Current Local Smoke Results

RTX2070 current-stack generator smoke results on the canonical FullRAG80 first 10 questions:

- Qwen2.5-1.5B current section+rerank detector smoke:
  - run: `smoke10_qwen15_bf16_rtx2070_section_rerank_detector_gate`
  - expected behavior match rate: `0.900`
  - abstain rate: `0.400`
  - forbidden claim hit count: `0`
  - mean latency: `73.6s`
- SmolLM3-3B current section+detector smoke:
  - run: `smoke10_smollm3_3b_rtx2070_section_detector`
  - config: `gating_finreg_smollm3_3b_local_rtx2070_section_detector_smoke`
  - expected behavior match rate: `0.800`
  - abstain rate: `0.100`
  - forbidden claim hit count: `1`
  - mean latency: `4.33s`
  - note: stable on CUDA, but leaked `<think>` markers in answers.
- Granite 3.3 2B current section+detector smoke:
  - run: `smoke10_granite33_2b_rtx2070_section_detector`
  - config: `gating_finreg_granite33_2b_rtx2070_section_detector_smoke`
  - expected behavior match rate: `0.900`
  - abstain rate: `0.200`
  - forbidden claim hit count: `0`
  - mean latency: `4.52s`
  - note: stable on CUDA and currently the cleanest RTX2070 fallback candidate.

Granite 3.3 2B with the original RTX2070 smoke runtime did not remain stable when extended to a
20-question CUDA smoke:

- run: `smoke20_granite33_2b_rtx2070_section_detector`
- completed before crash: `10/20`
- partial expected behavior match rate: `0.900`
- partial abstain rate: `0.200`
- partial forbidden claim hit count: `0`
- failure: CUDA `illegal instruction` during generation on question `fullrag_011`

This means model replacement alone has not fixed the long-process RTX2070 CUDA generation
instability. The same failure class has now appeared with LFM2, Qwen2.5-3B, and Granite.
The local stack at the time of testing was `torch 2.6.0+cu124`, `transformers 5.0.0`,
NVIDIA driver `595.71.05`, and RTX 2070 compute capability `7.5`.

The current stable Granite RTX2070 runtime is:

- Python env: `venv312`
- torch: `2.6.0+cu124`
- transformers: `4.49.0`
- tokenizers: `0.21.4`
- huggingface-hub: `0.36.2`
- config: `gating_finreg_granite33_2b_rtx2070_section_detector_noempty_smoke`
- key runtime change: keep `use_cache: true`, but set `clear_cuda_cache_after_generate: false`

Validation after the runtime change:

- `smoke20_granite33_2b_rtx2070_section_detector_tf449_noempty`: completed `20/20`
- `smoke40_granite33_2b_rtx2070_section_detector_tf449_noempty`: completed `40/40`
- `fullrag80_granite33_2b_rtx2070_section_detector_tf449_noempty`: completed `80/80`

The failed isolation checks were useful but should not be used as the main path:

- `use_cache: false` was much slower and still crashed with CUDA `misaligned address`.
- `transformers 4.49.0` plus the original per-generation `empty_cache` still crashed in a 20Q run
  with CUDA `invalid program counter`.

Granite CPU fallback was smoke-tested:

- run: `smoke2_granite33_2b_cpu_section_detector`
- config: `gating_finreg_granite33_2b_cpu_section_detector_smoke`
- expected behavior match rate: `1.000`
- forbidden claim hit count: `0`
- mean latency: `66.2s`
- note: stable in the 2-question smoke but too slow for fast iteration.

Qwen2.5-3B was also smoke-tested with the current quality/rewrite config, but failed during
answer-quality rewrite generation with `CUDA error: misaligned address`. It should not be used for
long RTX2070 runs unless generation is further isolated or moved off the unstable CUDA path.

The latest RTX2070 detector/gating smoke used:

- config: `gating_finreg_local_qwen15_rtx2070_section_rerank_detector_smoke`
- benchmark: `benchmarks/finreg/full_rag_questions.jsonl`
- limit: `10`
- answer-quality rewrite: disabled

Observed summary:

- detector run rate: `1.0`
- abstain rate: `0.4`
- expected behavior match rate: `0.9`
- forbidden claim hit rate: `0.0`
- mean answer-include risk: `0.677`

Report path:

- `reports/finreg_real_life_benchmark/smoke10_qwen15_bf16_rtx2070_section_rerank_detector_gate/summary.json`

## Latest FullRAG80 Results

Latest completed FullRAG80 run:

- run: `fullrag80_granite33_2b_rtx2070_section_detector_tf449_noempty`
- config: `gating_finreg_granite33_2b_rtx2070_section_detector_noempty_smoke`
- benchmark: `benchmarks/finreg/full_rag_questions.jsonl`
- total: `80`
- expected behavior match rate: `0.8125` (`65/80`)
- current rubric replay expected behavior match rate: `0.900` (`72/80`)
- abstain rate: `0.1125` (`9/80`)
- answer rate: `0.8875` (`71/80`)
- forbidden claim hit rate: `0.0`
- mean expected point coverage: `0.5706`
- answer-quality rewrite rate: `0.0`
- mean latency: `5.22s`

Report path:

- `reports/finreg_real_life_benchmark/fullrag80_granite33_2b_rtx2070_section_detector_tf449_noempty/summary.json`

This is the first current-stack Granite RTX2070 FullRAG80 run completed in one pass without CUDA
crash. The original report was written before the evaluator gained broader benchmark-independent
refutation/caution markers, so its stored `summary.json` shows `65/80`. Replaying the same answers
with the current evaluator gives `72/80`. LFM2 remains `66/80` under the same replay, so the
evaluator change did not lift the LFM2 baseline.

Best LFM2 FullRAG80 run so far:

- run: `fullrag80_lfm2_guarded_v3_quality_guard_relaxed_escape_v2_cpu_detector`
- config: `gating_finreg_lfm2_26b_local_rtx2070_evidence_retry_cpu_detector_quality_guard_relaxed_escape`
- benchmark: `benchmarks/finreg/full_rag_questions.jsonl`
- total: `80`
- expected behavior match rate: `0.825` (`66/80`)
- abstain rate: `0.475` (`38/80`)
- answer rate: `0.525` (`42/80`)
- forbidden claim hit rate: `0.0`
- mean expected point coverage: `0.3590`
- answer-quality rewrite rate: `0.525` (`42/80`)

Report path:

- `reports/finreg_real_life_benchmark/fullrag80_lfm2_guarded_v3_quality_guard_relaxed_escape_v2_cpu_detector/summary.json`

This run used LFM2 on CUDA with detector/logit sampling on CPU as a deliberate stability fallback.
GPU detector placement was tested, but full runs failed with CUDA misaligned-address or illegal
instruction errors. LFM2 CUDA generation can still intermittently fail, so the stable runner uses
`--resume` and retries from `per_question.partial.jsonl` instead of restarting the benchmark.
This run had two LFM2 CUDA crashes and resumed successfully.

The previous quality-guard baseline remains:

- run: `fullrag80_lfm2_guarded_v3_relaxed_answer_quality_guard_cpu_detector`
- config: `gating_finreg_lfm2_26b_local_rtx2070_evidence_retry_cpu_detector_relaxed_answer_quality_guard`
- expected behavior match rate: `0.8125` (`65/80`)
- abstain rate: `0.525` (`42/80`)
- forbidden claim hit rate: `0.0`

The less conservative comparison baseline remains:

- run: `fullrag80_lfm2_guarded_v3_relaxed_answer_cpu_detector`
- config: `gating_finreg_lfm2_26b_local_rtx2070_evidence_retry_cpu_detector_relaxed_answer`
- expected behavior match rate: `0.775` (`62/80`)
- abstain rate: `0.45` (`36/80`)
- forbidden claim hit rate: `0.0`

## Known Current Issues

- Granite 3.3 2B is now the preferred fast local RTX2070 model for current-stack checks.
- Qwen 1.5B generated degenerate repeated punctuation with fp16 on the local
  machine, so any Qwen 1.5B smoke config should keep using `dtype: bfloat16`.
- The latest Granite FullRAG80 run has `15` automatic mismatches: `5` factual-supported,
  `5` false-premise, `3` low-evidence-policy, and `2` multi-source-nuanced rows.
- After the evaluator marker/token fix, the Granite run has `8` remaining replay mismatches:
  `fullrag_015`, `fullrag_017`, `fullrag_018`, `fullrag_021`, `fullrag_022`, `fullrag_023`,
  `fullrag_025`, and `fullrag_073`.
- Granite answer-quality rewrite improved the first-16 smoke score to `15/16`, but it crashed with
  CUDA `illegal instruction` once extra rewrite generations accumulated. Do not use rewrite-enabled
  Granite CUDA as the main path yet.
- Granite `max_tokens: 128` without rewrite completed a 20Q smoke, but did not improve the first-20
  score versus the stable `max_tokens: 96` run.
- Granite `max_prompt_tokens: 1536` plus `max_tokens: 128` reached `40/80` with a promising
  partial score, but then crashed with CUDA `illegal instruction`. Do not use the larger prompt
  window as the main RTX2070 path.
- Granite `--k 8` and query-expansion FullRAG80 attempts also crashed in long local CUDA runs.
  Query expansion improved first-25 score from `18/25` to `21/25`, so the retrieval idea is useful,
  but it should be validated on CPU, remote/API, or an isolated process path rather than another
  long-lived RTX2070 CUDA process.
- CPU validation config for that path: `gating_finreg_granite33_2b_cpu_section_detector_queryexp`.
- A broader prompt-wording change on the stable `1024/96` runtime was stopped after `33/80`
  because it regressed old pass cases. Keep the original prompt for now.
- The best LFM2 FullRAG80 run improved automatic expected-behavior match versus the quality-guard
  baseline by adding a benchmark-independent high-quality relaxed escape after exhausted
  `retrieve_more`.
- LFM2 mismatch breakdown: `14` mismatches total; `7` factual-supported, `5`
  multi-source-nuanced, `1` false-premise, and `1` low-evidence-policy row.
- Of those LFM2 mismatches, `6` were answered rows that still failed expected behavior; `8` were
  false abstains after `retrieve_more`.
- The LFM2 relaxed escape reduced abstain from `52.5%` to `47.5%`, but the net gain is small:
  `3` rows improved and `2` rows regressed versus the quality-guard baseline.
- Large local artifacts such as model checkpoints, vector stores, reports, and
  detector asset folders are not expected to exist on a fresh clone.

## Next Work

1. Use FullRAG80 as the main test dataset for gating and model comparisons.
2. Keep 50Q runs only for fast smoke/debug/replay passes.
3. Use the stable Granite RTX2070 runtime only for reproducing the known baseline or short smoke
   checks:
   `transformers==4.49.0`, `tokenizers==0.21.4`, and
   `clear_cuda_cache_after_generate: false`.
4. Do not use local RTX2070 CUDA for long exploratory FullRAG80 runs that change retrieval, prompt
   size, rewrite behavior, or context shape. Use CPU, remote/API, or an isolated process path for
   those.
5. Prefer GPU detector placement when VRAM allows; document CPU fallback explicitly. Current LFM2
   FullRAG80 work uses CPU detector because GPU detector placement was not stable on the RTX2070.
6. Keep the Granite CPU fallback for stability validation when local CUDA keeps failing. Use
   `gating_finreg_granite33_2b_cpu_section_detector_queryexp` for the next no-CUDA validation of
   the query-expansion candidate.
7. Next gating work should harden the relaxed escape so it keeps the false-abstain wins without
   regressing borderline answered/abstained rows.
8. Keep corpus changes targeted; the current corpus is sufficient for detector
   and gating research unless a specific hard case exposes a missing document.
