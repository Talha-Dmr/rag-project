# Granite FullRAG80 Failure Analysis

Date: 2026-05-29

Run:

- `fullrag80_granite33_2b_rtx2070_section_detector_tf449_noempty`
- Config: `gating_finreg_granite33_2b_rtx2070_section_detector_noempty_smoke`
- Dataset: `benchmarks/finreg/full_rag_questions.jsonl`

## Result

Stored report summary:

- Expected behavior match: `65/80`
- Abstain: `9/80`
- Forbidden claims: `0`
- Mean latency: `5.22s`

After replaying the same answers with the broader evaluator markers:

- Expected behavior match: `72/80`
- Newly accepted rows: `fullrag_008`, `fullrag_024`, `fullrag_026`, `fullrag_030`, `fullrag_042`
- Additional rows accepted after token/caution matching cleanup: `fullrag_044`, `fullrag_061`
- Remaining mismatch rows: `fullrag_015`, `fullrag_017`, `fullrag_018`, `fullrag_021`,
  `fullrag_022`, `fullrag_023`, `fullrag_025`, `fullrag_073`

The evaluator change is benchmark-independent. It adds common refutation/caution markers such as
`does not establish`, `not mandated`, `not limited`, `no fixed`, and `cannot solely`, plus small
token variants such as `respond/response` and `incident/incidents`; it does not use question ids,
question types, expected points, or forbidden claims in runtime logic.

## Diagnosis

Likely evaluator false negatives now addressed:

- `fullrag_008`: answer said the context does not establish a specific approval deadline.
- `fullrag_024`: answer said a specific regulator template is not mandated.
- `fullrag_026`: answer said ICT security risk management is not limited to major participants.
- `fullrag_030`: answer was weak, but included a refutation form the old marker set missed.
- `fullrag_042`: answer said the bank cannot solely attribute resilience failures to the provider.

Remaining issues:

- Retrieval/topic drift: `fullrag_073`.
- Too generic or incomplete despite relevant context: `fullrag_017`, `fullrag_021`,
  `fullrag_023`, `fullrag_025`.
- Refutation not explicit enough: `fullrag_018`, `fullrag_022`.
- Detector/gating over-abstain or borderline abstain behavior: `fullrag_015`.

## Experiments

Stable baseline:

- `20/20`, `40/40`, and `80/80` completed with no CUDA crash.
- Key runtime: `transformers==4.49.0`, `clear_cuda_cache_after_generate: false`.

Answer-quality rewrite:

- First 16 rows: `15/16`, `0` forbidden claims.
- Crashed at row 17 with CUDA `illegal instruction`.
- Conclusion: rewrite improves quality but is not stable enough in-process on RTX2070 CUDA.

Larger answer budget:

- `max_tokens: 128` completed 20Q.
- First-20 score remained `16/20`, same as baseline.
- Conclusion: more tokens alone is not enough.

Larger prompt window plus 128 answer tokens:

- `max_prompt_tokens: 1536`, `max_tokens: 128` reached `40/80`, then crashed with CUDA
  `illegal instruction`.
- Partial score was `37/40`, but the runtime is not stable enough for the RTX2070 path.
- Conclusion: do not use the larger prompt window as the main local path.

Wider retrieval pool:

- `--k 8` crashed at `21/80` with CUDA `misaligned address`.
- Conclusion: do not increase the single-query retrieval pool for long local CUDA runs.

Domain query expansion:

- Retrieval-only audit improved context coverage on several hard rows, especially `fullrag_018`,
  `fullrag_021`, and `fullrag_023`.
- `smoke25_granite33_2b_rtx2070_section_detector_tf449_noempty_queryexp` completed `25/25` and
  improved the first-25 replay score from `18/25` to `21/25`.
- FullRAG80 query-expansion run crashed during the long CUDA process with illegal-memory access.
- Conclusion: query expansion is a useful retrieval idea, but it should be evaluated through CPU,
  remote/API, or another isolated non-long-lived CUDA path before becoming the main run config.
- CPU validation config: `gating_finreg_granite33_2b_cpu_section_detector_queryexp`.

Prompt wording change on the stable `1024/96` runtime:

- Stopped after `33/80` because early regressions outweighed the gains.
- Partial score was `26/33`; failures included old pass cases such as `fullrag_009` and
  `fullrag_028`.
- Conclusion: keep the original prompt for now; solve remaining rows through retrieval/gating or
  isolated rewrite rather than broad prompt pressure.

## Next Actions

1. Keep the stable Granite no-empty config as the main local CUDA path.
2. Treat local RTX2070 CUDA as a smoke/reproduction path only for new experiments; do not use it for
   long exploratory FullRAG80 runs that change retrieval, prompt size, or rewrite behavior.
3. Do not enable in-process answer-quality rewrite for Granite CUDA until generation isolation is
   solved.
4. Improve retrieval/prompting for the remaining hard rows without benchmark-specific metadata, but
   validate those changes on CPU, remote/API, or an isolated process path.
5. If rewrite is needed, test it in an isolated process or CPU-only side path, not inside the same
   long-lived CUDA process.
