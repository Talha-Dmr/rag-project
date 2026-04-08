# Current FinReg Baseline

This document summarizes the **current working FinReg baseline** after the move from the
synthetic bootstrap corpus to the real phase-1 corpus.

Use this as the primary reference for the active FinReg stack.

## Canonical Config

- config: `config/gating_finreg_ebcar_logit_mi_sc009.yaml`

## Active Stack

- domain: prudential / supervisory finreg
- corpus: real phase-1 regulatory corpus
- chunking: `section_aware`
- retrieval first-pass `k`: `20`
- reranker: `ebcar`
- family balancing: enabled, with named-regulator balancing in final context selection
- generator: local `Qwen/Qwen2.5-1.5B-Instruct`
- detector: FEVER `DeBERTa-v3-base`
- gating strategy: `retrieve_more`
- epistemic signal: `logit_mi`

## Corpus

Primary processed corpus:

- `data/processed/finreg/finreg_phase1_corpus.jsonl`

Raw source families:

- `BCBS`
- `EBA`
- `PRA/BoE`
- `ECB`

Current phase-1 corpus characteristics:

- `18` official source documents
- section-aware index rebuilt from real source material
- current `rag_finreg` collection size during the section-aware runs: about `1380` chunks

## Current Question Sets

Primary working seed set:

- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2.jsonl`

Primary working expanded set:

- `data/domain_finreg/questions_finreg_conflict_phase1_refined_v2_50.jsonl`

Interpretation:

- `20Q refined v2` is the current hand-curated phase-1-aligned working set
- `50Q refined v2` is the current template-expanded confirmation set built from that seed

## Current Results

### 20Q refined v2

Files:

- `evaluation_results/auto_eval/finreg_phase1_refined_v2_default_seed7.json`
- `evaluation_results/auto_eval/finreg_phase1_refined_v2_default_seed11.json`
- `evaluation_results/auto_eval/finreg_phase1_refined_v2_default_seed19.json`

Observed:

- `seed7`: `abstain_rate = 0.25`
- `seed11`: `abstain_rate = 0.30`
- `seed19`: `abstain_rate = 0.25`
- mean abstain rate: `0.2667`

### 50Q refined v2

Files:

- `evaluation_results/auto_eval/finreg_phase1_refined_v2_50_default_seed7.json`
- `evaluation_results/auto_eval/finreg_phase1_refined_v2_50_default_seed11.json`
- `evaluation_results/auto_eval/finreg_phase1_refined_v2_50_default_seed19.json`

Observed:

- `seed7`: `abstain_rate = 0.30`
- `seed11`: `abstain_rate = 0.28`
- `seed19`: `abstain_rate = 0.28`
- mean abstain rate: `0.2867`

Action mix on the 50Q set is stable:

- `none`: about `23-24`
- `retrieve_more`: about `11-13`
- `abstain`: about `14-15`

## What This Means

The current FinReg baseline is now:

- built on real regulatory evidence rather than synthetic notes
- stable across `3` seeds on both `20Q` and `50Q`
- intentionally somewhat conservative
- using `retrieve_more` meaningfully instead of collapsing to either always-answer or always-abstain

This should be treated as a **defensible working baseline**, not a final optimized endpoint.

## Immediate Next Step

The next improvement should focus on:

1. question tagging for the refined v2 sets
2. targeted retrieval/reranking improvements for the remaining hard comparison questions
3. only then, further corpus expansion if the hard cases still reflect evidence coverage gaps
