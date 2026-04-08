# Current FinReg Baseline

This document summarizes the **current working FinReg baseline** after corpus finalization on
the real phase-1.5 prudential / supervisory corpus.

Use this as the primary reference for the active FinReg stack.

## Canonical Config

- config: `config/gating_finreg_ebcar_logit_mi_sc009.yaml`

## Active Stack

- domain: prudential / supervisory finreg
- corpus: real phase-1.5 prudential / supervisory regulatory corpus
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

Current corpus characteristics:

- `31` official source documents
- source families: `BCBS`, `EBA`, `ECB`, `PRA/BoE`, selected `Fed`
- section-aware index rebuilt from real source material
- current `rag_finreg` collection size: about `2221` chunks

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

- `evaluation_results/auto_eval/finreg_phase15_refined_v2_default_seed7.json`
- `evaluation_results/auto_eval/finreg_phase15_refined_v2_default_seed11.json`
- `evaluation_results/auto_eval/finreg_phase15_refined_v2_default_seed19.json`

Observed:

- `seed7`: `abstain_rate = 0.35`
- `seed11`: `abstain_rate = 0.30`
- `seed19`: `abstain_rate = 0.25`
- mean abstain rate: `0.30`

### 50Q refined v2

Files:

- `evaluation_results/auto_eval/finreg_phase15_refined_v2_50_default_seed7.json`
- `evaluation_results/auto_eval/finreg_phase15_refined_v2_50_default_seed11.json`
- `evaluation_results/auto_eval/finreg_phase15_refined_v2_50_default_seed19.json`

Observed:

- `seed7`: `abstain_rate = 0.28`
- `seed11`: `abstain_rate = 0.22`
- `seed19`: `abstain_rate = 0.28`
- mean abstain rate: `0.26`

Action mix on the 50Q set is stable:

- `none`: about `19-31`
- `retrieve_more`: about `8-17`
- `abstain`: about `11-14`

## What This Means

The current FinReg baseline is now:

- built on real regulatory evidence rather than synthetic notes
- stable across `3` seeds on both `20Q` and `50Q`
- intentionally somewhat conservative
- using `retrieve_more` meaningfully instead of collapsing to either always-answer or always-abstain

This should be treated as a **defensible working baseline**, not a final optimized endpoint.

## Immediate Next Step

The corpus is now in a practical stopping range for this project. The next work should focus on:

1. question-set maturity and taxonomy
2. retrieval / reranking maturity
3. generator baseline clarity
4. only then detector / gate ablations
