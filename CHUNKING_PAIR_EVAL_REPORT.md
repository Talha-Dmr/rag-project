# Chunking Pair Evaluation Report

- Date: 2026-04-09
- Scope: compare `semantic_mpnet` vs `section_aware` for finreg retrieval + grounding
- Benchmark: `output/benchmarks/benchmark_finreg_retrieval_50.json`
- Question set: `data/domain_finreg/questions_finreg_conflict_50.jsonl`

## Why This Was Run

We previously compared several chunking variants using retrieval metrics only and treated
fixed-size chunking as the baseline. That was incorrect for this project track because the
intended default chunking family was semantic chunking, not fixed-size.

This follow-up evaluation compares two stronger candidates directly:

- `semantic_mpnet`
- `section_aware`

The target metrics were:

- `abstain_rate`
- `answered_contradiction_rate`
- `unsupported_answer_rate`
- `coverage`
- `doc@10`
- `chunk@10`
- `chunk@MRR`

## Implementation Work Added

The following files were added or updated to support this evaluation:

- `scripts/run_chunking_pair_eval.py`
- `src/rag/hallucination_detector.py`
- `config/gating_finreg_ebcar_logit_mi_sc009_localdet.yaml`

Main implementation notes:

- Added a dedicated pair-eval orchestration script for chunking comparison.
- Added progress output and resume-friendly behavior for long semantic chunking runs.
- Fixed hallucination detector label mapping so FEVER-style exported labels can be used.
- Added offline-safe local detector config using local cache paths for:
  - the detector tokenizer base model
  - the embedding model

## Detector Issue And Fix

The first grounding run was not usable because the detector never loaded.

Observed issue:

- Config pointed to a nonexistent detector checkpoint:
  - `models/checkpoints/sgld_lora_warmstart_ambigqa_mini_noise5e-5_balanced/best_model`
- As a result, the run produced:
  - `Hallucination detection requested but detector not available`

Fix applied:

- Switched to an existing local exported model:
  - `src/electra_daberta/final_fever_deberta_v3_base_model`
- Updated detector loading logic to support FEVER-style labels:
  - `SUPPORTS -> entailment`
  - `REFUTES -> contradiction`
  - `NOT ENOUGH INFO -> neutral`
- Added offline tokenizer fallback through the local Hugging Face cache.

## Final Summary Output

Current generated summary:

- `evaluation_results/chunking_pair_eval/summary/comparison.md`
- `evaluation_results/chunking_pair_eval/summary/grounding_comparison.csv`
- `evaluation_results/chunking_pair_eval/summary/retrieval_comparison.csv`

## Final Results

### Grounding Metrics

| Strategy | Abstain | Coverage | Answered Contradiction | Unsupported Answer |
| --- | ---: | ---: | ---: | ---: |
| semantic_mpnet | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| section_aware | 0.0000 | 1.0000 | 0.0000 | 0.0000 |

### Retrieval Metrics

| Strategy | Method | Doc@10 | Chunk@10 | Chunk@MRR |
| --- | --- | ---: | ---: | ---: |
| semantic_mpnet | bm25 | 0.8800 | 0.4200 | 0.2885 |
| semantic_mpnet | dense | 0.9200 | 0.4200 | 0.2072 |
| semantic_mpnet | hybrid | 0.9000 | 0.4400 | 0.2513 |
| semantic_mpnet | adaptive_or_stochastic | 0.9000 | 0.4200 | 0.2450 |
| section_aware | bm25 | 0.9400 | 0.4000 | 0.2467 |
| section_aware | dense | 0.9400 | 0.3000 | 0.1587 |
| section_aware | hybrid | 0.9600 | 0.3800 | 0.2569 |
| section_aware | adaptive_or_stochastic | 0.9600 | 0.3600 | 0.2513 |

## Interpretation

### What We Can Say With Confidence

- `semantic_mpnet` is better on chunk-level retrieval quality.
- `section_aware` is better on document-level recall.
- The strongest single chunk-level result in this run is:
  - `semantic_mpnet + bm25` on `chunk@MRR = 0.2885`
- `section_aware + dense` underperforms on chunk-level retrieval.

### What We Cannot Yet Claim

The grounding side did run successfully after the detector fix, but it did not produce
discriminative behavior:

- both strategies answered all 50 questions
- both produced `0.0` on answered contradiction and unsupported answer rates
- both produced `0.0` abstain

That means the current grounding setup is not separating the two chunking strategies in a
useful way on this question slice. This is not evidence that both strategies are equally
safe; it only means this detector + threshold combination is not producing differentiating
signals here.

## Current Recommendation

If we need a recommendation based on the current evidence:

- keep `semantic_mpnet` as the leading candidate for this track
- keep `section_aware` as a document-recall-friendly alternative, but not the default

Reason:

- this evaluation does not validate a grounding advantage for `section_aware`
- retrieval evidence is stronger for `semantic_mpnet` at the chunk level, which is more
  directly aligned with evidence quality in RAG

## Suggested Next Steps

1. Keep `semantic_mpnet` as the working default candidate for chunking experiments.
2. If grounding behavior matters for the final decision, run a stricter grounding follow-up:
   - sharper detector
   - stronger thresholds
   - manual inspection of sampled answers
3. If needed, promote this work to a PR with:
   - the new evaluation script
   - the detector loading fix
   - the offline local detector config
   - this report

## Working Tree Items Relevant For PR

- `CHUNKING_PAIR_EVAL_REPORT.md`
- `scripts/run_chunking_pair_eval.py`
- `src/rag/hallucination_detector.py`
- `config/gating_finreg_ebcar_logit_mi_sc009_localdet.yaml`
