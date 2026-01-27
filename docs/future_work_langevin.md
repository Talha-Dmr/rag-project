# Future Work: Langevin-Based Uncertainty for Adaptive RAG

## Goal
Implement a Langevin-based uncertainty mechanism for the RAG system, accepting the higher
complexity in exchange for stronger epistemic uncertainty estimates and a more rigorous
Bayesian framing.

## What We Agreed On (Summary)
- Full Langevin is academically strong but practically expensive and brittle.
- We still want to attempt true Langevin, while recognizing clear failure modes.
- There is a real chance it does not converge or does not fit time/compute constraints.
- Risk can be reduced with staged validation and constrained parameter updates
  (e.g., LoRA/adapters instead of full model).

## Key Risks (Why It Might Fail)
1. Compute/time limits: parameter-space Langevin on 7B+ models is extremely expensive.
2. Gradient access: requires open-source model and full training graph.
3. Convergence/mixing: step size and noise can cause divergence or poor sampling.
4. Energy function design: a weak or ill-posed energy destroys sampling quality.
5. Evaluation: uncertainty improvements may not appear in ECE/coverage metrics.

## Risk Mitigations
- Start with a small model and toy task to validate the Langevin loop.
- Use constrained parameter updates (LoRA/adapters) before full model weights.
- Define explicit acceptance criteria and stop early if criteria fail.
- Keep a parallel proxy baseline (temperature ensemble / self-consistency) for comparison.

## Proposed Staged Approach
Stage 0: Baseline + Evaluation
- Establish current RAG + hallucination detector metrics.
- Add uncertainty proxy baselines for comparison (temperature ensemble, self-consistency).

Stage 1: Toy Langevin Proof-of-Concept
- Implement Langevin on a small model and small task.
- Verify mixing and stability on a controlled problem.

Stage 2: Constrained Langevin (LoRA/Adapters)
- Apply Langevin updates only on LoRA/adapters.
- Measure calibration and coverage changes against baselines.

Stage 3: Integrate with Adaptive Gating
- Use Langevin-based uncertainty as a gating signal
  (answer vs retrieve-more vs abstain).
- Validate end-to-end behavior on AmbigQA-style queries.

## Success Criteria (Draft)
- Langevin sampler is stable (no divergence) and produces non-degenerate samples.
- On a controlled dataset, ECE improves vs proxy baselines.
- Gating decisions reduce hallucination without unacceptable coverage loss.

## Open Questions
- Available GPU and max runtime per experiment?
- Target model size for the first non-toy experiment?
- Hard success thresholds (ECE, F1, hallucination rate) for go/no-go?

## Progress So Far (January 2026)
### Data + Setup
- Prepared AmbigQA-only NLI mini dataset:
  - `data/training/nli_dataset_ambigqa_mini` (train 30k, val 3k, test 1k).
- Added LoRA support in model loader (PEFT) and enabled training of specific modules.

### New/Updated Configs
- `config/sgld_lora_pilot_ambigqa_mini.yaml` (LoRA + SGLD, mini dataset).
- `config/adamw_lora_sanity_ambigqa_mini.yaml` (LoRA + AdamW sanity check, mini dataset).
- Existing pilots for comparison remain:
  - `config/adamw_pilot_ambigqa_mini.yaml`
  - `config/sgld_pilot_ambigqa_mini.yaml`

### Code Changes (Key)
- `src/training/utils/model_utils.py`
  - Added `modules_to_save` passthrough for LoRA to keep `classifier`/`pooler` trainable.
- `scripts/run_sgld_lora_sweep.py`
  - Small grid sweep for LoRA + SGLD on mini dataset.

### Sanity Result (LoRA + AdamW)
Run:
- `PYTHONPATH=. venv/bin/python scripts/train_hallucination_model.py --config adamw_lora_sanity_ambigqa_mini --data-dir data/training/nli_dataset_ambigqa_mini --output-dir models/checkpoints/adamw_lora_sanity_ambigqa_mini`

Metrics (val):
- Accuracy 0.7460
- F1 (macro) 0.7100
- ECE 0.0207
- Brier 0.2889

Interpretation:
- LoRA + classifier/pooler training works and learns properly.

### SGLD LoRA Sweep (Small Noise)
Run:
- `PYTHONPATH=. venv/bin/python scripts/run_sgld_lora_sweep.py --data-dir data/training/nli_dataset_ambigqa_mini`

Results file:
- `evaluation_results/sgld_lora_sweep_20260122_141921/sweep_results.json`

Outcome:
- All runs collapsed to single-class predictions (accuracy ≈ 0.33, F1_macro ≈ 0.165).
- Indicates SGLD still unstable/degenerate even with smaller noise.

### Prior Sweep (Higher Noise, Pre-`modules_to_save`)
Results file:
- `evaluation_results/sgld_lora_sweep_20260122_140457/sweep_results.json`

Outcome:
- Collapsed to single-class predictions (mostly contradiction).
- Motivated enabling `modules_to_save` and lowering noise.

### Current Hypothesis
- SGLD needs warm-start from a well-trained LoRA model and significantly smaller noise.
- Next step is warm-start SGLD from AdamW LoRA checkpoint:
  - `models/checkpoints/adamw_lora_sanity_ambigqa_mini/checkpoint-step-1500`
