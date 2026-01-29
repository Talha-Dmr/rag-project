# Direction Memo: Differentiation + Advancement

## Goal
Stay differentiated from generic RAG while pushing uncertainty forward.

## Pillar A — Posterior-Approx at Adapter Level
- Focus on LoRA-level posterior approximations (LoRA-SWAG, MC Dropout, small ensembles).
- Claim approximate Bayesian grounding (not just heuristic sampling).

## Pillar B — Conflict-Aware Decision Policy
- Explicit triage: answer / retrieve_more / abstain.
- Optimize coverage vs risk (cost-aware retrieval policy).

## Pillar C — Conflict-Focused Evaluation
- Use energy + macro as conflict-heavy domains.
- Report conflict-specific metrics (abstain_on_conflict, wrong_on_conflict, resolve_rate).

## Near-Term Execution
- Prototype LoRA-SWAG and MC Dropout baselines.
- Compare against SGLD warm-start results.
- Produce coverage-risk curves and ablation table.

## Notes from Posterior_Sampling_Algorithms_for_Epistemic_Uncertainty-1.pdf (draft)
- Gradient-based posterior sampling is the theoretical anchor; overdamped Langevin is the most feasible.
- HMC/NUTS are described as theoretical upper bounds but impractical at LLM scale.
- MALA reduces discretization bias but adds accept/reject overhead.
- SGLD is treated as a practical training-time Bayesian approximation.
- Proxy methods (temperature ensemble, self-consistency, MC Dropout, deep ensembles) give moderate signals but are not true posterior sampling.
- Action: validate these claims with literature before stating strongly.

## Sourced Framing (short list)
- SGLD: posterior-sampling view via noisy SGD with annealed steps (Welling & Teh, 2011).
- MC Dropout: approximate Bayesian inference interpretation (Gal & Ghahramani, 2016).
- Deep Ensembles: non-Bayesian alternative with strong uncertainty (Lakshminarayanan et al., 2017).
- Self-consistency: multi-sample decoding for reasoning, not explicit posterior sampling (Wang et al., 2022).

References:
- https://www.maths.ox.ac.uk/node/24560
- https://proceedings.mlr.press/v48/gal16.html
- https://arxiv.org/abs/1612.01474
- https://arxiv.org/abs/2203.11171

## Research Queue
- New posterior-approx methods for adapters.
- LLM-specific uncertainty calibration papers.
- Recent RAG gating or abstention policy work.
- Validate “proxy vs true epistemic” claims with citations; decide how strongly to state.
- Extract actionable ideas from Posterior_Sampling_Algorithms_for_Epistemic_Uncertainty-1.pdf.
