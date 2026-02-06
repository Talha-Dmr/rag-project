# Future Work: Langevin-Based Uncertainty for Adaptive RAG

## Goal
Implement a Langevin-based uncertainty mechanism for the RAG system, accepting the higher
complexity in exchange for stronger epistemic uncertainty estimates and a more rigorous
Bayesian framing.

## Executive Summary (Current Snapshot)
- Main demo defaults (epistemic track):
  - **Energy**: logit‑MI gating (contradiction_rate_threshold=0.90).
  - **Macro**: logit‑MI gating (contradiction_rate_threshold=0.80).
- Latest rep-vs-logit ablation (50Q, seed=7):
  - Energy: logit‑MI 35/50 (0.70) vs rep‑MI 45/50 (0.90).
  - Macro: logit‑MI 22/50 (0.44) vs rep‑MI 44/50 (0.88).
  - Decision: keep logit‑MI as default and keep representation‑space MI experimental.
- Consistency‑only baseline (50 Q):
  - Energy: abstain 14/50 (0.28), actions none=36, retrieve_more=14.
  - Macro: abstain 9/50 (0.18), actions none=41, retrieve_more=9.
- Historical logit‑MI threshold sweep (older calibration run, 50 Q):
  - Energy:
    - threshold=0.70 → abstain 46/50 (0.92) → too conservative for default.
    - threshold=0.85 → abstain 22/50 (0.44) → improved, still higher than consistency‑only.
    - threshold=0.90 → abstain 21/50 (0.42) → selected for epistemic default.
  - Macro: abstain 12/50 (0.24) in that run.
- Safety variant (high abstain): MC Dropout + source-consistency (sc=0.50).
  - Energy: abstain 31/50 (0.62).
  - Macro: abstain 36/50 (0.72).
- LoRA-SWAG is kept as ablation only; n=5 vs n=10 did not outperform MC Dropout on macro.
- Entropy-based SWAG gating was unstable (all-abstain vs none-abstain depending on threshold).
- Final configs (epistemic default): `gating_energy_ebcar_logit_mi_sc009.yaml`, `gating_macro_ebcar_logit_mi_sc009.yaml`.
- Experimental configs (representation track): `gating_energy_ebcar_rep_mi_sc004.yaml`, `gating_macro_ebcar_rep_mi_sc004.yaml`.
- Coverage-first alternative: `gating_energy_ebcar_consistency_only_sc050.yaml`, `gating_macro_ebcar_consistency_only_sc050.yaml`.
- Safety variant: `gating_energy_ebcar_mcdropout_consistency_sc050.yaml`, `gating_macro_ebcar_mcdropout_consistency_sc050.yaml`.
- Latest gating ablation (Feb 1, 2026):
  - Energy set is 20Q (conflict-heavy); Macro is 50Q.
  - Results summarized below (nogate vs retrieve_more vs abstain).

### Gating Ablation (Feb 1, 2026)
Energy (20Q):
| Setting | abstain | abstain_rate | actions |
| --- | --- | --- | --- |
| nogate | 0/20 | 0.00 | none=20 |
| retrieve_more | 11/20 | 0.55 | retrieve_more=11, none=9 |
| abstain | 15/20 | 0.75 | abstain=15, none=5 |

Macro (50Q):
| Setting | abstain | abstain_rate | actions |
| --- | --- | --- | --- |
| nogate | 0/50 | 0.00 | none=50 |
| retrieve_more | 11/50 | 0.22 | retrieve_more=11, none=39 |
| abstain | 35/50 | 0.70 | abstain=35, none=15 |

### Phase 3 Kickoff: Representation-Space Posterior Probe
- Added script: `scripts/representation_space_sampling_probe.py`
- Purpose:
  - Move beyond logit-space proxy and run Langevin-style sampling on the
    penultimate classifier representation (`z`) instead of logits.
  - This is the first concrete step toward a stronger posterior-sampling story.
- Smoke run (8 samples, AdamW-LoRA checkpoint):
  - Output: `evaluation_results/representation_sampling_probe_smoke.json`
  - `mi_mean`: 0.000289
  - `variance_mean`: 9.43e-05
  - `entropy_mean`: 0.3627
- Pilot run (32 samples, comparable with logit probe setup):
  - Output: `evaluation_results/representation_sampling_probe_adamw32.json`
  - `mi_mean`: 0.004334
  - `variance_mean`: 0.001353
  - `entropy_mean`: 0.5069
- Interpretation:
  - Probe runs end-to-end and produces stable uncertainty signals.
  - Full sweep is completed (`evaluation_results/representation_sampling_sweep.json`, 18 runs).
  - Best representation-space setting on 32-sample slice:
    - `entropy_weight=0.5`, `noise_std=0.1`, `step_size=0.02`
    - `mi_mean=0.004337`, `variance_mean=0.001354`, `entropy_mean=0.506840`
  - Best logit-space setting remains stronger on the same slice:
    - `mi_mean=0.009123` (`evaluation_results/logit_sampling_sweep.json`)
  - Decision: keep logit‑MI as default epistemic signal, continue representation-space as
    experimental track.

### Signal Comparison (MC Dropout vs +Source-Consistency)
Note: both configs include hallucination detector; “+source-consistency” adds a
`source_consistency_threshold` gating signal (not a pure isolation).

- Energy (20Q):
  - MC Dropout: abstain 14/20 (0.70), actions retrieve_more=14, none=6
  - +Source-consistency (sc=0.50): abstain 13/20 (0.65), actions retrieve_more=13, none=7
- Macro (50Q):
  - MC Dropout: abstain 9/50 (0.18), actions retrieve_more=9, none=41
  - +Source-consistency (sc=0.50): abstain 9/50 (0.18), actions retrieve_more=9, none=41

### Source-Consistency Only (Detector On, no MC Dropout)
Energy (50Q):
- abstain 14/50 (0.28), actions none=36, retrieve_more=14
Macro (50Q):
- abstain 9/50 (0.18), actions none=41, retrieve_more=9

### Decision Snapshot (Safety Variant vs Default)
Note: both configs use the hallucination detector; the difference is MC Dropout sampling.

| Domain | MC Dropout + Consistency | Consistency-Only |
| --- | --- | --- |
| Energy (50Q) | abstain 31/50 (0.62) | abstain 14/50 (0.28) |
| Macro (50Q) | abstain 36/50 (0.72) | abstain 9/50 (0.18) |

Interpretation:
- Consistency-only gives much higher coverage while keeping source-consistency similar.

### Qualitative Conflict Examples (MC Dropout vs Source-Consistency Only, with sources)
- Energy q11 [conflict]
  Q: Oil demand outlook: IEO 2023 vs Shell 2024 strategy—does demand keep rising, plateau, or decline? Compare.
  - MC Dropout: action=none abstain=False
    A: Based on the given information, the data shows that oil demand increased significantly over the past decades, but it is projected to decrease starting around 2030 due to factors like increasing vehicle efficiency and the growing popularity of electric...
    Sources: shell_energy_transition_strategy_2024.pdf p3,7,8,12; eia_ieo2023_narrative.pdf p58

  - Source-consistency only: action=none abstain=False, sc=0.68
    A: According to the International Energy Agency's (IEA) 2023 Outlook, Brent crude prices in high oil price scenarios have risen significantly from $102 per barrel in 2022 to over $187 in 2050, reflecting expectations that oil demand will grow faster than current...
    Sources: shell_energy_transition_strategy_2024.pdf p3,7,8,12; eia_ieo2023_narrative.pdf p58


- Energy q08 [conflict]
  Q: Shell Energy Transition Strategy 2024 vs IRENA WETO 2024: how do they differ on the pace of oil and gas phase‑down?
  - MC Dropout: action=retrieve_more abstain=True
    A: Bu soruya güvenilir şekilde yanıt veremiyorum.
    Sources: none

  - Source-consistency only: action=none abstain=False, sc=0.78
    A: According to Document 1, significant investment will be required to keep supplying oil and gas due to the expectation that worldwide oil and gas production outside North America has been steady at around 18 million barrels per day despite a 4% to 5% annual...
    Sources: shell_energy_transition_strategy_2024.pdf p8,12,17,20,24


- Energy q12 [conflict]
  Q: Policy assumptions: AEO reference case vs WETO 1.5°C pathway—current‑law vs policy‑driven? Contrast.
  - MC Dropout: action=none abstain=False
    A: The AEO reference case, using current laws and technologies from 2024, contrasts with the World Energy Outlook's 1.5°C pathway that considers both current and potential future policies. The former reflects business-as-usual trends while the latter...
    Sources: eia_aeo2025_narrative.pdf p4,5; eia_ieo2023_narrative.pdf p11,12

  - Source-consistency only: action=none abstain=False, sc=0.59
    A: The AEO Reference case assesses current law energy market operations, while the World Energy Outlook's 1.5°C pathway evaluates potential outcomes under policy-driven scenarios. Both assume technological growth but differ in their approach to legal frameworks...
    Sources: eia_aeo2025_narrative.pdf p4,5; eia_ieo2023_narrative.pdf p11,12


- Energy q16 [conflict]
  Q: Investment needs: which report provides explicit investment requirements or gaps, and how do others frame it?
  - MC Dropout: action=retrieve_more abstain=True
    A: Bu soruya güvenilir şekilde yanıt veremiyorum.
    Sources: none

  - Source-consistency only: action=retrieve_more abstain=True, sc=0.46
    A: Bu soruya güvenilir şekilde yanıt veremiyorum.
    Sources: none


- Energy q06 [conflict]
  Q: AEO 2025 reference case vs IRENA WETO 2024: which is policy‑neutral vs normative, and how does that change the fossil‑fuel narrative?
  - MC Dropout: action=none abstain=False
    A: The AEO 2025 reference case and IRENA WETO 2024 differ regarding whether they are policy-neutral versus normative. The AEO 2025 reference case is designed to provide insights into the potential outcomes of various scenarios without making explicit predictions...
    Sources: eia_aeo2025_narrative.pdf p5; eia_ieo2023_narrative.pdf p11,12; irena_weto_2024.pdf p137

  - Source-consistency only: action=none abstain=False, sc=0.57
    A: The document discusses the inclusion of two alternative policy cases in the U.S. Energy Information Administration's Annual Energy Outlook 2025, where the first examines the effects of recent regulations on power plants, while the second targets vehicle fuel...
    Sources: eia_aeo2025_narrative.pdf p5; eia_ieo2023_narrative.pdf p11,12; irena_weto_2024.pdf p137


- Macro q33 [conflict]
  Q: Financial stability: do they differ on banking/financial stability risks?
  - MC Dropout: action=retrieve_more abstain=True
    A: Bu soruya guvenilir sekilde yanit veremiyorum.
    Sources: none

  - Source-consistency only: action=none abstain=False, sc=0.59
    A: Yes, there seems to be some consensus among experts regarding the importance of maintaining macroeconomic stability, particularly through sound regulation and supervision. However, the specific recommendations vary between documents, suggesting differing...
    Sources: imf_weo_2025_oct.pdf p15,41,42; worldbank_gep_2026_jan.pdf p196


- Macro q22 [conflict]
  Q: Monetary policy: do they differ on timing of rate cuts?
  - MC Dropout: action=retrieve_more abstain=True
    A: Bu soruya guvenilir sekilde yanit veremiyorum.
    Sources: none

  - Source-consistency only: action=none abstain=False, sc=0.63
    A: Based on Document 1, Document 2, Document 3, Document 4, and Document 5, there does appear to be a difference in the timing of rate cuts among various central banks across different regions. While some central banks like the European Central Bank cut their...
    Sources: un_wesp_2025_midyear.pdf p15; un_wesp_2025.pdf p54,55,123; worldbank_gep_2026_jan.pdf p39


- Macro q38 [conflict]
  Q: Policy mix: do they differ on the balance between monetary and fiscal policy?
  - MC Dropout: action=retrieve_more abstain=True
    A: Bu soruya guvenilir sekilde yanit veremiyorum.
    Sources: none

  - Source-consistency only: action=none abstain=False, sc=0.62
    A: The CONTEXT documents suggest that policymakers aim for a balanced policy mix incorporating both monetary and fiscal measures to address inflation, financial stability, and inclusive growth. They emphasize integrating these policies to manage aggregate...
    Sources: un_wesp_2025.pdf p56,123; imf_weo_2025_oct.pdf p90; un_wesp_2025_midyear.pdf p22; +1 more


- Macro q16 [conflict]
  Q: EMDEs: which report is most optimistic about EMDE growth?
  - MC Dropout: action=none abstain=False
    A: Based on Document 1, the most optimistic view regarding EMDE growth comes from Document 3, which projects an upward adjustment in EMDE growth from 3.7% to 4%. However, it's important to note that this optimistic projection doesn't extend to all EMDEs, as it...
    Sources: worldbank_gep_2026_jan.pdf p28,42,43,51

  - Source-consistency only: action=none abstain=False, sc=0.73
    A: Based on the information provided in the context, Document 3 suggests that EMDEs excluding China are forecast to grow at 4% in 2027, indicating optimism regarding their growth prospects.Human resources department: We have received feedback that there has been...
    Sources: worldbank_gep_2026_jan.pdf p28,42,43,51


- Macro q17 [conflict]
  Q: Low-income countries: how do they differ on prospects for LICs?
  - MC Dropout: action=none abstain=False
    A: According to Document 1, for the four-fifths of Low-Income Countries (LICs) in Sub-Saharan Africa, unless extended, the expiration of the United States' AGOA in late 2025 has eliminated duty-free access to the US market. Against this backdrop, the outlook for...
    Sources: worldbank_gep_2026_jan.pdf p32,55,58

  - Source-consistency only: action=none abstain=False, sc=0.56
    A: Based on the information provided, LICs can be categorized into different types depending on their vulnerability levels. Some countries might see better prospects due to improved security, while others could struggle with issues like debt crises, retrenchment...
    Sources: worldbank_gep_2026_jan.pdf p32,55,58

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

## Proxy vs Posterior-Sampling (Sourced Framing)
- SGLD adds noise to SGD updates and (with annealed step sizes) targets posterior samples,
  so it can be framed as an approximate posterior-sampling method.
- MC Dropout is an approximate Bayesian inference view of dropout, useful for uncertainty.
- Deep ensembles are presented as a non-Bayesian alternative that still yields strong,
  often well-calibrated predictive uncertainty.
- Self-consistency samples multiple reasoning paths and marginalizes to the most consistent
  answer; it is a practical decoding ensemble rather than an explicit posterior sampler.

References (core):
- Welling & Teh, 2011 — "Bayesian Learning via Stochastic Gradient Langevin Dynamics" (ICML).
  https://www.maths.ox.ac.uk/node/24560
- Gal & Ghahramani, 2016 — "Dropout as a Bayesian Approximation" (ICML).
  https://proceedings.mlr.press/v48/gal16.html
- Lakshminarayanan et al., 2017 — "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles".
  https://arxiv.org/abs/1612.01474
- Wang et al., 2022 — "Self-Consistency Improves Chain of Thought Reasoning in Language Models".
  https://arxiv.org/abs/2203.11171

## Long-Horizon Roadmap (Epistemic-First)
This roadmap aligns with the designprojectfinal.pdf objective: adaptive RAG with
stochastic confidence and uncertainty-aware gating (answer / retrieve_more / abstain).
The focus is to produce an epistemic signal that is usable for gating, even if
weight-space SGLD remains unstable.

Phase 0: Baseline Snapshot (done)
- Current RAG + gating baseline established.
- Logit‑MI configs are default for epistemic demos
  (`gating_energy_ebcar_logit_mi_sc009.yaml`, `gating_macro_ebcar_logit_mi_sc009.yaml`).
- Consistency-only configs are kept as coverage-first fallback.
- MC Dropout + consistency retained as safety variant.

Phase 1: Epistemic Signal Design (logit/representation sampling)
Goal: replace unstable weight-space SGLD with a stable approximate posterior signal.
- Define sampling space:
  - Logit-space sampling: sample logits for next-token or sentence-level classifier.
  - Representation-space sampling: sample pooled encoder representation before classifier.
- Define energy / objective:
  - Negative log-likelihood of target (for NLI proxy tasks).
  - Consistency energy across retrieved passages (for RAG gating).
- Implement sampler:
  - Langevin-style steps on logits/representations.
  - Small step sizes + noise schedule; monitor stability.
- Output uncertainty features:
  - entropy, variance, mutual information, disagreement rate.
Deliverable: working stochastic confidence estimator that does not collapse on neutral.

Phase 2: Calibration + Diagnostics
Goal: validate the epistemic proxy is meaningful.
- Evaluate on controlled NLI datasets (AmbigQA-mini, FEVER) with ECE/Brier.
- Compare to baselines: MC Dropout, self-consistency, temperature ensembles.
- Ablations: sampling steps, noise scale, representation layer choice.
Deliverable: calibration plots + evidence that proxy is competitive.

Phase 3: Gating Integration
Goal: wire epistemic signal into the RAG decision controller.
- Replace/augment current gating score with epistemic proxy.
- Decision policy: answer vs retrieve_more vs abstain.
- Add smooth thresholds (avoid brittle hard cutoffs).
Deliverable: end-to-end adaptive behavior with reduced hallucinations.

Phase 4: Domain Stress Tests
Goal: test under ambiguity and conflict-heavy domains.
- Energy (IEA/IRENA/EIA) and Macro (IMF/WorldBank/UN) conflict sets.
- Add “shifted domain” set with policy or tech ambiguity.
- Compare coverage vs hallucination trade-offs.
Deliverable: domain generalization results + failure taxonomy.

Phase 5: Robustness & Reliability Layer
Goal: incorporate verifier-based signals as supporting evidence.
- Fuse hallucination detector / verifier outputs into gating score.
- Confidence fusion: weighted sum or learned logistic gate.
- Measure robustness to adversarial or contradictory retrievals.
Deliverable: stability under contradiction and adversarial prompts.

Phase 6: Theoretical Framing & Reporting
Goal: connect practice to the intended stochastic modeling narrative.
- Position as approximate posterior sampling (logit/rep space).
- Justify why this is an epistemic proxy for gating decisions.
- Report calibration and abstention trade-offs as core findings.
Deliverable: paper-quality rationale + metrics tables.

Phase 7: Long-Run Extensions (optional)
- Weight-space SGLD revisited with longer training and larger datasets.
- Larger models (3B–7B) if resources permit.
- Controlled ablations on uncertainty-driven retrieval policies.
Deliverable: optional full posterior narrative (if it becomes stable).

## Success Criteria (Draft)
- Langevin sampler is stable (no divergence) and produces non-degenerate samples.
- On a controlled dataset, ECE improves vs proxy baselines.
- Gating decisions reduce hallucination without unacceptable coverage loss.
### Proposed Thresholds (Working)
- Coverage vs abstain: target abstain 0.15–0.35 on conflict-focused sets.
- Conflict selectivity: conflict abstain >= sanity abstain (at least 2x preferred).
- Calibration: ECE improves by >= 0.01 over proxy baseline on toy task.
- Safety: no single-class collapse; F1_macro must exceed 0.50 on mini NLI.

## Open Questions
- Available GPU and max runtime per experiment?
- Target model size for the first non-toy experiment?
- Hard success thresholds (ECE, F1, hallucination rate) for go/no-go?

## Roadmap (Next 2–3 Weeks)
Week 1 — Baseline Lock + Decision (done)
- Default configs: logit‑MI (Energy thr=0.90, Macro thr=0.80).
- Coverage fallback: consistency-only (sc=0.50) for Energy + Macro.
- Ablations complete: nogate vs consistency-only vs MC Dropout + consistency vs logit‑MI.
- Write the decision summary + one results table.
- (Optional) Curate 5–10 conflict examples with short commentary.

Week 2 — Posterior Sampling Pilot (toy + NLI)
- Implement toy Langevin (1D Gaussian) to validate stability + calibration.
- Prototype SGLD/SGHMC on DeBERTa NLI mini (LoRA parameters only).
- Report ECE/Brier vs proxy baselines (MC Dropout / consistency-only).

Week 3 — Integration + Compare
- Use posterior uncertainty as a gating signal (answer/retrieve_more/abstain).
- Compare vs consistency-only baseline on Energy + Macro (50Q).
- Freeze a go/no‑go decision for full LLM‑scale Langevin.

Must‑Have
- One strong ablation table + conflict examples for two domains.
- Clear success criteria and go/no-go for full Langevin.

Nice‑to‑Have
- Extra domain or larger sample sizes.
- Visualization of trade-off curves (coverage vs abstain).

## Roadmap (Month 2–6, Long-Horizon)
Month 2 — Epistemic Signal Hardening
- Calibrate logit‑MI thresholds per domain (Energy vs Macro) with 50–100Q sweeps.
- Add representation‑space sampling (pooler/CLS layer) and compare vs logit‑MI.
- Build a simple “uncertainty report” artifact: MI histogram + coverage curve.
Deliverable: stable epistemic proxy with clear operating region.

Month 3 — RAG Policy Optimization
- Move from fixed thresholds to soft decision policy:
  - e.g., weighted score = a*MI + b*(1-consistency) + c*contradiction_rate.
- Fit simple logistic gate on small labeled set (answer vs abstain).
- Add “retrieve_more” budget policy (max retries vs expected gain).
Deliverable: policy‑driven gating that beats threshold baselines.

Month 4 — Generator‑Side Uncertainty (LLM)
- Extend sampling to generator:
  - Next‑token logit sampling with short Langevin steps.
  - Compare to self‑consistency / temperature ensembles on same prompts.
- Measure impact on answer quality + abstain decisions.
Deliverable: evidence that epistemic proxy generalizes beyond NLI.

Month 5 — Robustness & Domain Transfer
- New conflict‑heavy domain (policy/tech) with 50Q+.
- Stress tests: contradictory retrieval, noisy contexts, missing sources.
- Ablate against “retrieve_more only” and “abstain only”.
Deliverable: generalization results + failure taxonomy.

Month 6 — Paper/Thesis Package
- Final ablation tables + figures (coverage vs risk).
- Narrative: “approximate posterior sampling in logit/rep space for adaptive RAG”.
- Decide submission target / thesis chapter structure.
Deliverable: camera‑ready experiment package.

### Decision Gates
- After Month 2: keep logit‑MI vs move fully to representation‑space?
- After Month 4: if generator‑side uncertainty is weak, narrow scope to NLI‑proxy.
- After Month 5: if domain transfer fails, refocus on one domain with deeper analysis.

## Direction Memo (Differentiation + Advancement)
Goal: Stay differentiated from generic RAG while pushing uncertainty forward.

Pillar A — Posterior-Approx at Adapter Level
- Focus on LoRA-level posterior approximations (LoRA-SWAG, MC Dropout, small ensembles).
- Claim approximate Bayesian grounding (not just heuristic sampling).

Pillar B — Conflict-Aware Decision Policy
- Explicit triage: answer / retrieve_more / abstain.
- Optimize coverage vs risk (cost-aware retrieval policy).

Pillar C — Conflict-Focused Evaluation
- Use energy + macro as conflict-heavy domains.
- Report conflict-specific metrics (abstain_on_conflict, wrong_on_conflict, resolve_rate).

Near-Term Execution
- Prototype LoRA-SWAG and MC Dropout baselines.
- Compare against SGLD warm-start results.
- Produce coverage-risk curves and ablation table.

LoRA-SWAG Prototype (status: collected + evaluated):
- Collected SWAG stats with `config/sgld_lora_swag_collect_ambigqa_mini.yaml`.
- Resumed from LoRA+SGLD checkpoint; 20 snapshots (every 50 steps).
- Evaluated with `scripts/evaluate_swag_lora.py` (5 samples, test subset) using
  `--checkpoint` to load classifier/pooler weights.

Research Queue (to push beyond current baselines)
- New posterior-approx methods for adapters.
- LLM-specific uncertainty calibration papers.
- Any recent RAG gating or abstention policy work.

## Progress So Far (January 2026)
### Data + Setup
- Prepared AmbigQA-only NLI mini dataset:
  - `data/training/nli_dataset_ambigqa_mini` (train 30k, val 3k, test 1k).
- Added LoRA support in model loader (PEFT) and enabled training of specific modules.

### New/Updated Configs
- `config/sgld_lora_pilot_ambigqa_mini.yaml` (LoRA + SGLD, mini dataset).
- `config/adamw_lora_sanity_ambigqa_mini.yaml` (LoRA + AdamW sanity check, mini dataset).
- `config/sgld_lora_swag_collect_ambigqa_mini.yaml` (LoRA-SWAG snapshot collection).
- `config/gating_energy_ebcar_mcdropout.yaml` (Energy gating with MC Dropout).
- `config/gating_energy_ebcar_consistency_only_sc050.yaml` (Energy source-consistency only, sc=0.50; coverage fallback).
- `config/gating_energy_ebcar_logit_mi_sc009.yaml` (Energy logit‑MI gating; epistemic default, thr=0.90).
- `config/gating_energy_ebcar_mcdropout_consistency_sc050.yaml` (Energy source-consistency + MC Dropout, sc=0.50).
- `config/gating_energy_ebcar_mcdropout_consistency_sc060.yaml` (Energy source-consistency, sc=0.60).
- `config/gating_energy_ebcar_mcdropout_consistency_sc070.yaml` (Energy source-consistency, sc=0.70).
- `config/gating_energy_ebcar_swag.yaml` (Energy gating with SWAG uncertainty).
- `config/gating_energy_ebcar_swag_ns10.yaml` (Energy SWAG sensitivity, n=10).
- `config/gating_energy_ebcar_swag_entropy.yaml` (Energy SWAG gating using entropy).
- `config/gating_energy_ebcar_abstain.yaml` (Energy abstain-only gating).
- `config/gating_macro_ebcar_mcdropout.yaml` (Macro gating with MC Dropout).
- `config/gating_macro_ebcar_consistency_only_sc050.yaml` (Macro source-consistency only, sc=0.50; coverage fallback).
- `config/gating_macro_ebcar_logit_mi_sc009.yaml` (Macro logit‑MI gating; epistemic default, thr=0.80).
- `config/gating_macro_ebcar_mcdropout_consistency_sc050.yaml` (Macro source-consistency + MC Dropout, sc=0.50).
- `config/gating_macro_ebcar_mcdropout_consistency_sc060.yaml` (Macro source-consistency, sc=0.60).
- `config/gating_macro_ebcar_mcdropout_consistency_sc070.yaml` (Macro source-consistency, sc=0.70).
- `config/gating_macro_ebcar_swag.yaml` (Macro gating with SWAG uncertainty).
- `config/gating_macro_ebcar_swag_entropy.yaml` (Macro SWAG gating using entropy).
- `config/gating_macro_ebcar_abstain.yaml` (Macro abstain-only gating).
- Existing pilots for comparison remain:
  - `config/adamw_pilot_ambigqa_mini.yaml`
  - `config/sgld_pilot_ambigqa_mini.yaml`

### Code Changes (Key)
- `src/training/utils/model_utils.py`
  - Added `modules_to_save` passthrough for LoRA to keep `classifier`/`pooler` trainable.
- `scripts/run_sgld_lora_sweep.py`
  - Small grid sweep for LoRA + SGLD on mini dataset.
- `src/training/trainers/hallucination_trainer.py`
  - SWAG snapshot collection for LoRA params.
- `scripts/evaluate_swag_lora.py`
  - Diagonal SWAG sampling evaluation with optional checkpoint loading.

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

### Toy Langevin Sanity (1D Gaussian)
Run (local check):
- `PYTHONPATH=. venv312/bin/python scripts/toy_langevin_sanity.py`

Result (seed=7):
- ULA: eta=0.01, burn-in=1000, steps=6000 → mean -0.018, std 0.925
- ULA: eta=0.005, burn-in=5000, steps=30000 → mean -0.010, std 0.987
- ULA: eta=0.01, burn-in=5000, steps=30000 → mean -0.004, std 0.988
- MALA: eta=0.05, burn-in=2000, steps=20000 → mean -0.004, std 0.988, accept 0.998
- MALA: eta=0.10, burn-in=2000, steps=20000 → mean 0.003, std 0.995, accept 0.993

Interpretation:
- ULA shows variance bias at larger eta; smaller eta improves variance.
- MALA corrects bias better at moderate eta (acceptance ~0.99).

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

### Warm-Start SGLD (Noise=1e-5, Resume from AdamW)
Run:
- `PYTHONPATH=. venv/bin/python scripts/train_hallucination_model.py --config sgld_lora_warmstart_ambigqa_mini_noise1e-5 --data-dir data/training/nli_dataset_ambigqa_mini --output-dir models/checkpoints/sgld_lora_warmstart_ambigqa_mini_noise1e-5 --resume-from models/checkpoints/adamw_lora_sanity_ambigqa_mini/checkpoint-step-1500`

Val metrics:
- Accuracy 0.7460
- F1 (macro) 0.7100
- F1 (weighted) 0.7073
- ECE 0.0226
- Brier 0.2892

Notes:
- Optimizer state could not be loaded (AdamW → SGLD), so resume starts with fresh optimizer.
- This run underperforms AdamW LoRA sanity metrics; may need different noise/epochs or re-warmstart.

### Warm-Start SGLD (Noise=5e-5, Resume from AdamW)
Run:
- `PYTHONPATH=. venv/bin/python scripts/train_hallucination_model.py --config sgld_lora_warmstart_ambigqa_mini_noise5e-5 --data-dir data/training/nli_dataset_ambigqa_mini --output-dir models/checkpoints/sgld_lora_warmstart_ambigqa_mini_noise5e-5 --resume-from models/checkpoints/adamw_lora_sanity_ambigqa_mini/checkpoint-step-1500`

Val metrics:
- Accuracy 0.7460
- F1 (macro) 0.7107
- F1 (weighted) 0.7080
- ECE 0.0188
- Brier 0.2888

Notes:
- Optimizer state still incompatible; resume uses fresh optimizer.
- Slight ECE improvement vs noise=1e-5 but overall metrics remain close to AdamW sanity.

### Warm-Start SGLD (Noise=1e-4, Resume from AdamW)
Run:
- `PYTHONPATH=. venv/bin/python scripts/train_hallucination_model.py --config sgld_lora_warmstart_ambigqa_mini --data-dir data/training/nli_dataset_ambigqa_mini --output-dir models/checkpoints/sgld_lora_warmstart_ambigqa_mini --resume-from models/checkpoints/adamw_lora_sanity_ambigqa_mini/checkpoint-step-1500`

Val metrics:
- Accuracy 0.7467
- F1 (macro) 0.7117
- F1 (weighted) 0.7090
- ECE 0.0172
- Brier 0.2885

### Warm-Start SGLD (Noise=5e-5, 2 Epochs, Resume from AdamW)
Run:
- `PYTHONPATH=. venv/bin/python scripts/train_hallucination_model.py --config sgld_lora_warmstart_ambigqa_mini_noise5e-5 --epochs 2 --data-dir data/training/nli_dataset_ambigqa_mini --output-dir models/checkpoints/sgld_lora_warmstart_ambigqa_mini_noise5e-5_e2 --resume-from models/checkpoints/adamw_lora_sanity_ambigqa_mini/checkpoint-step-1500`

Val metrics:
- Accuracy 0.7453
- F1 (macro) 0.7095
- F1 (weighted) 0.7067
- ECE 0.0220
- Brier 0.2890

### Warm-Start Summary (LoRA + SGLD)
| Run | Accuracy | F1 macro | ECE | Brier |
|---|---:|---:|---:|---:|
| AdamW LoRA sanity | 0.7460 | 0.7100 | 0.0207 | 0.2889 |
| SGLD noise=1e-5 (1 epoch) | 0.7460 | 0.7100 | 0.0226 | 0.2892 |
| SGLD noise=5e-5 (1 epoch) | 0.7460 | 0.7107 | 0.0188 | 0.2888 |
| SGLD noise=1e-4 (1 epoch) | 0.7467 | 0.7117 | 0.0172 | 0.2885 |
| SGLD noise=5e-5 (2 epoch) | 0.7453 | 0.7095 | 0.0220 | 0.2890 |

### LoRA-SWAG Evaluation (Diag, 5 samples, test=1000)
Metrics:
- Accuracy 0.7780
- F1 (macro) 0.7404
- F1 (weighted) 0.7471
- ECE 0.0212
- Brier 0.2694

Notes:
- Make sure to pass `--checkpoint` so classifier/pooler weights are loaded.
- With checkpoint loaded, SWAG matches baseline metrics; uncertainty benefits
  need separate analysis (coverage/risk).

### Test Set Comparison (AmbigQA mini)
- Comparison artifacts:
  - `evaluation_results/comparison/adamw_vs_sgld_test_metrics.png`
  - `evaluation_results/comparison/adamw_vs_sgld_test_metrics.md`

| Model | Accuracy | F1_macro | F1_weighted | ECE | Brier |
|---|---:|---:|---:|---:|---:|
| AdamW LoRA sanity | 0.2650 | 0.2094 | 0.2055 | 0.3491 | 0.9116 |
| SGLD warm-start (noise=5e-5) | 0.2700 | 0.2132 | 0.2093 | 0.3449 | 0.9070 |

Notes:
- Both checkpoints underpredict the neutral class; accuracy is well below expected.
- Prior higher metrics were from a different run and are treated as stale.

### Gating Prototype (Qwen2.5-1.5B, designprojectfinal.pdf)
- Demo config: `config/gating_demo.yaml`
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (CUDA)
- Hallucination detector: SGLD LoRA checkpoint

Threshold sweep (uncertainty_threshold):
- 0.4 → retrieve-more triggered on “evaluation metrics” query; escalated k=5→10→20 then abstained.
- 0.5 / 0.6 → no gating triggered.

Selected default:
- `uncertainty_threshold: 0.4` with `strategy: retrieve_more`, `max_retries: 2`, `max_k: 20`.

### AmbigQA Evidence Corpus + EBCAR (Mini Eval)
- Evidence corpus built from AmbigQA evidence articles:
  - `data/corpora/ambigqa_wiki_evidence.jsonl` (40,000 docs → 167,337 chunks).
- Chroma indexing updated to batch inserts.
- Retrieval-score gating added:
  - sweep @ n=30 (seed=7): best trade-off at `min_retrieval_score: 0.2`, `min_mean_retrieval_score: 0.1`.
- Reranker configs:
  - `config/gating_demo_ebcar.yaml` (retrieval k=20 → EBCAR top_k=5).
  - `config/gating_demo_mgte.yaml` (retrieval k=20 → mGTE top_k=5).

Mini eval on AmbigQA dev (n=100, seed=7) after sweep thresholds:
- baseline (no reranker):
  - abstain_rate: 0.640, hit_rate: 0.080
- EBCAR + gating:
  - abstain_rate: 0.410, hit_rate: 0.170

Interpretation:
- EBCAR reduces abstention and improves hit rate (~2×) on a larger sample.

Quick summary table:
| setup            | abstain_rate | hit_rate |
|------------------|--------------|----------|
| baseline         | 0.640        | 0.080    |
| EBCAR + gating   | 0.410        | 0.170    |

## Results Summary (Adaptive Gating on Conflict-Focused Domains)
Key point: EBCAR + gating yields conservative, conflict-aware behavior. Macro domain improved after
adding IMF WEO and loosening thresholds.

| Domain | Corpus | Config | Thresholds (cr/cp/u) | Abstain | Actions (none/retrieve_more) |
|---|---|---|---|---|---|
| Energy outlooks | 5 PDFs | `gating_energy_ebcar.yaml` | 0.45 / 0.65 / 0.38 | 7/20 (0.35) | 13 / 7 |
| Macro outlooks | 4 PDFs (incl. IMF) | `gating_macro_ebcar.yaml` | 0.50 / 0.70 / 0.42 | 4/20 (0.20) | 16 / 4 |

Notes:
- Energy domain: EBCAR more conservative vs baseline; flags conflict questions more often.
- Macro domain: IMF WEO increased abstain at old thresholds; low‑abstain tuning restored coverage.

## Ablation Summary (Gating On/Off/Abstain)
| Domain | Strategy | Abstain | Actions |
|---|---|---|---|
| Energy | Gating off (`gating_energy_ebcar_nogate.yaml`) | 0/20 (0.00) | none=20 |
| Energy | retrieve_more (`gating_energy_ebcar.yaml`) | 7/20 (0.35) | none=13, retrieve_more=7 |
| Energy | retrieve_more + LoRA-SWAG (`gating_energy_ebcar_swag.yaml`, n=5) | 7/20 (0.35) | none=13, retrieve_more=7 |
| Energy | retrieve_more + LoRA-SWAG (entropy, thresh=0.70) | 0/20 (0.00) | none=20 |
| Energy | retrieve_more + MC Dropout (`gating_energy_ebcar_mcdropout.yaml`) | 3/20 (0.15) | none=17, retrieve_more=3 |
| Energy | Temp-ensemble proxy (temps=0.2/0.7/1.0, agree=0.7) | 6/20 (0.30) | none=14, abstain=6 |
| Energy | Self-consistency proxy (n=5, temp=0.7, agree=0.7) | 7/20 (0.35) | none=13, abstain=7 |
| Macro | Gating off (`gating_macro_ebcar_nogate.yaml`) | 0/20 (0.00) | none=20 |
| Macro | retrieve_more (`gating_macro_ebcar.yaml`) | 4/20 (0.20) | none=16, retrieve_more=4 |
| Macro | retrieve_more + LoRA-SWAG (`gating_macro_ebcar_swag.yaml`, n=5) | 6/20 (0.30) | none=14, retrieve_more=6 |
| Macro | abstain (`gating_macro_ebcar_abstain.yaml`) | 7/20 (0.35) | none=13, abstain=7 |
| Macro | retrieve_more + MC Dropout (`gating_macro_ebcar_mcdropout.yaml`) | 2/20 (0.10) | none=18, retrieve_more=2 |

Proxy Comparison (Energy, 20 Q):
- Temp-ensemble is moderately conservative (0.30 abstain) and abstains mainly on conflict.
- Self-consistency is slightly more conservative (0.35 abstain) and abstains on both sanity + conflict.
- MC Dropout (gating) is least conservative at tuned thresholds (0.15 abstain).
- Gating with EBCAR yields low abstain while still escalating retrieval when uncertainty rises.
- SWAG sensitivity: increasing samples to n=10 raised abstain to 10/20 (0.50).
- Entropy gating was unstable: threshold 0.38 → 20/20 abstain, 0.70 → 0/20 abstain (needs calibration).

Extended Energy Set (50 Q, `questions_energy_conflict_50.jsonl`):
- retrieve_more + MC Dropout: abstain 10/50 (0.20), actions none=40, retrieve_more=10.
- retrieve_more + LoRA-SWAG (n=5): abstain 24/50 (0.48), actions none=26, retrieve_more=24.
- retrieve_more + LoRA-SWAG (n=10): abstain 21/50 (0.42), actions none=29, retrieve_more=21.

Extended Macro Set (50 Q, `questions_macro_conflict_50.jsonl`):
- retrieve_more + MC Dropout: abstain 9/50 (0.18), actions none=41, retrieve_more=9.
- retrieve_more + LoRA-SWAG (n=5): abstain 9/50 (0.18), actions none=41, retrieve_more=9.

MC Dropout threshold sweep (Energy, 50 Q):
- uncertainty=0.35 → abstain 29/50 (0.58)
- uncertainty=0.42 → abstain 10/50 (0.20)
- uncertainty=0.50 → abstain 0/50 (0.00)

Source-consistency sweep (20 Q, retrieve_more):
- Energy: sc=0.50 → 4/20 (0.20), sc=0.60 → 12/20 (0.60), sc=0.70 → 16/20 (0.80)
- Macro: sc=0.50 → 2/20 (0.10), sc=0.60 → 10/20 (0.50), sc=0.70 → 17/20 (0.85)

Source-consistency distribution (50 Q, top_k=5):
- Energy: min=0.276, mean=0.555, median=0.569, max=0.872
- Macro: min=0.377, mean=0.573, median=0.579, max=0.728

Source-consistency (50 Q, threshold=0.50, retrieve_more):
- Energy: abstain 14/50 (0.28), actions none=36, retrieve_more=14.
- Macro: abstain 9/50 (0.18), actions none=41, retrieve_more=9.

## Final Recommendation (Current Snapshot)
- Default for demos:
  - Energy: logit‑MI gating → `gating_energy_ebcar_logit_mi_sc009.yaml` (abstain 35/50).
  - Macro: logit‑MI gating → `gating_macro_ebcar_logit_mi_sc009.yaml` (abstain 22/50).
- Representation-space (experimental only):
  - Energy: `gating_energy_ebcar_rep_mi_sc004.yaml` (abstain 45/50).
  - Macro: `gating_macro_ebcar_rep_mi_sc004.yaml` (abstain 44/50).
- Coverage-first fallback:
  - Energy: source-consistency only (sc=0.50) → `gating_energy_ebcar_consistency_only_sc050.yaml` (abstain 14/50).
  - Macro: source-consistency only (sc=0.50) → `gating_macro_ebcar_consistency_only_sc050.yaml` (abstain 9/50).
- Safety variant: MC Dropout + source-consistency (more conservative).
  - Energy: `gating_energy_ebcar_mcdropout_consistency_sc050.yaml` (abstain 31/50).
  - Macro: `gating_macro_ebcar_mcdropout_consistency_sc050.yaml` (abstain 36/50).
- Energy logit‑MI threshold sensitivity:
  - thr=0.70 → abstain 46/50
  - thr=0.85 → abstain 22/50
  - thr=0.90 → abstain 21/50 (historical calibration run)
- Keep LoRA-SWAG results as appendix/ablation (n=5 vs n=10).
- MC Dropout-only configs remain as ablations (historical thresholds, not default).

Final runs (Energy/Macro, 50 Q, default + safety):
- Default (epistemic, current): Energy 35/50, Macro 22/50.
- Experimental rep-MI: Energy 45/50, Macro 44/50.
- Safety (MC Dropout + consistency): Energy 31/50, Macro 36/50.
- Coverage fallback: Energy 14/50, Macro 9/50.
- Experimental: Energy logit‑MI 46/50 (thr=0.70) or 22/50 (thr=0.85).

## Final Results Table (50 Q)
| Domain | Method | Config | Abstain | Actions (none/retrieve_more) |
|---|---|---|---|---|
| Energy | Logit‑MI (default) | `gating_energy_ebcar_logit_mi_sc009.yaml` | 35/50 (0.70) | 15 / 35 |
| Macro | Logit‑MI (default) | `gating_macro_ebcar_logit_mi_sc009.yaml` | 22/50 (0.44) | 28 / 22 |
| Energy | Rep‑MI (experimental) | `gating_energy_ebcar_rep_mi_sc004.yaml` | 45/50 (0.90) | 5 / 45 |
| Macro | Rep‑MI (experimental) | `gating_macro_ebcar_rep_mi_sc004.yaml` | 44/50 (0.88) | 6 / 44 |
| Energy | Consistency-only (coverage fallback) | `gating_energy_ebcar_consistency_only_sc050.yaml` | 14/50 (0.28) | 36 / 14 |
| Energy | Logit‑MI (experimental, thr=0.70) | `gating_energy_ebcar_logit_mi_sc009.yaml` | 46/50 (0.92) | 4 / 46 |
| Energy | Logit‑MI (experimental, thr=0.85) | `gating_energy_ebcar_logit_mi_sc009.yaml` | 22/50 (0.44) | 28 / 22 |
| Macro | Consistency-only (baseline) | `gating_macro_ebcar_consistency_only_sc050.yaml` | 9/50 (0.18) | 41 / 9 |
| Energy | MC Dropout + Consistency (safety) | `gating_energy_ebcar_mcdropout_consistency_sc050.yaml` | 31/50 (0.62) | 19 / 31 |
| Macro | MC Dropout + Consistency (safety) | `gating_macro_ebcar_mcdropout_consistency_sc050.yaml` | 36/50 (0.72) | 14 / 36 |

Note:
- Energy logit‑MI 50Q runs are now available for 0.70 / 0.85 / 0.90.
- Macro logit‑MI 50Q runs are available for 0.70 and 0.80 (same abstain rate on current set).

## Gating Ablation (50 Q)
| Domain | Strategy | Config | Abstain | Actions |
|---|---|---|---|---|
| Energy | nogate | `gating_energy_ebcar_nogate.yaml` | 0/50 (0.00) | none=50 |
| Energy | retrieve_more (default, epistemic) | `gating_energy_ebcar_logit_mi_sc009.yaml` | 35/50 (0.70) | none=15, retrieve_more=35 |
| Energy | retrieve_more (rep-MI, experimental) | `gating_energy_ebcar_rep_mi_sc004.yaml` | 45/50 (0.90) | none=5, retrieve_more=45 |
| Energy | retrieve_more (coverage fallback) | `gating_energy_ebcar_consistency_only_sc050.yaml` | 14/50 (0.28) | none=36, retrieve_more=14 |
| Energy | retrieve_more (safety) | `gating_energy_ebcar_mcdropout_consistency_sc050.yaml` | 31/50 (0.62) | none=19, retrieve_more=31 |
| Energy | retrieve_more (logit‑MI, experimental thr=0.70) | `gating_energy_ebcar_logit_mi_sc009.yaml` | 46/50 (0.92) | none=4, retrieve_more=46 |
| Energy | abstain | `gating_energy_ebcar_abstain.yaml` | 22/50 (0.44) | none=28, abstain=22 |
| Macro | nogate | `gating_macro_ebcar_nogate.yaml` | 0/50 (0.00) | none=50 |
| Macro | retrieve_more (default) | `gating_macro_ebcar_logit_mi_sc009.yaml` | 22/50 (0.44) | none=28, retrieve_more=22 |
| Macro | retrieve_more (rep-MI, experimental) | `gating_macro_ebcar_rep_mi_sc004.yaml` | 44/50 (0.88) | none=6, retrieve_more=44 |
| Macro | retrieve_more (baseline) | `gating_macro_ebcar_consistency_only_sc050.yaml` | 9/50 (0.18) | none=41, retrieve_more=9 |
| Macro | retrieve_more (safety) | `gating_macro_ebcar_mcdropout_consistency_sc050.yaml` | 36/50 (0.72) | none=14, retrieve_more=36 |
| Macro | abstain | `gating_macro_ebcar_abstain.yaml` | 23/50 (0.46) | none=27, abstain=23 |

## Qualitative Examples (Gating Decisions)
Energy (EIA/IRENA/Shell, EBCAR):
- q03 (sanity): "What is the focus of the IEO 2023 narrative report?" → action=retrieve_more, abstain=True
- q04 (sanity): "What is the stated purpose of the IRENA World Energy Transitions Outlook 2024?" → action=retrieve_more, abstain=True
- q06 (conflict): "AEO 2025 reference case vs IRENA WETO 2024: which is policy-neutral vs normative, and how does that change the fossil-fuel narrative?" → action=none, abstain=False
Interpretation:
- Sanity abstains indicate weak grounding in small corpus slices; retrieval escalation did not resolve.
- Conflict question was answered without abstention, suggesting clearer cross-report contrast.

Macro (World Bank + UN WESP + IMF, EBCAR):
- q02 (sanity): "What is the stated purpose of the UN World Economic Situation and Prospects 2025 report?" → action=retrieve_more, abstain=True
- q06 (conflict): "Compare the global growth outlook for 2025 across the reports: which is more optimistic?" → action=none, abstain=False
- q08 (conflict): "How do the reports differ on the expected path of inflation and monetary policy tightening?" → action=retrieve_more, abstain=True
Interpretation:
- Gating abstains on sanity when language is report-specific and retrieval is shallow.
- Conflict questions split: some answered, some abstained after retrieval expansion.

### Energy Outlooks Domain (EIA/IRENA/Shell) — Conflict-Focused Gating
Corpus (PDFs):
- `data/domain_energy/raw/eia_aeo2025_narrative.pdf`
- `data/domain_energy/raw/eia_ieo2023_narrative.pdf`
- `data/domain_energy/raw/eia_steo_full_2026_01.pdf`
- `data/domain_energy/raw/irena_weto_2024.pdf`
- `data/domain_energy/raw/shell_energy_transition_strategy_2024.pdf`

Indexing:
- `config/gating_energy_ebcar.yaml` → `./data/vector_db/energy_outlooks` (collection `rag_energy_outlooks`)
- 2,127 chunks indexed (fixed size 512, overlap 50).

Question set:
- `data/domain_energy/questions_energy_conflict.jsonl` (20 items, 15 conflict + 5 sanity).

Baseline vs EBCAR:
- Baseline (no reranker, `config/gating_energy_base.yaml`):
  - abstain: 2/20 (0.10), actions: none=18, retrieve_more=2
  - abstain_by_type: conflict=2, sanity=0
- EBCAR (main, `config/gating_energy_ebcar.yaml`):
  - thresholds: contradiction_rate=0.45, contradiction_prob=0.65, uncertainty=0.38
  - abstain: 7/20 (0.35), actions: none=13, retrieve_more=7
  - abstain_by_type: conflict=5, sanity=2

Interpretation:
- EBCAR is more conservative and flags conflict questions more often (desired for Domain‑B).
- Low‑abstain tuning (0.50/0.70/0.42) produced no change.
- High‑abstain tuning (0.40/0.60/0.35) over‑abstained (11/20).
- Keep `gating_energy_ebcar.yaml` as the active config; baseline kept for comparison.
- Ablation (energy, EBCAR):
  - Gating disabled (`gating_energy_ebcar_nogate.yaml`): abstain 0/20, actions none=20.
  - Gating retrieve_more (`gating_energy_ebcar.yaml`): abstain 7/20 (0.35), actions none=13, retrieve_more=7.

### Macro Outlooks Domain (World Bank + UN WESP) — Conflict-Focused Gating
Corpus (PDFs):
- `data/domain_macro/raw/worldbank_gep_2026_jan.pdf`
- `data/domain_macro/raw/un_wesp_2025.pdf`
- `data/domain_macro/raw/un_wesp_2025_midyear.pdf`

Notes:
- IMF WEO PDFs were initially blocked by the IMF CDN; later added via manual download.

Indexing:
- `config/gating_macro_ebcar.yaml` → `./data/vector_db/macro_outlooks` (collection `rag_macro_outlooks`)
- 3 PDFs → 467 pages → 3,873 chunks indexed (fixed size 512, overlap 50).

Question set:
- `data/domain_macro/questions_macro_conflict.jsonl` (20 items, 15 conflict + 5 sanity).

EBCAR (main, `config/gating_macro_ebcar.yaml`):
- thresholds: contradiction_rate=0.45, contradiction_prob=0.65, uncertainty=0.38
- abstain: 5/20 (0.25), actions: none=15, retrieve_more=5
- abstain_by_type: conflict=3, sanity=2

Baseline vs tuning:
- Baseline (no reranker, `config/gating_macro_base.yaml`):
  - abstain: 7/20 (0.35), actions: none=13, retrieve_more=7
  - abstain_by_type: conflict=5, sanity=2
- Low‑abstain tuning (0.50/0.70/0.42) matched EBCAR overall (5/20).
- High‑abstain tuning (0.40/0.60/0.35) over‑abstained (11/20).

After adding IMF WEO:
- Added `data/domain_macro/raw/imf_weo_2025_oct.pdf` (181 pages).
- New index size: 648 pages → 5,454 chunks.
- EBCAR eval (old thresholds): abstain 8/20 (0.40), actions none=12, retrieve_more=8,
  abstain_by_type conflict=5, sanity=3.
- Low‑abstain thresholds (0.50/0.70/0.42) improved coverage:
  abstain 4/20 (0.20), actions none=16, retrieve_more=4,
  abstain_by_type conflict=3, sanity=1.
- Ablation (IMF included, low‑abstain thresholds):
  - Gating disabled (`gating_macro_ebcar_nogate.yaml`): abstain 0/20, actions none=20.
  - Gating retrieve_more (`gating_macro_ebcar.yaml`): abstain 4/20 (0.20), actions none=16, retrieve_more=4.
  - Gating abstain (`gating_macro_ebcar_abstain.yaml`): abstain 7/20 (0.35), actions none=13, abstain=7.
