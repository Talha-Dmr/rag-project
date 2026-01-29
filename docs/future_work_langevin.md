# Future Work: Langevin-Based Uncertainty for Adaptive RAG

## Goal
Implement a Langevin-based uncertainty mechanism for the RAG system, accepting the higher
complexity in exchange for stronger epistemic uncertainty estimates and a more rigorous
Bayesian framing.

## Executive Summary (Current Snapshot)
- Main uncertainty method for the paper: MC Dropout gating (stable and cost-effective).
- Final MC Dropout runs (50 Q):
  - Energy: abstain 13/50 (0.26), actions none=37, retrieve_more=13.
  - Macro: abstain 9/50 (0.18), actions none=41, retrieve_more=9.
- LoRA-SWAG is kept as ablation only; n=5 vs n=10 did not outperform MC Dropout on macro.
- Entropy-based SWAG gating was unstable (all-abstain vs none-abstain depending on threshold).
- Final configs: `gating_energy_ebcar_mcdropout.yaml`, `gating_macro_ebcar_mcdropout.yaml`.

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
Week 1 — Consolidate Evidence
- Freeze configs for energy + macro (current EBCAR settings).
- Summarize ablations in one table (baseline vs gating off vs retrieve_more vs abstain).
- Curate 5–10 qualitative conflict examples with short commentary.

Week 2 — Langevin Readiness
- Define success thresholds (ECE/coverage/abstain trade-off).
- Implement toy Langevin diagnostic (sanity check on mixing).
- Warm-start SGLD from AdamW LoRA checkpoint and re-evaluate.

Week 3 — End-to-End Validation
- Plug uncertainty into gating on AmbigQA-scale sample.
- Compare against proxy baselines (temperature ensemble/self-consistency).
- Draft experiment section (methods + results snapshot).

Must‑Have
- One strong ablation table + conflict examples for two domains.
- Clear success criteria and go/no-go for full Langevin.

Nice‑to‑Have
- Extra domain or larger sample sizes.
- Visualization of trade-off curves (coverage vs abstain).

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
- `config/gating_energy_ebcar_swag.yaml` (Energy gating with SWAG uncertainty).
- `config/gating_energy_ebcar_swag_ns10.yaml` (Energy SWAG sensitivity, n=10).
- `config/gating_energy_ebcar_swag_entropy.yaml` (Energy SWAG gating using entropy).
- `config/gating_macro_ebcar_swag.yaml` (Macro gating with SWAG uncertainty).
- `config/gating_macro_ebcar_swag_entropy.yaml` (Macro SWAG gating using entropy).
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
- 1D target N(0,1), eta=0.01, burn-in=1000, steps=6000

Result:
- mean ≈ 0.142, std ≈ 1.043 (close to N(0,1))

Note:
- Mean offset likely due to short chain; longer chain or smaller eta should reduce bias.
- Longer chain checks:
  - eta=0.005, burn-in=5000, steps=30000 → mean ≈ -0.037, std ≈ 0.839
  - eta=0.01, burn-in=5000, steps=30000 → mean ≈ -0.026, std ≈ 0.861
- Interpretation: unadjusted Langevin shows discretization bias in variance; consider smaller eta and/or MALA if strict correctness is required.
- MALA sanity (eta=0.05, burn-in=2000, steps=20000): mean ≈ 0.026, std ≈ 1.009, accept ≈ 1.00
- MALA sanity (eta=0.1, burn-in=2000, steps=20000): mean ≈ 0.023, std ≈ 1.010, accept ≈ 0.99

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
| AdamW LoRA sanity | 0.7780 | 0.7404 | 0.7471 | 0.0209 | 0.2694 |
| SGLD warm-start (noise=5e-5) | 0.7780 | 0.7404 | 0.7471 | 0.0194 | 0.2693 |

Notes:
- SGLD warm-start gives a small ECE improvement; other metrics are effectively identical.

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

## Final Recommendation (Current Snapshot)
- Use MC Dropout gating as the main uncertainty method for the paper.
- Energy: `gating_energy_ebcar_mcdropout.yaml` with uncertainty_threshold=0.42 (balanced abstain 0.20).
- Macro: `gating_macro_ebcar_mcdropout.yaml` (abstain 0.18); SWAG n=5 shows no gain.
- Keep LoRA-SWAG results as an appendix/ablation (n=5 vs n=10).

Final runs (Energy/Macro, 50 Q, MC Dropout):
- Energy: abstain 13/50 (0.26), actions none=37, retrieve_more=13.
- Macro: abstain 9/50 (0.18), actions none=41, retrieve_more=9.

## Final Results Table (50 Q)
| Domain | Method | Config | Abstain | Actions (none/retrieve_more) |
|---|---|---|---|---|
| Energy | MC Dropout (final) | `gating_energy_ebcar_mcdropout.yaml` | 13/50 (0.26) | 37 / 13 |
| Macro | MC Dropout (final) | `gating_macro_ebcar_mcdropout.yaml` | 9/50 (0.18) | 41 / 9 |

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
- IMF WEO PDFs were blocked by the IMF CDN in this environment; skipped for now.

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
