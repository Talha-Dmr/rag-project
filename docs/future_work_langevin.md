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
| Energy outlooks | 5 PDFs | `gating_energy_ebcar.yaml` | 0.45 / 0.65 / 0.38 | 6/20 (0.30) | 14 / 6 |
| Macro outlooks | 4 PDFs (incl. IMF) | `gating_macro_ebcar.yaml` | 0.50 / 0.70 / 0.42 | 3/20 (0.15) | 17 / 3 |

Notes:
- Energy domain: EBCAR more conservative vs baseline; flags conflict questions more often.
- Macro domain: IMF WEO increased abstain at old thresholds; low‑abstain tuning restored coverage.

## Qualitative Examples (Gating Decisions)
Energy (EIA/IRENA/Shell, EBCAR):
- q03 (sanity): "What is the focus of the IEO 2023 narrative report?" → action=retrieve_more, abstain=True
- q04 (sanity): "What is the stated purpose of the IRENA World Energy Transitions Outlook 2024?" → action=retrieve_more, abstain=True
- q06 (conflict): "AEO 2025 reference case vs IRENA WETO 2024: which is policy-neutral vs normative, and how does that change the fossil-fuel narrative?" → action=none, abstain=False

Macro (World Bank + UN WESP + IMF, EBCAR):
- q02 (sanity): "What is the stated purpose of the UN World Economic Situation and Prospects 2025 report?" → action=retrieve_more, abstain=True
- q06 (conflict): "Compare the global growth outlook for 2025 across the reports: which is more optimistic?" → action=none, abstain=False
- q08 (conflict): "How do the reports differ on the expected path of inflation and monetary policy tightening?" → action=retrieve_more, abstain=True

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
  - abstain: 6/20 (0.30), actions: none=14, retrieve_more=6
  - abstain_by_type: conflict=4, sanity=2

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
  abstain 3/20 (0.15), actions none=17, retrieve_more=3,
  abstain_by_type conflict=2, sanity=1.
- Ablation (IMF included, low‑abstain thresholds):
  - Gating disabled (`gating_macro_ebcar_nogate.yaml`): abstain 0/20, actions none=20.
  - Gating retrieve_more (`gating_macro_ebcar.yaml`): abstain 5/20 (0.25), actions none=15, retrieve_more=5.
  - Gating abstain (`gating_macro_ebcar_abstain.yaml`): abstain 8/20 (0.40), actions none=12, abstain=8.
