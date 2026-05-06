# FinReg Detector Eval v1

This is a small internal NLI-style evaluation set for the hallucination detector layer.

It is not a public benchmark and should not be used as a broad FinReg capability claim.
Its purpose is narrower: compare detector candidates before using them in FinReg gating.

## File

- `data/domain_finreg/detector_eval_finreg_v1.jsonl`

## Scope

- 36 premise/hypothesis pairs
- 12 entailment, 12 neutral, 12 contradiction
- Premises are short snippets derived from the current official FinReg corpus
- Sources cover BCBS, EBA, ECB, PRA/BoE, and Federal Reserve material

## Intended Use

Use this before changing the default detector:

1. Run detector-only evaluation with no LLM loaded.
2. Compare current FEVER detector against candidate LoRA assets.
3. Prefer candidates that improve contradiction recall without excessive false contradictions.
4. Only then run full RAG/gating ablations.

## Important Metrics

- `macro_f1`
- `contradiction_recall`
- `false_contradiction_rate`
- `entailment_recall`

For gating, high false contradiction rate is dangerous because it can cause unnecessary abstention
or misleading risk signals on actually supported answers.

## First Local Run

Command:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 venv312/bin/python scripts/evaluate_finreg_detector_set.py \
  --output-dir evaluation_results/finreg_detector_eval_v1
```

Results:

| model | accuracy | macro_f1 | contradiction_recall | false_contradiction_rate |
| --- | ---: | ---: | ---: | ---: |
| current_fever_deberta_v3_base | 0.3333 | 0.1702 | 0.0000 | 0.0417 |
| balanced_recovery_v2 | 0.3056 | 0.2447 | 0.4167 | 0.4167 |
| targeted_multipleqa_recovery_v2 | 0.2778 | 0.2202 | 0.3333 | 0.4167 |
| phase2_1_mixed_v2 | 0.2778 | 0.2210 | 0.3333 | 0.3750 |

Interpretation:

- The current FEVER detector is too conservative and misses contradictions.
- The asset detectors catch more contradictions but create too many false contradictions.
- None of these should become the hard-gate default yet.

## Threshold Sweep

Command:

```bash
venv312/bin/python scripts/sweep_finreg_detector_thresholds.py \
  --details evaluation_results/finreg_detector_eval_v1/details.jsonl \
  --output evaluation_results/finreg_detector_eval_v1/threshold_sweep.json
```

Selected operating points with `false_contradiction_rate <= 0.10`:

| model | selected_threshold | status | contradiction_recall | false_contradiction_rate | macro_f1 |
| --- | ---: | --- | ---: | ---: | ---: |
| current_fever_deberta_v3_base | 0.40 | safe_but_zero_contradiction_recall | 0.0000 | 0.0000 | 0.1667 |
| balanced_recovery_v2 | 0.50 | safe_with_nonzero_contradiction_recall | 0.0833 | 0.0417 | 0.2619 |
| targeted_multipleqa_recovery_v2 | 0.50 | safe_with_nonzero_contradiction_recall | 0.0833 | 0.0417 | 0.2070 |
| phase2_1_mixed_v2 | 0.60 | safe_but_zero_contradiction_recall | 0.0000 | 0.0000 | 0.1667 |

Conclusion:

- Thresholding reduces false contradictions, but then contradiction recall collapses.
- `balanced_recovery_v2` is the best shadow candidate, not a default detector.
- The next detector improvement should target FinReg-specific training or calibration, not just threshold tuning.

## FinReg Adaptation Smoke

A first continuation fine-tune was attempted with:

- config: `config/adamw_lora_finreg_detector_v1.yaml`
- data: `data/training/nli_dataset_finreg_detector_v1`
- warm start: `detector-assets/models/checkpoints/adamw_lora_balanced_recovery_v2/best_model`

Training data:

- train: 66 examples
- validation: 18 examples
- held-out test: this eval set, 36 examples

Validation result:

- best validation macro-F1: 0.1000
- model mostly collapsed toward neutral on validation

Held-out test comparison:

| model | accuracy | macro_f1 | contradiction_recall | false_contradiction_rate |
| --- | ---: | ---: | ---: | ---: |
| balanced_recovery_v2 | 0.3056 | 0.2447 | 0.4167 | 0.4167 |
| finreg_adapted_v1 | 0.3056 | 0.2447 | 0.4167 | 0.4167 |

Threshold sweep selected the same operating point for both:

- threshold: 0.50
- contradiction recall: 0.0833
- false contradiction rate: 0.0417

Interpretation:

- This first FinReg adaptation did not improve over the warm-start model.
- The adaptation dataset is too small and too template-like to move the detector meaningfully.
- Do not use `finreg_adapted_v1` as a default detector.
- Next attempt should either use substantially more FinReg-style NLI data or switch to a stronger base model / better domain-NLI source.

## Reviewed Candidate Adaptation

A second continuation fine-tune used corpus-derived candidate rows after a first manual/Codex review
pass.

Data flow:

- candidate pool: `data/domain_finreg/detector_candidate_pool_v11.jsonl`
- review set: `data/domain_finreg/manual_review/detector_v3/review_set.csv`
- reviewed first pass: `data/domain_finreg/manual_review/detector_v3/reviewed_set_codex_v1.csv`
- training data: `data/training/nli_dataset_finreg_detector_reviewed_v1`

Training split:

- selected reviewed rows: 66 balanced examples
- train: 54 examples, 18 per label
- validation: 12 examples, 4 per label
- held-out test: this eval set, 36 examples

Training result:

- config: `config/adamw_lora_finreg_detector_reviewed_v1.yaml`
- warm start: `detector-assets/models/checkpoints/adamw_lora_balanced_recovery_v2/best_model`
- best validation macro-F1: 0.7460 at epoch 2

Held-out test comparison:

| model | accuracy | macro_f1 | contradiction_recall | false_contradiction_rate |
| --- | ---: | ---: | ---: | ---: |
| current_fever_deberta_v3_base | 0.3333 | 0.1702 | 0.0000 | 0.0417 |
| balanced_recovery_v2 | 0.3056 | 0.2447 | 0.4167 | 0.4167 |
| finreg_reviewed_v1 | 0.3056 | 0.2440 | 0.4167 | 0.4583 |

Threshold sweep selected operating points:

| model | selected_threshold | status | contradiction_recall | false_contradiction_rate | macro_f1 |
| --- | ---: | --- | ---: | ---: | ---: |
| current_fever_deberta_v3_base | 0.40 | safe_but_zero_contradiction_recall | 0.0000 | 0.0000 | 0.1667 |
| balanced_recovery_v2 | 0.50 | safe_with_nonzero_contradiction_recall | 0.0833 | 0.0417 | 0.2619 |
| finreg_reviewed_v1 | 0.60 | safe_but_zero_contradiction_recall | 0.0000 | 0.0000 | 0.2215 |

Interpretation:

- The reviewed candidate data improved the tiny validation split, but did not transfer to held-out eval.
- `finreg_reviewed_v1` is not better than `balanced_recovery_v2` and has a higher false contradiction rate at the raw decision point.
- At a safe threshold, `finreg_reviewed_v1` loses all contradiction recall.
- Do not use `finreg_reviewed_v1` as the default detector.
- The current evidence says more of the same small auto-generated data is unlikely to solve the detector problem.

## RAG-Aware FinReg Adaptation

A larger pseudo-labeled RAG-aware dataset was generated from the current official FinReg corpus.
This data is closer to the detector call shape because the premise is a retrieved-context-like
window and the hypothesis is an answer-like claim.

Data:

- script: `scripts/build_finreg_detector_ragaware_dataset.py`
- active dataset: `data/training/nli_dataset_finreg_detector_ragaware_v2`
- train: 360 examples, 120 per label
- validation: 72 examples, 24 per label
- held-out test: this eval set, 36 examples

Small-model continuation:

- config: `config/adamw_lora_finreg_detector_ragaware_v1.yaml`
- warm start: `detector-assets/models/checkpoints/adamw_lora_balanced_recovery_v2/best_model`
- best validation macro-F1: 0.4889

Held-out test comparison:

| model | accuracy | macro_f1 | contradiction_recall | false_contradiction_rate |
| --- | ---: | ---: | ---: | ---: |
| current_fever_deberta_v3_base | 0.3333 | 0.1702 | 0.0000 | 0.0417 |
| balanced_recovery_v2 | 0.3056 | 0.2447 | 0.4167 | 0.4167 |
| finreg_reviewed_v1 | 0.3056 | 0.2440 | 0.4167 | 0.4583 |
| finreg_ragaware_v1 | 0.3056 | 0.2447 | 0.4167 | 0.4167 |

Interpretation:

- The larger RAG-aware pseudo data did not improve the small DeBERTa continuation model.
- The model behaved almost the same as `balanced_recovery_v2` on held-out eval.
- This suggests the main issue is not just quantity; the pseudo labels and base model matter.

## FEVER-Base FinReg Adaptation

A stronger adaptation was then attempted by starting from the existing local FEVER
`DeBERTa-v3-base` detector instead of the smaller recovery checkpoint.

RAG-aware FEVER-base run:

- config: `config/adamw_lora_finreg_detector_fever_ragaware_v1.yaml`
- base model: `electra_deberta/final_fever_deberta_v3_base_model`
- data: `data/training/nli_dataset_finreg_detector_ragaware_v2`
- best checkpoint: `models/checkpoints/adamw_lora_finreg_detector_fever_ragaware_v1/best_model`

Combined-data FEVER-base run:

- config: `config/adamw_lora_finreg_detector_fever_combo_v1.yaml`
- data: `data/training/nli_dataset_finreg_detector_combo_v1`
- combo data = RAG-aware v2 plus reviewed v1 train/val
- best checkpoint: `models/checkpoints/adamw_lora_finreg_detector_fever_combo_v1/best_model`

Held-out test comparison:

| model | accuracy | macro_f1 | contradiction_recall | false_contradiction_rate |
| --- | ---: | ---: | ---: | ---: |
| current_fever_deberta_v3_base | 0.3333 | 0.1702 | 0.0000 | 0.0417 |
| balanced_recovery_v2 | 0.3056 | 0.2447 | 0.4167 | 0.4167 |
| finreg_ragaware_v1 | 0.3056 | 0.2447 | 0.4167 | 0.4167 |
| finreg_fever_ragaware_v1 | 0.3333 | 0.3327 | 0.2500 | 0.1250 |
| finreg_fever_combo_v1 | 0.3056 | 0.2741 | 0.0833 | 0.1667 |

Fine-grained threshold sweep with `false_contradiction_rate <= 0.10`:

| model | selected_threshold | status | contradiction_recall | false_contradiction_rate | macro_f1 |
| --- | ---: | --- | ---: | ---: | ---: |
| current_fever_deberta_v3_base | 0.34 | safe_with_nonzero_contradiction_recall | 0.0833 | 0.0833 | 0.2074 |
| balanced_recovery_v2 | 0.49 | safe_with_nonzero_contradiction_recall | 0.0833 | 0.0417 | 0.2619 |
| finreg_ragaware_v1 | 0.50 | safe_with_nonzero_contradiction_recall | 0.0833 | 0.0417 | 0.2619 |
| finreg_fever_ragaware_v1 | 0.37 | safe_with_nonzero_contradiction_recall | 0.1667 | 0.0833 | 0.2970 |
| finreg_fever_combo_v1 | 0.37 | safe_with_nonzero_contradiction_recall | 0.0833 | 0.0833 | 0.2886 |

Current conclusion:

- `finreg_fever_ragaware_v1` is the strongest detector candidate so far on this held-out FinReg eval.
- It is meaningfully better than the previous small-model candidates at the safe operating point.
- It should still be treated as a candidate, not the default detector, until full RAG/gating ablations confirm that it improves answer behavior.
- The combo run did not help; adding the small reviewed set on top of RAG-aware pseudo data reduced held-out detector quality.
- Further detector gains likely require cleaner, eval-like FinReg NLI data rather than more noisy pseudo-labeled corpus sentences.
