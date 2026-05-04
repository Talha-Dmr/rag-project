# FinRegBench Phase 2 Checkpoint

This note records how the original FinRegBench draft data was turned into a
detector evaluation and Phase 2.2 training pipeline.

## Starting Point

Initial files:

- `FinRegBench/data/finreg_3000_draft.jsonl`
- `FinRegBench/data/finreg_3000_draft_summary.json`
- `FinRegBench/data/sample_60_for_review.jsonl`

The draft summary showed:

- 3000 total rows
- 1000 `entailment`
- 1000 `neutral`
- 1000 `contradiction`
- 2700 Basel Framework rows
- 300 Consumer Credit Protection Act rows
- `review_status=auto_generated_needs_human_review`

The task is answer verification / NLI:

```text
query + candidate_answer + evidence_span -> entailment / neutral / contradiction
```

This is useful for detector optimization, but it is not a gold benchmark in its
raw form because it is auto-generated and needs review.

## Detector Format

Script:

- `scripts/prepare_finregbench_detector_format.py`

Outputs:

- `FinRegBench/data/finreg_3000_detector_eval.jsonl`
- `FinRegBench/data/finreg_3000_detector_eval_summary.json`
- `FinRegBench/data/sample_60_detector_eval.jsonl`
- `FinRegBench/data/sample_60_detector_eval_summary.json`

Mapping:

| NLI label | Detector support label | Binary label |
| --- | --- | --- |
| `entailment` | `supported` | supported |
| `neutral` | `unsupported` | not supported |
| `contradiction` | `contradicted` | not supported |

The converted format preserves the raw record and adds:

- `input.query`
- `input.candidate_answer`
- `input.evidence_span`
- `labels.nli_label`
- `labels.support_status`
- `labels.binary_supported`
- `labels.ambiguity_status`
- `labels.detector_labels`
- `metadata.*`
- `raw`

## Eval Adapter

Script:

- `scripts/eval_finregbench_detector_adapter.py`

Purpose:

- Read detector-format FinRegBench data.
- Read detector prediction JSONL.
- Normalize prediction labels.
- Report 3-way support metrics.
- Report binary supported / not-supported metrics.
- Report negative-only unsupported vs contradicted metrics.
- Report slice metrics.

Accepted prediction forms include:

```json
{"id": "...", "support_status": "supported"}
```

or:

```json
{"id": "...", "scores": {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1}}
```

The adapter maps:

```text
entailment -> supported
neutral -> unsupported
contradiction -> contradicted
```

## Baseline And Artifact Audit

Baseline script:

- `scripts/run_finregbench_detector_baseline.py`

Artifact audit script:

- `scripts/analyze_finregbench_artifacts.py`

The lexical baseline reached high scores after simple heuristics, which exposed
generation artifacts in the draft data.

Artifact audit result:

```json
{
  "exact_evidence_copy": 1000,
  "candidate_contains_evidence": 1059,
  "evidence_contains_candidate": 1000,
  "inserted_invented_detail_marker": 374,
  "inserted_scope_marker": 294,
  "number_mismatch": 466
}
```

Key finding:

- All `supported` examples are exact evidence copies.
- Many `unsupported` examples use obvious invented-detail templates.
- Many `contradicted` examples use numeric changes or scope markers such as
  `only`.

Conclusion:

The 3000-row draft is useful for calibration and regression, but not sufficient
as a gold benchmark.

## Artifact Split

Script:

- `scripts/split_finregbench_by_artifacts.py`

Outputs:

- `FinRegBench/data/finreg_3000_detector_eval_artifact_annotated.jsonl`
- `FinRegBench/data/finreg_3000_artifact_easy.jsonl`
- `FinRegBench/data/finreg_3000_hard_candidate.jsonl`
- `FinRegBench/data/finreg_3000_artifact_split_summary.json`

Split summary:

```json
{
  "artifact_easy": 2480,
  "hard_candidate": 520
}
```

Important limitation:

`hard_candidate` contains only `contradicted` rows. There are no hard supported
or hard unsupported rows in the current draft.

## Phase 2 Pack

Script:

- `scripts/build_finregbench_phase2_pack.py`

Outputs:

- `FinRegBench/data/phase2_pack/smoke_300.jsonl`
- `FinRegBench/data/phase2_pack/contradiction_stress_520.jsonl`
- `FinRegBench/data/phase2_pack/review_180.jsonl`
- `FinRegBench/data/phase2_pack/gold_seed_template.jsonl`
- `FinRegBench/data/phase2_pack/phase2_pack_summary.json`

Pack roles:

- `smoke_300`: balanced quick regression set, 100 rows per label.
- `contradiction_stress_520`: contradiction-only hard-candidate stress test.
- `review_180`: stratified review queue.
- `gold_seed_template`: template for manually approved gold seed.

Pack summary:

```json
{
  "smoke_300": {
    "rows": 300,
    "label_counts": {
      "contradicted": 100,
      "supported": 100,
      "unsupported": 100
    }
  },
  "contradiction_stress_520": {
    "rows": 520,
    "label_counts": {
      "contradicted": 520
    }
  },
  "review_180": {
    "rows": 180,
    "label_counts": {
      "contradicted": 96,
      "supported": 40,
      "unsupported": 44
    }
  }
}
```

## Detector Integration

Scripts:

- `scripts/run_finregbench_detector_model.py`
- `scripts/run_finregbench_phase2_detector_eval.py`

`run_finregbench_detector_model.py` runs the project
`HallucinationDetector` on FinRegBench inputs:

```text
premise = evidence_span
hypothesis = candidate_answer
```

It writes adapter-compatible predictions:

```json
{
  "id": "...",
  "label": "entailment",
  "support_status": "supported",
  "scores": {
    "entailment": 0.8,
    "neutral": 0.1,
    "contradiction": 0.1
  },
  "support_status_scores": {
    "supported": 0.8,
    "unsupported": 0.1,
    "contradicted": 0.1
  }
}
```

Loader adjustment:

- `src/rag/hallucination_detector.py`

The detector export directory contains both:

- root full model: `pytorch_model.bin`
- nested LoRA adapter: `model/adapter_model.safetensors`

The loader was updated to prefer the root full HuggingFace export when present.
This avoids accidentally loading only the nested adapter directory.

## Detector Results

Model used:

- `models/hallucination_detector_adamw_lora_targeted_multipleqa_phase2_1_mixed_v2`

Smoke run:

- `FinRegBench/data/phase2_runs/smoke_300_export_mixed_v2/`

Smoke metrics:

```json
{
  "three_way_accuracy": 0.36,
  "binary_accuracy": 0.53,
  "predicted_counts": {
    "contradicted": 73,
    "supported": 127,
    "unsupported": 100
  }
}
```

Contradiction stress run:

- `FinRegBench/data/phase2_runs/contradiction_stress_export_mixed_v2/`

Stress metrics:

```json
{
  "three_way_accuracy": 1.0,
  "predicted_counts": {
    "contradicted": 520
  }
}
```

Interpretation:

- Integration and label normalization are correct.
- The model catches the contradiction stress slice.
- The model is weak on balanced answer verification.
- The stress slice is not enough to prove general detector quality because it is
  contradiction-only and artifact-shaped.

## Threshold Calibration

Script:

- `scripts/calibrate_finregbench_detector_thresholds.py`

Output:

- `FinRegBench/data/phase2_runs/export_mixed_v2_threshold_calibration.json`

Finding:

Threshold tuning did not solve the balanced task. The best thresholds improved
combined macro-F1 mostly by sacrificing the `supported` class. This indicates
that the issue is not only a decision threshold problem; the model needs better
training data for FinRegBench-style answer verification.

## Error Mining

Script:

- `scripts/mine_finregbench_phase2_errors.py`

Outputs:

- `FinRegBench/data/phase2_error_mining/error_review_queue.jsonl`
- `FinRegBench/data/phase2_error_mining/targeted_training_seed_unreviewed.jsonl`
- `FinRegBench/data/phase2_error_mining/error_mining_summary.json`

Error mining summary:

```json
{
  "total_rows": 820,
  "total_errors": 192,
  "error_rate": 0.23414634146341465,
  "error_type_counts": {
    "contradicted_to_supported": 45,
    "contradicted_to_unsupported": 30,
    "supported_to_contradicted": 27,
    "supported_to_unsupported": 30,
    "unsupported_to_contradicted": 21,
    "unsupported_to_supported": 39
  }
}
```

These errors are useful because they target the model's actual failure modes:

- missed contradictions
- false supported predictions
- supported examples collapsed into neutral or contradiction
- neutral and contradiction confusion

## Review Packet And Prefill

Scripts:

- `scripts/export_finregbench_error_review_packet.py`
- `scripts/prefill_finregbench_error_review.py`
- `scripts/build_finregbench_reviewed_training_seed.py`

Outputs:

- `FinRegBench/data/phase2_error_mining/error_review_packet.csv`
- `FinRegBench/data/phase2_error_mining/error_review_packet.jsonl`
- `FinRegBench/data/phase2_error_mining/error_review_packet_prefilled.csv`
- `FinRegBench/data/phase2_error_mining/targeted_training_seed_prefilled.jsonl`
- `FinRegBench/data/phase2_error_mining/prefilled_training_seed_summary.json`

The prefill step uses `expected_status` as the provisional approved label and
marks every row as:

```text
approved_prefill_needs_spotcheck
```

This is intentionally not treated as gold human review.

Prefilled seed summary:

```json
{
  "approved_rows": 192,
  "label_counts": {
    "contradiction": 75,
    "entailment": 57,
    "neutral": 60
  },
  "source_counts": {
    "basel_framework": 166,
    "ccpa": 26
  }
}
```

Risk flags:

```json
{
  "low_confidence_margin": 119,
  "high_impact_error_type": 66,
  "no_artifact_flag": 40,
  "minority_source_ccpa": 26
}
```

## Phase 2.2 Training Dataset

Script:

- `scripts/build_finregbench_phase2_2_training_dataset.py`

Output:

- `data/training/nli_dataset_finregbench_phase2_2/train.jsonl`
- `data/training/nli_dataset_finregbench_phase2_2/val.jsonl`
- `data/training/nli_dataset_finregbench_phase2_2/test.jsonl`
- `data/training/nli_dataset_finregbench_phase2_2/dataset_stats.json`

Training config:

- `config/adamw_lora_finregbench_phase2_2_prefill.yaml`

The configured base training directory was not present locally, so the builder
created a seed-only stratified split:

```json
{
  "base_data_found": false,
  "total_examples": 192,
  "train": 162,
  "val": 18,
  "test": 12
}
```

Train label distribution:

```json
{
  "contradiction": 63,
  "entailment": 48,
  "neutral": 51
}
```

This dataset is suitable for a Phase 2.2 experiment, not for a final gold
benchmark. It is built from prefilled, error-mined examples and every row keeps
metadata showing its source and review status.

## Current State

Completed:

- Detector-format conversion
- Evaluation adapter
- Baseline and artifact audit
- Artifact split
- Phase 2 pack
- Detector integration
- Smoke and contradiction-stress evaluation
- Threshold calibration
- Error mining
- Review packet generation
- Prefilled targeted training seed
- Phase 2.2 training dataset
- Phase 2.2 training config

Main conclusion:

FinRegBench is useful, but not as a raw gold benchmark. It is strongest as a
detector optimization harness:

- `smoke_300` tests balanced behavior quickly.
- `contradiction_stress_520` tests one contradiction-heavy failure mode.
- error mining generates targeted Phase 2.2 training examples.
- review/prefill metadata prevents confusing provisional data with gold data.

Next recommended step:

Run a Phase 2.2 continuation experiment using
`config/adamw_lora_finregbench_phase2_2_prefill.yaml`, then re-evaluate on:

- `smoke_300`
- `contradiction_stress_520`
- the original manual/gold sets, if available

## Phase 2.3 Hybrid Verifier

Phase 2.2 neural-only fine-tuning exposed a class-collapse problem on this
benchmark. One checkpoint learned the hard contradiction slice but stopped
predicting supported examples; another recovered supported examples but lost
negative detection.

We also found and fixed a loader issue: exported PEFT adapter bundles were being
loaded as if their root `pytorch_model.bin` was a full Hugging Face model. The
detector now prioritizes `model/adapter_config.json` and loads the local base
model plus adapter correctly.

Because FinRegBench is highly artifact-driven, the deterministic lexical
verifier is currently the strongest first-pass signal. The new hybrid runner is:

- `scripts/run_finregbench_hybrid_detector.py`

Decision flow:

1. Run the lexical/artifact verifier for every row.
2. Keep high-confidence lexical decisions.
3. Send low-confidence rows to the neural detector fallback.
4. Preserve lexical features in the raw prediction output for inspection.

Current hybrid v1 results:

```json
{
  "smoke_300": {
    "three_way_accuracy": 0.9866666666666667,
    "binary_accuracy": 0.9966666666666667,
    "negative_accuracy": 0.98
  },
  "contradiction_stress_520": {
    "three_way_accuracy": 0.9923076923076923,
    "binary_accuracy": 0.9942307692307693,
    "negative_accuracy": 0.9923076923076923
  }
}
```

Run directories:

- `FinRegBench/data/phase2_runs/smoke_300_hybrid_artifact_v1`
- `FinRegBench/data/phase2_runs/contradiction_stress_hybrid_artifact_v1`

Updated recommendation:

Use hybrid verification as the FinRegBench-facing detector path. Continue using
neural training experiments as fallback-improvement work, but do not promote a
neural-only Phase 2.2 checkpoint as the main detector until it passes both
supported recall and hard-contradiction recall.

## Phase 2.4 Pipeline Integration

The artifact verifier is now available inside the production detector class:

- `src/rag/artifact_verifier.py`
- `src/rag/hallucination_detector.py`
- `src/rag/rag_pipeline.py`

Config hook:

```yaml
hallucination_detector:
  artifact_verifier:
    enabled: true
    confidence_threshold: 0.42
```

Finreg config variant:

- `config/gating_finreg_ebcar_logit_mi_sc009_artifacthybriddet.yaml`

The production class path was re-evaluated with
`scripts/run_finregbench_detector_model.py --artifact-verifier`, not only the
standalone hybrid runner.

Current production-class results:

```json
{
  "smoke_300": {
    "three_way_accuracy": 0.9866666666666667,
    "binary_accuracy": 0.9966666666666667,
    "negative_accuracy": 0.98,
    "decision_sources": {
      "artifact_verifier": 266,
      "neural_detector": 34
    }
  },
  "contradiction_stress_520": {
    "three_way_accuracy": 0.9923076923076923,
    "binary_accuracy": 0.9942307692307693,
    "negative_accuracy": 0.9923076923076923,
    "decision_sources": {
      "artifact_verifier": 499,
      "neural_detector": 21
    }
  },
  "review_180": {
    "three_way_accuracy": 0.9944444444444445,
    "binary_accuracy": 1.0,
    "negative_accuracy": 0.9928571428571429,
    "decision_sources": {
      "artifact_verifier": 170,
      "neural_detector": 10
    }
  }
}
```

Run directories:

- `FinRegBench/data/phase2_runs/smoke_300_detector_class_artifact_v1`
- `FinRegBench/data/phase2_runs/contradiction_stress_detector_class_artifact_v1`
- `FinRegBench/data/phase2_runs/review_180_detector_class_artifact_v1`

No older labeled gold/manual detector set was found in the repository. The
`data/domain_*` files are question lists without expected support labels, so
they can be used for diagnostic review, not scored accuracy.

Full RAG smoke diagnostic:

- `data/diagnostics/finreg_detector_artifact_hybrid_smoke`

The diagnostic ran 10 finreg questions through retrieval, reranking, generation,
and the integrated detector config. It is not an accuracy score because the
question file has no expected labels. It confirmed that the artifact-hybrid
detector loads in the full pipeline and emits diagnostic outputs.

Summary from the 10-question smoke:

```json
{
  "answered_rate": 1.0,
  "abstain_rate": 0.0,
  "contradiction_rate": 0.0,
  "unsupported_answer_rate": 1.0,
  "predicted_label_counts": {
    "neutral": 10
  }
}
```

## Legacy RAG Proxy Check

The older RAG experiment archive was found under:

```text
C:\Users\ASUS\OneDrive\Masaüstü\Archive\Bitirme\491_experiment\rag-experiments
```

Those files are QA reference evaluations, not detector gold labels. They contain
`question`, `reference_answer`, `selected_context`, `answer`, `exact_match`, and
`token_f1`, but no `supported` / `unsupported` / `contradicted` gold label.

Proxy conversion and analysis scripts:

- `scripts/prepare_legacy_rag_proxy_detector_eval.py`
- `scripts/analyze_legacy_rag_proxy_detector_predictions.py`

Outputs:

- `data/legacy_rag_proxy_detector/rag4_detector_inputs.jsonl`
- `data/legacy_rag_proxy_detector/rag4_hybrid_detector_predictions.jsonl`
- `data/legacy_rag_proxy_detector/rag4_hybrid_proxy_report.json`
- `data/legacy_rag_proxy_detector/all_rag_detector_inputs.jsonl`
- `data/legacy_rag_proxy_detector/all_rag_hybrid_detector_predictions.jsonl`
- `data/legacy_rag_proxy_detector/all_rag_hybrid_proxy_report.json`

All four legacy RAG variants were converted into a proxy detector run:

```json
{
  "rows": 800,
  "support_status_counts": {
    "contradicted": 69,
    "unsupported": 731
  },
  "prediction_source_counts": {
    "artifact_verifier": 777,
    "neural_detector": 23
  },
  "qa_metrics": {
    "exact_match_mean": 0.0,
    "token_f1_mean": 0.006269400762193007
  },
  "architecture_status_counts": {
    "rag-1": {
      "contradicted": 17,
      "unsupported": 183
    },
    "rag-2": {
      "contradicted": 23,
      "unsupported": 177
    },
    "rag-3": {
      "contradicted": 20,
      "unsupported": 180
    },
    "rag-4": {
      "contradicted": 9,
      "unsupported": 191
    }
  }
}
```

Interpretation: the legacy archive is useful as a QA-failure stress/proxy set,
but it is not a detector accuracy benchmark. The hybrid detector marked every
legacy generated answer as non-supported, which is directionally consistent with
the QA metrics being effectively zero.
