# Phase 2.2 Detector Artifact Manifest

This PR intentionally keeps trained model weights, optimizer states, raw
training outputs, raw eval predictions, and vector DB binaries outside GitHub.
The external artifact bundle should be shared as `detector-assets-phase2.2`.

The Phase 2.2 checkpoints listed below were trained locally. The PR contains
the code, configs, summary metadata, and reproduce commands needed to connect
the external artifacts back to the implementation.

## External Artifact Layout

```text
detector-assets-phase2.2/
  models/checkpoints/
    adamw_lora_finregbench_phase2_2_mixed_v2_balanced/
    adamw_lora_finregbench_phase2_2_mixed_v3_balanced_recall/
    adamw_lora_finregbench_phase2_2_mixed_warmstart/
    adamw_lora_finregbench_phase2_2_prefill_warmstart/
    adamw_lora_targeted_multipleqa_phase2_1_mixed_v2/base_model_unwrapped/
  data/training/
    nli_dataset_finregbench_phase2_2/
    nli_dataset_finregbench_phase2_2_mixed/
    nli_dataset_finregbench_phase2_2_mixed_v2/
  FinRegBench/data/phase2_runs/
    smoke_300_export_mixed_v2/
    contradiction_stress_export_mixed_v2/
    smoke_300_hybrid_artifact_v1/
    contradiction_stress_hybrid_artifact_v1/
    review_180_detector_class_artifact_v1/
  README.md
  artifact_manifest_phase2_2.json
```

## Checkpoints

| Model | Artifact path | Config | Training data | Best metric |
| --- | --- | --- | --- | --- |
| `adamw_lora_finregbench_phase2_2_mixed_v2_balanced` | `models/checkpoints/adamw_lora_finregbench_phase2_2_mixed_v2_balanced/best_model` | `config/adamw_lora_finregbench_phase2_2_mixed_v2_balanced.yaml` | `data/training/nli_dataset_finregbench_phase2_2_mixed_v2` | `f1_macro=0.2254976064499874`, epoch 2 |
| `adamw_lora_finregbench_phase2_2_mixed_v3_balanced_recall` | `models/checkpoints/adamw_lora_finregbench_phase2_2_mixed_v3_balanced_recall/best_model` | `config/adamw_lora_finregbench_phase2_2_mixed_v3_balanced_recall.yaml` | `data/training/nli_dataset_finregbench_phase2_2_mixed_v2` | `f1_macro=0.1909375611665688`, epoch 2 |
| `adamw_lora_finregbench_phase2_2_mixed_warmstart` | `models/checkpoints/adamw_lora_finregbench_phase2_2_mixed_warmstart/best_model` | `config/adamw_lora_finregbench_phase2_2_mixed_warmstart.yaml` | `data/training/nli_dataset_finregbench_phase2_2_mixed` | `f1_macro=0.23823191733639493`, epoch 0 |
| `adamw_lora_finregbench_phase2_2_prefill_warmstart` | `models/checkpoints/adamw_lora_finregbench_phase2_2_prefill_warmstart/best_model` | `config/adamw_lora_finregbench_phase2_2_prefill_localexport.yaml` | `data/training/nli_dataset_finregbench_phase2_2` | `f1_macro=0.17391304347826086`, epoch 0 |

Each checkpoint `best_model` directory contains:

- `model.pt`
- `optimizer.pt`
- `training_state.pt`

The Phase 2.1 local base export is required at:

- `models/checkpoints/adamw_lora_targeted_multipleqa_phase2_1_mixed_v2/base_model_unwrapped`

That directory contains the full local model and tokenizer files:

- `pytorch_model.bin`
- `config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `added_tokens.json`
- `spm.model`

## Training Metadata

Training metadata is stored in each checkpoint's `training_state.pt`.
`trainer_state.json` was not present in these local checkpoint directories.

Common settings:

- Optimizer: `adamw`
- Scheduler: `linear`
- Mixed precision: `fp16`
- Base model/tokenizer: `models/checkpoints/adamw_lora_targeted_multipleqa_phase2_1_mixed_v2/base_model_unwrapped`
- Best checkpoint policy: `metric_for_best=f1_macro`, `mode=max`, `save_strategy=best`

## Eval Outputs

| Run | Rows | Three-way accuracy | Binary accuracy | Negative accuracy |
| --- | ---: | ---: | ---: | ---: |
| `FinRegBench/data/phase2_runs/smoke_300_export_mixed_v2` | 300 | 0.36 | 0.53 | 0.325 |
| `FinRegBench/data/phase2_runs/contradiction_stress_export_mixed_v2` | 520 | 1.0 | 1.0 | 1.0 |
| `FinRegBench/data/phase2_runs/smoke_300_hybrid_artifact_v1` | 300 | 0.9866666666666667 | 0.9966666666666667 | 0.98 |
| `FinRegBench/data/phase2_runs/contradiction_stress_hybrid_artifact_v1` | 520 | 0.9923076923076923 | 0.9942307692307693 | 0.9923076923076923 |
| `FinRegBench/data/phase2_runs/review_180_detector_class_artifact_v1` | 180 | 0.9944444444444445 | 1.0 | 0.9928571428571429 |

Raw `detector_inputs.jsonl`, `raw_predictions.jsonl`, and
`predictions_normalized.jsonl` files are included only in the external artifact
bundle.

## GitHub vs Drive Boundary

Keep these in GitHub:

- Code under `scripts/` and `src/`
- Configs under `config/`
- Small docs, summaries, and manifests
- Small metric summaries such as this document and `artifact_manifest_phase2_2.json`

Keep these in Drive:

- `model.pt`, `optimizer.pt`, `pytorch_model.bin`, `*.safetensors`
- Full checkpoint/export directories
- Raw training output directories
- Raw FinRegBench run directories and JSONL predictions
- Vector DB files such as `chroma.sqlite3`, `length.bin`, `data_level0.bin`, `header.bin`, and `link_lists.bin`
