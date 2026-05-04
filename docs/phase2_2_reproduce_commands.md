# Phase 2.2 Reproduce Commands

Run these commands from the repository root after restoring the external
`detector-assets-phase2.2` bundle into the repository layout.

The local runs used the Phase 2.1 unwrapped export as the base model:

```text
models/checkpoints/adamw_lora_targeted_multipleqa_phase2_1_mixed_v2/base_model_unwrapped
```

## Training

```powershell
python scripts/train_hallucination_model.py `
  --config config/adamw_lora_finregbench_phase2_2_prefill_localexport.yaml `
  --data-dir data/training/nli_dataset_finregbench_phase2_2 `
  --output-dir models/checkpoints/adamw_lora_finregbench_phase2_2_prefill_warmstart `
  --init-from models/checkpoints/adamw_lora_targeted_multipleqa_phase2_1_mixed_v2/best_model
```

```powershell
python scripts/train_hallucination_model.py `
  --config config/adamw_lora_finregbench_phase2_2_mixed_warmstart.yaml `
  --data-dir data/training/nli_dataset_finregbench_phase2_2_mixed `
  --output-dir models/checkpoints/adamw_lora_finregbench_phase2_2_mixed_warmstart `
  --init-from models/checkpoints/adamw_lora_finregbench_phase2_2_prefill_warmstart/best_model
```

```powershell
python scripts/train_hallucination_model.py `
  --config config/adamw_lora_finregbench_phase2_2_mixed_v2_balanced.yaml `
  --data-dir data/training/nli_dataset_finregbench_phase2_2_mixed_v2 `
  --output-dir models/checkpoints/adamw_lora_finregbench_phase2_2_mixed_v2_balanced `
  --init-from models/checkpoints/adamw_lora_finregbench_phase2_2_mixed_warmstart/best_model
```

```powershell
python scripts/train_hallucination_model.py `
  --config config/adamw_lora_finregbench_phase2_2_mixed_v3_balanced_recall.yaml `
  --data-dir data/training/nli_dataset_finregbench_phase2_2_mixed_v2 `
  --output-dir models/checkpoints/adamw_lora_finregbench_phase2_2_mixed_v3_balanced_recall `
  --init-from models/checkpoints/adamw_lora_finregbench_phase2_2_mixed_warmstart/best_model
```

## Eval

Neural export evals:

```powershell
python scripts/run_finregbench_detector_model.py `
  --input FinRegBench/data/smoke_300_detector_inputs.jsonl `
  --output-dir FinRegBench/data/phase2_runs/smoke_300_export_mixed_v2 `
  --model-path models/hallucination_detector_adamw_lora_targeted_multipleqa_phase2_1_mixed_v2
```

```powershell
python scripts/run_finregbench_detector_model.py `
  --input FinRegBench/data/contradiction_stress_detector_inputs.jsonl `
  --output-dir FinRegBench/data/phase2_runs/contradiction_stress_export_mixed_v2 `
  --model-path models/hallucination_detector_adamw_lora_targeted_multipleqa_phase2_1_mixed_v2
```

Production-class artifact verifier evals:

```powershell
python scripts/run_finregbench_detector_model.py `
  --input FinRegBench/data/smoke_300_detector_inputs.jsonl `
  --output-dir FinRegBench/data/phase2_runs/smoke_300_hybrid_artifact_v1 `
  --artifact-verifier
```

```powershell
python scripts/run_finregbench_detector_model.py `
  --input FinRegBench/data/contradiction_stress_detector_inputs.jsonl `
  --output-dir FinRegBench/data/phase2_runs/contradiction_stress_hybrid_artifact_v1 `
  --artifact-verifier
```

```powershell
python scripts/run_finregbench_detector_model.py `
  --input FinRegBench/data/review_180_detector_inputs.jsonl `
  --output-dir FinRegBench/data/phase2_runs/review_180_detector_class_artifact_v1 `
  --artifact-verifier
```

## Environment Notes

The local checkpoint metadata records `fp16` mixed precision training with the
`adamw` optimizer and a linear scheduler. GPU/CPU inventory was not serialized
in `training_state.pt`; record the exact device name separately when uploading
the Drive artifact if that is needed for audit.
