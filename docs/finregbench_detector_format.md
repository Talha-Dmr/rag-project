# FinRegBench Detector Eval Format

`FinRegBench/data/finreg_3000_draft.jsonl` is an auto-generated
answer-verification dataset.  The original labels are NLI-style:

- `entailment`: candidate answer is supported by the evidence span
- `contradiction`: candidate answer conflicts with the evidence span
- `neutral`: candidate answer is not stated by the evidence span

For detector optimization, keep the original NLI label and add a canonical
detector label layer:

| Original label | `support_status` | `binary_supported` | Extra detector labels |
| --- | --- | --- | --- |
| `entailment` | `supported` | `true` | `supported` |
| `contradiction` | `contradicted` | `false` | `contradicted`, `conflicting_answer`, `needs_review_or_retrieval` |
| `neutral` | `unsupported` | `false` | `unsupported`, `missing_evidence`, `needs_review_or_retrieval` |

Ambiguity is represented separately:

- `ambiguity_status=unambiguous` when `ambiguity_type` is empty or `none`
- `ambiguity_status=ambiguous` for any other `ambiguity_type`

This avoids collapsing two different failure modes:

- `contradicted`: the answer says something incompatible with evidence
- `unsupported`: the answer may be plausible, but the evidence does not state it

## Converter

Run:

```bash
python scripts/prepare_finregbench_detector_format.py
```

Default outputs:

- `FinRegBench/data/finreg_3000_detector_eval.jsonl`
- `FinRegBench/data/finreg_3000_detector_eval_summary.json`

To convert the review sample:

```bash
python scripts/prepare_finregbench_detector_format.py \
  --input FinRegBench/data/sample_60_for_review.jsonl \
  --output FinRegBench/data/sample_60_detector_eval.jsonl \
  --summary FinRegBench/data/sample_60_detector_eval_summary.json
```

## Canonical Record Shape

Each converted row has:

- `input.query`
- `input.candidate_answer`
- `input.evidence_span`
- `labels.nli_label`
- `labels.support_status`
- `labels.binary_supported`
- `labels.ambiguity_status`
- `labels.ambiguity_type`
- `labels.detector_labels`
- `metadata.*`
- `raw`

The `raw` object preserves the original source row for traceability.

## Detector Adapter

Use the adapter to export model inputs:

```bash
python scripts/eval_finregbench_detector_adapter.py \
  --dataset FinRegBench/data/finreg_3000_detector_eval.jsonl \
  --export-inputs FinRegBench/data/finreg_3000_detector_inputs.jsonl
```

The detector should return a JSONL prediction file with one row per `id`.
Accepted direct label fields:

- `support_status`
- `predicted_support_status`
- `prediction`
- `predicted_label`
- `label`
- `nli_label`

Accepted labels and aliases include:

- `supported`, `entailment`, `true`
- `unsupported`, `neutral`, `not_supported`, `missing_evidence`, `false`
- `contradicted`, `contradiction`, `conflicting_answer`, `conflict`

The adapter also accepts score dictionaries in `support_status_scores`, `scores`,
or `label_scores`; the highest-scoring label is used.

Prediction example:

```json
{"id": "finreg3000_neutral_2522_8bfdd9f0ba21", "support_status": "unsupported"}
```

Score example:

```json
{"id": "finreg3000_neutral_2522_8bfdd9f0ba21", "support_status_scores": {"supported": 0.05, "unsupported": 0.88, "contradicted": 0.07}}
```

Evaluate:

```bash
python scripts/eval_finregbench_detector_adapter.py \
  --dataset FinRegBench/data/finreg_3000_detector_eval.jsonl \
  --predictions FinRegBench/data/finreg_3000_detector_predictions.jsonl \
  --report FinRegBench/data/finreg_3000_detector_eval_report.json \
  --slice-csv FinRegBench/data/finreg_3000_detector_eval_slices.csv
```

## Baseline Runner

Use the baseline runner to produce a deterministic local reference score:

```bash
python scripts/eval_finregbench_detector_adapter.py \
  --dataset FinRegBench/data/finreg_3000_detector_eval.jsonl \
  --export-inputs FinRegBench/data/finreg_3000_detector_inputs.jsonl

python scripts/run_finregbench_detector_baseline.py \
  --inputs FinRegBench/data/finreg_3000_detector_inputs.jsonl \
  --output FinRegBench/data/finreg_3000_detector_predictions_baseline.jsonl

python scripts/eval_finregbench_detector_adapter.py \
  --dataset FinRegBench/data/finreg_3000_detector_eval.jsonl \
  --predictions FinRegBench/data/finreg_3000_detector_predictions_baseline.jsonl \
  --report FinRegBench/data/finreg_3000_detector_baseline_report.json \
  --slice-csv FinRegBench/data/finreg_3000_detector_baseline_slices.csv
```

The baseline uses lexical overlap, exact span matching, number mismatch, and
negation mismatch.  It is only a reference point; real detector results should
replace its prediction file.

## Artifact Audit

Because the draft benchmark is auto-generated, run the artifact audit before
interpreting detector scores as model quality:

```bash
python scripts/analyze_finregbench_artifacts.py \
  --dataset FinRegBench/data/finreg_3000_detector_eval.jsonl \
  --output FinRegBench/data/finreg_3000_artifact_audit.json
```

The audit reports surface patterns such as:

- exact copied evidence
- candidate/evidence containment
- inserted scope markers such as `only`
- invented-detail markers such as `email`, `xml`, or `watermark`
- number mismatches

High scores on artifact-heavy slices should be treated as calibration progress,
not proof that the detector generalizes.

To split the dataset into `artifact_easy` and `hard_candidate` rows:

```bash
python scripts/split_finregbench_by_artifacts.py \
  --dataset FinRegBench/data/finreg_3000_detector_eval.jsonl
```

Default outputs:

- `FinRegBench/data/finreg_3000_detector_eval_artifact_annotated.jsonl`
- `FinRegBench/data/finreg_3000_artifact_easy.jsonl`
- `FinRegBench/data/finreg_3000_hard_candidate.jsonl`
- `FinRegBench/data/finreg_3000_artifact_split_summary.json`

## Phase 2 Pack

Build the benchmark pack:

```bash
python scripts/build_finregbench_phase2_pack.py \
  --dataset FinRegBench/data/finreg_3000_detector_eval_artifact_annotated.jsonl \
  --output-dir FinRegBench/data/phase2_pack
```

Default outputs:

- `FinRegBench/data/phase2_pack/smoke_300.jsonl`
- `FinRegBench/data/phase2_pack/contradiction_stress_520.jsonl`
- `FinRegBench/data/phase2_pack/review_180.jsonl`
- `FinRegBench/data/phase2_pack/gold_seed_template.jsonl`
- `FinRegBench/data/phase2_pack/phase2_pack_summary.json`

Pack usage:

- `smoke_300`: fast regression and adapter smoke testing
- `contradiction_stress_520`: hard contradiction-only stress test
- `review_180`: manual review queue
- `gold_seed_template`: review decisions to fill before creating a gold seed

## Phase 2 Detector Integration

Run a detector against a pack and evaluate it:

```bash
python scripts/run_finregbench_phase2_detector_eval.py \
  --pack smoke_300 \
  --detector-command "python path/to/detector.py --input {input} --output {output}"
```

The command template can use:

- `{input}` or `{inputs}`: exported detector input JSONL
- `{dataset}`: original pack dataset JSONL
- `{output}` or `{predictions}`: raw detector prediction JSONL path to write

If predictions already exist:

```bash
python scripts/run_finregbench_phase2_detector_eval.py \
  --pack smoke_300 \
  --raw-predictions path/to/raw_predictions.jsonl
```

Default run outputs:

- `FinRegBench/data/phase2_runs/<pack>/detector_inputs.jsonl`
- `FinRegBench/data/phase2_runs/<pack>/raw_predictions.jsonl`
- `FinRegBench/data/phase2_runs/<pack>/predictions_normalized.jsonl`
- `FinRegBench/data/phase2_runs/<pack>/prediction_normalization_report.json`
- `FinRegBench/data/phase2_runs/<pack>/eval_report.json`
- `FinRegBench/data/phase2_runs/<pack>/eval_slices.csv`

The integration script normalizes common detector fields into canonical
`support_status` labels before evaluation.  Accepted label fields include:

- `support_status`
- `predicted_support_status`
- `prediction`
- `predicted_label`
- `label`
- `nli_label`
- `verdict`
- `decision`
- `class`

Accepted score fields include:

- `support_status_scores`
- `scores`
- `label_scores`
- `probabilities`
- `probs`

Canonical labels are:

- `supported`
- `unsupported`
- `contradicted`

Aliases such as `entailment`, `neutral`, `contradiction`, `grounded`,
`ungrounded`, `missing_evidence`, and `conflicting_answer` are normalized.
Any unmapped row is rejected and written to
`prediction_normalization_report.json`, so label mismatches cannot silently
pollute the scores.

## Phase 2.2 Training Seed

Mine detector errors:

```bash
python scripts/mine_finregbench_phase2_errors.py \
  --dataset FinRegBench/data/phase2_pack/smoke_300.jsonl \
  --predictions FinRegBench/data/phase2_runs/smoke_300_export_mixed_v2/predictions_normalized.jsonl \
  --run-name smoke_300 \
  --dataset FinRegBench/data/phase2_pack/contradiction_stress_520.jsonl \
  --predictions FinRegBench/data/phase2_runs/contradiction_stress_export_mixed_v2/predictions_normalized.jsonl \
  --run-name contradiction_stress_520 \
  --output-dir FinRegBench/data/phase2_error_mining
```

Export review packet:

```bash
python scripts/export_finregbench_error_review_packet.py
```

Prefill review decisions from expected labels:

```bash
python scripts/prefill_finregbench_error_review.py \
  --input FinRegBench/data/phase2_error_mining/error_review_packet.csv \
  --output FinRegBench/data/phase2_error_mining/error_review_packet_prefilled.csv
```

Build prefilled training seed:

```bash
python scripts/build_finregbench_reviewed_training_seed.py \
  --reviewed FinRegBench/data/phase2_error_mining/error_review_packet_prefilled.csv \
  --output FinRegBench/data/phase2_error_mining/targeted_training_seed_prefilled.jsonl \
  --summary FinRegBench/data/phase2_error_mining/prefilled_training_seed_summary.json \
  --accept-prefill
```

Build Phase 2.2 train/val/test data:

```bash
python scripts/build_finregbench_phase2_2_training_dataset.py \
  --seed-data FinRegBench/data/phase2_error_mining/targeted_training_seed_prefilled.jsonl \
  --base-data-dir data/training/nli_dataset_ambigqa_mini_targeted_multipleqa \
  --output-dir data/training/nli_dataset_finregbench_phase2_2 \
  --seed-placement stratified
```

The current generated Phase 2.2 dataset is seed-only because the configured
base training directory was not present locally:

- `data/training/nli_dataset_finregbench_phase2_2/train.jsonl`
- `data/training/nli_dataset_finregbench_phase2_2/val.jsonl`
- `data/training/nli_dataset_finregbench_phase2_2/test.jsonl`
- `data/training/nli_dataset_finregbench_phase2_2/dataset_stats.json`

Training config:

- `config/adamw_lora_finregbench_phase2_2_prefill.yaml`

The report includes:

- three-way metrics for `supported / unsupported / contradicted`
- binary metrics for `supported / not_supported`
- negative-only metrics for `unsupported / contradicted`
- confusion matrices
- per-label precision, recall, and F1
- slice reports for source, jurisdiction, difficulty, generation method,
  ambiguity status, and original NLI label
- sample errors for review

## Benchmark Status

The source data is marked `auto_generated_needs_human_review`.  Use it first as
an internal detector calibration and regression set.  Before treating it as a
gold benchmark, manually review a stratified sample across:

- label
- source document
- difficulty
- generation method
- ambiguity type
