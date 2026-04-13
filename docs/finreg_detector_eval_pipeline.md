# Finreg Detector Eval Pipeline

This document gives the run order for the finreg detector diagnostic workflow.
The scripts only prepare and generate artifacts. You will run the evaluations.

Current recommended manual-eval package:

- detector pair: `fever_local` + `targeted_v2`
- subset run: `evaluation_results/finreg_detector_manualeval_v3`
- annotation protocol: `docs/finreg_manual_eval_v3_protocol.md`

## 1. Diagnostic Run

This command compares detector variants under the same base config. Defaults:

- `fever_local` -> `gating_finreg_ebcar_logit_mi_sc009_localdet`
- `balanced` -> `gating_finreg_ebcar_logit_mi_sc009`
- `targeted_contraguard` -> `gating_finreg_ebcar_logit_mi_sc009_targetedcontraguarddet`

```powershell
python scripts\run_finreg_detector_diagnostic.py `
  --base-config gating_finreg_ebcar_logit_mi_sc009 `
  --questions data\domain_finreg\questions_finreg_conflict_50.jsonl `
  --seed 7 `
  --subset-size 60 `
  --top-k-contexts 5 `
  --output-dir evaluation_results\finreg_detector_diagnostic
```

Main outputs:

- `evaluation_results/finreg_detector_diagnostic/detector_comparison_report.md`
- `evaluation_results/finreg_detector_diagnostic/per_detector_summary.json`
- `evaluation_results/finreg_detector_diagnostic/<variant>/per_question.jsonl`
- `evaluation_results/finreg_detector_diagnostic/stratified_eval_subset.jsonl`
- `evaluation_results/finreg_detector_diagnostic/manual_annotation_sheet.csv`

## 2. Manual Annotation

Recommended current package:

```powershell
python scripts\run_finreg_detector_diagnostic.py `
  --base-config gating_finreg_ebcar_logit_mi_sc009 `
  --questions data\domain_finreg\questions_finreg_conflict_50.jsonl `
  --seed 7 `
  --subset-size 60 `
  --top-k-contexts 5 `
  --variant fever_local `
  --variant targeted_v2=gating_finreg_ebcar_logit_mi_sc009_targetedcontraguarddet_v2 `
  --subset-variant fever_local `
  --subset-variant targeted_v2 `
  --suspicious-conflict-threshold 0.08 `
  --suspicious-hallucination-topk-threshold 0.10 `
  --contradiction-signal-threshold 0.08 `
  --high-contradiction-gap-threshold 0.50 `
  --uncertainty-gap-threshold 0.08 `
  --high-entailment-threshold 0.55 `
  --low-contradiction-threshold 0.05 `
  --output-dir evaluation_results\finreg_detector_manualeval_v3
```

Use `docs/finreg_manual_eval_v3_protocol.md` while annotating.

Fill these fields in `manual_annotation_sheet.csv`:

- `label`
  - `supported`
  - `unsupported`
  - `contradicted`
  - `partial`
  - `ambiguous`
- `error_type`
  - `fabricated_fact`
  - `wrong_number_or_threshold`
  - `cross_document_conflict`
  - `outdated_regulation`
  - `misinterpretation`
  - `incomplete_reasoning`
- `notes`

The sheet also includes `retrieval_max_score` and `retrieval_mean_score` so the
post-analysis can separate retrieval-weak cases from detector mistakes.

## 3. Post-Annotation Analysis

```powershell
python scripts\analyze_finreg_manual_annotations.py `
  --annotations evaluation_results\finreg_detector_diagnostic\manual_annotation_sheet.csv `
  --output-dir evaluation_results\finreg_detector_diagnostic\manual_analysis
```

Generated outputs:

- `evaluation_results/finreg_detector_diagnostic/manual_analysis/manual_label_metrics.json`
- `evaluation_results/finreg_detector_diagnostic/manual_analysis/error_analysis_report.md`

## 4. Eval Set V2 Draft

```powershell
python scripts\build_finreg_eval_v2.py `
  --annotations evaluation_results\finreg_detector_diagnostic\manual_annotation_sheet.csv `
  --output evaluation_results\finreg_detector_diagnostic\finreg_eval_v2_80_100.jsonl
```

Notes:

- The script first reuses annotated examples.
- If a target bucket is under-filled, it emits `needs_authoring=true` placeholder rows.
- So the output is a draft scaffold for V2 authoring, not necessarily the final set.
