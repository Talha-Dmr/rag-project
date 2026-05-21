# Scripts Guide

The scripts directory is research-heavy. This guide lists the entry points that
matter for the current FinReg RAG + detector/gating workflow.

## Corpus And Index

- `fetch_finreg_phase1_sources.py`: fetch official FinReg source documents listed in `config/finreg_phase1_sources.yaml`.
- `build_finreg_phase1_corpus.py`: build `data/processed/finreg/finreg_phase1_corpus.jsonl`.
- `index_domain_corpus.py`: build or rebuild a Chroma index from a corpus/config.
- `audit_finreg_retrieval.py`: retrieval-only audit over full-RAG benchmark questions.

## Full-RAG Evaluation

- `run_finreg_real_life_benchmark.py`: main controlled/full-RAG benchmark runner.
- `compare_detector_gate_sensitivity.py`: compare detector/gate settings.
- `sweep_gating_thresholds.py`: threshold sweeps for gate behavior.

## Detector Data And Evaluation

- `build_finreg_detector_hardmix_dataset.py`: build hard-mix detector dataset.
- `build_finreg_retrieved_detector_eval.py`: build retrieved-context detector eval sets.
- `evaluate_hallucination_model.py`: evaluate detector checkpoints.
- `run_finreg_detector_diagnostic.py`: detector diagnostic runs.
- `run_finregbench_detector_model.py`: FinRegBench-style detector model run.

## Historical / Experimental

- `eval_grounding_proxy*.py`: earlier proxy evaluation flow.
- `run_chunking_*`: chunking ablation experiments.
- `sgld`, `logit`, and `representation` scripts: Langevin/uncertainty experiments.
- AmbigQA/ASQA/FEVER preparation scripts are historical inputs for detector/data experiments.

When adding new scripts, prefer a narrow CLI with explicit input/output paths and
document the intended config in this file.
