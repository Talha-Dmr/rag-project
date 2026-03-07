#!/usr/bin/env bash
set -euo pipefail

# Run 50Q detector ablation (balanced vs focal) on high-stakes domains.
# - Reuses existing balanced outputs from *_logit_mi_50_seed*.json
# - Runs missing focal outputs
# - Generates detector ablation summary when all files exist

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS_CSV="${1:-7,11,19}"
LIMIT="${2:-50}"
PYTHON_BIN="${PYTHON_BIN:-venv312/bin/python}"

if [[ "$LIMIT" != "50" ]]; then
  echo "This script is intended for 50Q ablation. Received limit=${LIMIT}."
  exit 1
fi

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
DOMAINS=("health" "finreg" "disaster")

for d in "${DOMAINS[@]}"; do
  for s in "${SEEDS[@]}"; do
    src="evaluation_results/auto_eval/${d}_logit_mi_50_seed${s}.json"
    dst="evaluation_results/auto_eval/${d}_balanced_50_seed${s}.json"
    if [[ -f "$src" && ! -f "$dst" ]]; then
      cp -f "$src" "$dst"
      echo "copied balanced: ${dst}"
    fi
  done
done

for d in "${DOMAINS[@]}"; do
  env PYTHONPATH=. HF_HOME=./models/llm TRANSFORMERS_CACHE=./models/llm \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}" \
    HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}" \
    TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
    LOG_LEVEL="${LOG_LEVEL:-WARNING}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PYTHON_BIN" -u scripts/eval_grounding_proxy_multi_seed.py \
      --config "gating_${d}_ebcar_logit_mi_sc009_focaldet" \
      --questions "data/domain_${d}/questions_${d}_conflict_50.jsonl" \
      --seeds "$SEEDS_CSV" \
      --limit 50 \
      --output-pattern "evaluation_results/auto_eval/${d}_focal_50_seed{seed}.json" \
      --skip-existing
done

need_files=0
for d in "${DOMAINS[@]}"; do
  for s in "${SEEDS[@]}"; do
    [[ -f "evaluation_results/auto_eval/${d}_balanced_50_seed${s}.json" ]] || need_files=$((need_files + 1))
    [[ -f "evaluation_results/auto_eval/${d}_focal_50_seed${s}.json" ]] || need_files=$((need_files + 1))
  done
done

if [[ "$need_files" -ne 0 ]]; then
  echo "Ablation run incomplete: ${need_files} files still missing."
  exit 1
fi

env PYTHONPATH=. "$PYTHON_BIN" scripts/summarize_detector_ablation.py \
  --set-size 50 \
  --json-out evaluation_results/auto_eval/detector_ablation_summary_50.json \
  --markdown-out docs/detector_ablation_report.md

echo "Detector ablation 50Q complete."
