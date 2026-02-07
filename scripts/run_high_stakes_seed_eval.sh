#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DOMAIN="${1:-all}"    # health | finreg | disaster | all
LIMIT="${2:-20}"
SEED="${3:-7}"

if [[ "$DOMAIN" != "health" && "$DOMAIN" != "finreg" && "$DOMAIN" != "disaster" && "$DOMAIN" != "all" ]]; then
  echo "Invalid domain: $DOMAIN"
  echo "Usage: $0 [health|finreg|disaster|all] [limit] [seed]"
  exit 1
fi

run_eval() {
  local config="$1"
  local questions="$2"
  local output="$3"
  local log_file="$4"

  env PYTHONPATH=. HF_HOME=./models/llm TRANSFORMERS_CACHE=./models/llm \
    venv312/bin/python -u scripts/eval_grounding_proxy.py \
      --config "$config" \
      --questions "$questions" \
      --limit "$LIMIT" \
      --seed "$SEED" \
      --output "$output" | tee "$log_file"
}

if [[ "$DOMAIN" == "health" || "$DOMAIN" == "all" ]]; then
  run_eval \
    "gating_health_ebcar_logit_mi_sc009" \
    "data/domain_health/questions_health_conflict.jsonl" \
    "evaluation_results/auto_eval/health_logit_mi_${LIMIT}.json" \
    "/tmp/health_logit_mi_${LIMIT}.log"
fi

if [[ "$DOMAIN" == "finreg" || "$DOMAIN" == "all" ]]; then
  run_eval \
    "gating_finreg_ebcar_logit_mi_sc009" \
    "data/domain_finreg/questions_finreg_conflict.jsonl" \
    "evaluation_results/auto_eval/finreg_logit_mi_${LIMIT}.json" \
    "/tmp/finreg_logit_mi_${LIMIT}.log"
fi

if [[ "$DOMAIN" == "disaster" || "$DOMAIN" == "all" ]]; then
  run_eval \
    "gating_disaster_ebcar_logit_mi_sc009" \
    "data/domain_disaster/questions_disaster_conflict.jsonl" \
    "evaluation_results/auto_eval/disaster_logit_mi_${LIMIT}.json" \
    "/tmp/disaster_logit_mi_${LIMIT}.log"
fi

echo "Done. Domain=${DOMAIN} limit=${LIMIT} seed=${SEED}"
