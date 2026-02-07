#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DOMAIN="${1:-all}"    # health | finreg | disaster | all
LIMIT="${2:-20}"
SEED="${3:-7}"
SET_SIZE="${4:-20}"   # 20 | 50

if [[ "$DOMAIN" != "health" && "$DOMAIN" != "finreg" && "$DOMAIN" != "disaster" && "$DOMAIN" != "all" ]]; then
  echo "Invalid domain: $DOMAIN"
  echo "Usage: $0 [health|finreg|disaster|all] [limit] [seed] [20|50]"
  exit 1
fi

if [[ "$SET_SIZE" != "20" && "$SET_SIZE" != "50" ]]; then
  echo "Invalid question set size: $SET_SIZE"
  echo "Usage: $0 [health|finreg|disaster|all] [limit] [seed] [20|50]"
  exit 1
fi

question_file() {
  local domain="$1"
  local path20="data/domain_${domain}/questions_${domain}_conflict.jsonl"
  local path50="data/domain_${domain}/questions_${domain}_conflict_50.jsonl"
  if [[ "$SET_SIZE" == "50" ]]; then
    if [[ -f "$path50" ]]; then
      echo "$path50"
      return 0
    fi
    echo "Missing question set: $path50" >&2
    return 1
  fi
  echo "$path20"
}

run_eval() {
  local config="$1"
  local domain="$2"
  local output="$3"
  local log_file="$4"
  local questions
  questions="$(question_file "$domain")"

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
    "health" \
    "evaluation_results/auto_eval/health_logit_mi_${LIMIT}.json" \
    "/tmp/health_logit_mi_${LIMIT}.log"
fi

if [[ "$DOMAIN" == "finreg" || "$DOMAIN" == "all" ]]; then
  run_eval \
    "gating_finreg_ebcar_logit_mi_sc009" \
    "finreg" \
    "evaluation_results/auto_eval/finreg_logit_mi_${LIMIT}.json" \
    "/tmp/finreg_logit_mi_${LIMIT}.log"
fi

if [[ "$DOMAIN" == "disaster" || "$DOMAIN" == "all" ]]; then
  run_eval \
    "gating_disaster_ebcar_logit_mi_sc009" \
    "disaster" \
    "evaluation_results/auto_eval/disaster_logit_mi_${LIMIT}.json" \
    "/tmp/disaster_logit_mi_${LIMIT}.log"
fi

echo "Done. Domain=${DOMAIN} limit=${LIMIT} seed=${SEED} set_size=${SET_SIZE}"
