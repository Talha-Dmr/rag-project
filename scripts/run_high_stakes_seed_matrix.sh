#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LIMIT="${1:-50}"
SET_SIZE="${2:-50}"       # 20 | 50
SEEDS_CSV="${3:-7,11,19}" # comma-separated
DOMAIN="${4:-all}"        # health | finreg | disaster | all
MAX_PARALLEL="${5:-${MAX_PARALLEL:-1}}"

AUTO_INDEX="${AUTO_INDEX:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PYTHON_BIN="${PYTHON_BIN:-venv312/bin/python}"
FORCE_PARALLEL="${FORCE_PARALLEL:-0}"
REUSE_PIPELINE="${REUSE_PIPELINE:-1}"

if [[ "$DOMAIN" != "health" && "$DOMAIN" != "finreg" && "$DOMAIN" != "disaster" && "$DOMAIN" != "all" ]]; then
  echo "Invalid domain: $DOMAIN"
  echo "Usage: $0 [limit] [20|50] [seeds_csv] [health|finreg|disaster|all]"
  exit 1
fi

if [[ "$SET_SIZE" != "20" && "$SET_SIZE" != "50" ]]; then
  echo "Invalid set size: $SET_SIZE"
  echo "Usage: $0 [limit] [20|50] [seeds_csv] [health|finreg|disaster|all] [max_parallel]"
  exit 1
fi

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "Invalid max_parallel: $MAX_PARALLEL"
  echo "Usage: $0 [limit] [20|50] [seeds_csv] [health|finreg|disaster|all] [max_parallel]"
  exit 1
fi

if [[ "$MAX_PARALLEL" -gt 1 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  TOTAL_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')"
  if [[ "$FORCE_PARALLEL" != "1" ]] && [[ "$TOTAL_MEM_MB" =~ ^[0-9]+$ ]] && [[ "$TOTAL_MEM_MB" -lt 14000 ]]; then
    echo "Detected GPU memory ${TOTAL_MEM_MB}MB; forcing max_parallel=1 to avoid OOM."
    echo "Override with FORCE_PARALLEL=1 if you want to force parallel runs."
    MAX_PARALLEL=1
  fi
fi

domain_corpus() {
  local domain="$1"
  echo "data/corpora/${domain}_corpus.jsonl"
}

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

run_eval_once() {
  local domain="$1"
  local seed="$2"
  ./scripts/run_high_stakes_seed_eval.sh "$domain" "$LIMIT" "$seed" "$SET_SIZE"
}

output_suffix() {
  if [[ "$LIMIT" == "$SET_SIZE" ]]; then
    echo ""
  else
    echo "_limit${LIMIT}"
  fi
}

output_path() {
  local domain="$1"
  local seed="$2"
  local suffix
  suffix="$(output_suffix)"
  echo "evaluation_results/auto_eval/${domain}_logit_mi_${SET_SIZE}_seed${seed}${suffix}.json"
}

run_eval_multi_seed() {
  local domain="$1"
  local seeds_csv="$2"
  local questions
  questions="$(question_file "$domain")"
  local suffix=""
  if [[ "$LIMIT" != "$SET_SIZE" ]]; then
    suffix="_limit${LIMIT}"
  fi
  local output_pattern="evaluation_results/auto_eval/${domain}_logit_mi_${SET_SIZE}_seed{seed}${suffix}.json"

  env PYTHONPATH=. HF_HOME=./models/llm TRANSFORMERS_CACHE=./models/llm \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    LOG_LEVEL="${LOG_LEVEL:-WARNING}" \
    TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}" \
    HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}" \
    HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PYTHON_BIN" -u scripts/eval_grounding_proxy_multi_seed.py \
      --config "gating_${domain}_ebcar_logit_mi_sc009" \
      --questions "$questions" \
      --seeds "$seeds_csv" \
      --limit "$LIMIT" \
      --output-pattern "$output_pattern" \
      $([[ "$SKIP_EXISTING" == "1" ]] && echo "--skip-existing")
}

index_domain() {
  local domain="$1"
  local config="gating_${domain}_ebcar_logit_mi_sc009"
  local corpus
  corpus="$(domain_corpus "$domain")"

  if [[ ! -f "$corpus" ]]; then
    echo "Missing corpus for ${domain}: ${corpus}"
    return 1
  fi

  env PYTHONPATH=. HF_HOME=./models/llm TRANSFORMERS_CACHE=./models/llm \
    HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    "$PYTHON_BIN" -u scripts/index_domain_corpus.py \
      --config "$config" \
      --corpus "$corpus"
}

ensure_and_run() {
  local domain="$1"
  local seed="$2"
  local suffix=""
  if [[ "$LIMIT" != "$SET_SIZE" ]]; then
    suffix="_limit${LIMIT}"
  fi
  local log_file="/tmp/${domain}_logit_mi_${SET_SIZE}_seed${seed}${suffix}.log"

  echo ""
  echo "=== ${domain} seed=${seed} limit=${LIMIT} set_size=${SET_SIZE} ==="
  if run_eval_once "$domain" "$seed"; then
    return 0
  fi

  if [[ "$AUTO_INDEX" != "1" ]]; then
    echo "Run failed and AUTO_INDEX=0. Skipping retry."
    return 1
  fi

  if [[ -f "$log_file" ]] && grep -q "Vector collection is empty (0 docs)" "$log_file"; then
    echo "Empty index detected for ${domain}; indexing corpus then retrying..."
    index_domain "$domain"
    run_eval_once "$domain" "$seed"
    return 0
  fi

  echo "Run failed for ${domain} seed=${seed}; see ${log_file}"
  return 1
}

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

DOMAINS=()
if [[ "$DOMAIN" == "all" ]]; then
  DOMAINS=("health" "finreg" "disaster")
else
  DOMAINS=("$DOMAIN")
fi

FAILURES=0
ACTIVE_JOBS=0
wait_for_one() {
  if wait -n; then
    :
  else
    FAILURES=$((FAILURES + 1))
  fi
  ACTIVE_JOBS=$((ACTIVE_JOBS - 1))
}

for d in "${DOMAINS[@]}"; do
  if [[ "$REUSE_PIPELINE" == "1" && "$MAX_PARALLEL" == "1" ]]; then
    echo ""
    echo "=== ${d} seeds=${SEEDS_CSV} limit=${LIMIT} set_size=${SET_SIZE} (reuse pipeline) ==="
    if run_eval_multi_seed "$d" "$SEEDS_CSV"; then
      continue
    fi
    echo "Multi-seed run failed for ${d}."
    FAILURES=$((FAILURES + 1))
    continue
  fi

  for s in "${SEEDS[@]}"; do
    if [[ "$SKIP_EXISTING" == "1" ]]; then
      out_file="$(output_path "$d" "$s")"
      if [[ -f "$out_file" ]]; then
        echo "Skipping existing result: ${out_file}"
        continue
      fi
    fi

    ensure_and_run "$d" "$s" &
    ACTIVE_JOBS=$((ACTIVE_JOBS + 1))

    while [[ "$ACTIVE_JOBS" -ge "$MAX_PARALLEL" ]]; do
      wait_for_one
    done
  done
done

while [[ "$ACTIVE_JOBS" -gt 0 ]]; do
  wait_for_one
done

if [[ "$FAILURES" -gt 0 ]]; then
  echo "Completed with failures: ${FAILURES}"
  exit 1
fi

echo ""
echo "All runs complete."
