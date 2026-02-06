#!/usr/bin/env bash
set -euo pipefail

# Run quick A/B comparison for logit-MI vs representation-MI gating.
# Usage:
#   scripts/run_rep_vs_logit_ablation.sh [limit] [seed]
# Example:
#   scripts/run_rep_vs_logit_ablation.sh 50 7

LIMIT="${1:-50}"
SEED="${2:-7}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="venv312/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  echo "Missing python env: $PY_BIN"
  exit 1
fi

export PYTHONPATH=.
export HF_HOME="./models/llm"
export TRANSFORMERS_CACHE="./models/llm"

run_eval() {
  local config="$1"
  local questions="$2"
  local output="$3"
  local log="$4"
  "$PY_BIN" -u scripts/eval_grounding_proxy.py \
    --config "$config" \
    --questions "$questions" \
    --limit "$LIMIT" \
    --seed "$SEED" \
    --output "$output" > "$log" 2>&1
}

mkdir -p evaluation_results/auto_eval

run_eval "gating_energy_ebcar_logit_mi_sc009" \
  "data/domain_energy/questions_energy_conflict_50.jsonl" \
  "evaluation_results/auto_eval/energy_logit_mi_${LIMIT}.json" \
  "/tmp/energy_logit_mi_${LIMIT}.log"

run_eval "gating_energy_ebcar_rep_mi_sc004" \
  "data/domain_energy/questions_energy_conflict_50.jsonl" \
  "evaluation_results/auto_eval/energy_rep_mi_${LIMIT}.json" \
  "/tmp/energy_rep_mi_${LIMIT}.log"

run_eval "gating_macro_ebcar_logit_mi_sc009" \
  "data/domain_macro/questions_macro_conflict_50.jsonl" \
  "evaluation_results/auto_eval/macro_logit_mi_${LIMIT}.json" \
  "/tmp/macro_logit_mi_${LIMIT}.log"

run_eval "gating_macro_ebcar_rep_mi_sc004" \
  "data/domain_macro/questions_macro_conflict_50.jsonl" \
  "evaluation_results/auto_eval/macro_rep_mi_${LIMIT}.json" \
  "/tmp/macro_rep_mi_${LIMIT}.log"

"$PY_BIN" - <<PY
import json
pairs = [
  ("energy_logit", "evaluation_results/auto_eval/energy_logit_mi_${LIMIT}.json"),
  ("energy_rep", "evaluation_results/auto_eval/energy_rep_mi_${LIMIT}.json"),
  ("macro_logit", "evaluation_results/auto_eval/macro_logit_mi_${LIMIT}.json"),
  ("macro_rep", "evaluation_results/auto_eval/macro_rep_mi_${LIMIT}.json"),
]
for name, path in pairs:
  d = json.load(open(path))
  print(
    name,
    f"abstain={d['abstain']}/{d['total']} ({d['abstain_rate']:.2f})",
    f"unc={d['stats_all'].get('uncertainty_mean', 0.0):.6f}",
    f"contr={d['stats_all'].get('contradiction_rate', 0.0):.3f}",
    f"actions={d.get('actions', {})}",
  )
PY

echo "Done. JSON outputs in evaluation_results/auto_eval/"
