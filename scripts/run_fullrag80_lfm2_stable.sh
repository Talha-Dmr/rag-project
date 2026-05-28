#!/usr/bin/env bash
set -u

RUN_NAME="${RUN_NAME:-fullrag80_lfm2_guarded_v3_cpu_detector_stable}"
CONFIG="${CONFIG:-gating_finreg_lfm2_26b_local_rtx2070_evidence_retry_cpu_detector}"
QUESTIONS="${QUESTIONS:-benchmarks/finreg/full_rag_questions.jsonl}"
OUT_DIR="${OUT_DIR:-reports/finreg_real_life_benchmark}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
PYTHON_BIN="${PYTHON_BIN:-venv312/bin/python}"
LOG_DIR="${LOG_DIR:-logs}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}_resume_loop.log"

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  echo "[$(date -Is)] attempt ${attempt}/${MAX_ATTEMPTS}: ${RUN_NAME}" | tee -a "$LOG_FILE"
  "$PYTHON_BIN" scripts/run_finreg_real_life_benchmark.py \
    --mode full-rag \
    --config "$CONFIG" \
    --questions "$QUESTIONS" \
    --run-name "$RUN_NAME" \
    --output-dir "$OUT_DIR" \
    --resume 2>&1 | tee -a "$LOG_FILE"
  status=${PIPESTATUS[0]}

  if [ "$status" -eq 0 ]; then
    echo "[$(date -Is)] completed: ${RUN_NAME}" | tee -a "$LOG_FILE"
    exit 0
  fi

  echo "[$(date -Is)] failed with status ${status}; resuming after cooldown" | tee -a "$LOG_FILE"
  sleep 5
  attempt=$((attempt + 1))
done

echo "[$(date -Is)] failed after ${MAX_ATTEMPTS} attempts: ${RUN_NAME}" | tee -a "$LOG_FILE"
exit 1
