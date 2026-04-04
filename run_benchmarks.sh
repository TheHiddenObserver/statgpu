#!/bin/bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p results

TOTAL=7
STEP=0
OVERALL_START=$(date +%s)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a results/bench_master.log
}

run_experiment() {
    local name="$1"
    local logfile="$2"
    shift 2
    STEP=$((STEP + 1))
    log "--- EXPERIMENT ${STEP}/${TOTAL}: ${name} ---"
    log "Log file: results/${logfile}"
    local t0=$(date +%s)
    set +e
    CUDA_VISIBLE_DEVICES=0 timeout 3600 python dev/benchmarks/benchmark_all_methods_large_scale.py "$@" \
        > "results/${logfile}" 2>&1
    local exit_code=$?
    set -e
    local elapsed=$(( $(date +%s) - t0 ))
    if [ $exit_code -eq 124 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [TIMEOUT] Experiment exceeded 3600s limit (elapsed=${elapsed}s)" \
            >> "results/${logfile}"
        log "TIMEOUT ${name} after ${elapsed}s (limit 3600s) — see results/${logfile}"
    elif [ $exit_code -eq 0 ]; then
        log "DONE ${name} in ${elapsed}s"
    else
        log "FAILED ${name} (exit=${exit_code}) in ${elapsed}s — see results/${logfile}"
    fi
}

log "=========================================="
log "Starting benchmark suite (${TOTAL} experiments)"
log "Host: $(hostname)  PID: $$"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'N/A')"
log "Python: $(python --version 2>&1)"
log "=========================================="

run_experiment "large_scale" "bench_large_scale.log" \
  --devices cpu,cuda --repeats 5 --warmup-runs 2 \
  --n-reg 200000 --p-reg 64 --n-logit 150000 --p-logit 48 \
  --n-cox 100000 --p-cox 24 \
  --json-out results/bench_large_scale.json

run_experiment "high_dim" "bench_high_dim.log" \
  --devices cpu,cuda --repeats 5 --warmup-runs 2 \
  --n-reg 5000 --p-reg 2000 --n-logit 8000 --p-logit 1000 \
  --n-cox 5000 --p-cox 500 \
  --json-out results/bench_high_dim.json

run_experiment "largeN_largeP" "bench_largeN_largeP.log" \
  --devices cpu,cuda --repeats 5 --warmup-runs 2 \
  --n-reg 120000 --p-reg 512 --n-logit 100000 --p-logit 256 \
  --n-cox 80000 --p-cox 128 \
  --json-out results/bench_largeN_largeP.json

run_experiment "extreme_n" "bench_extreme_n.log" \
  --devices cpu,cuda --repeats 3 --warmup-runs 1 \
  --n-reg 500000 --p-reg 32 --n-logit 400000 --p-logit 32 \
  --n-cox 300000 --p-cox 16 \
  --json-out results/bench_extreme_n.json

run_experiment "with_inference" "bench_with_inference.log" \
  --devices cpu,cuda --repeats 5 --warmup-runs 2 \
  --compute-inference \
  --n-reg 60000 --p-reg 64 --n-logit 80000 --p-logit 48 \
  --n-cox 50000 --p-cox 24 \
  --json-out results/bench_with_inference.json

run_experiment "gpu_cleanup" "bench_gpu_cleanup.log" \
  --devices cpu,cuda --repeats 5 --warmup-runs 2 \
  --gpu-memory-cleanup \
  --n-reg 120000 --p-reg 128 --n-logit 100000 --p-logit 64 \
  --n-cox 80000 --p-cox 48 \
  --json-out results/bench_gpu_cleanup.json

run_experiment "baseline" "bench_baseline.log" \
  --devices cpu,cuda --repeats 3 --warmup-runs 1 \
  --n-reg 60000 --p-reg 64 --n-logit 80000 --p-logit 48 \
  --n-cox 50000 --p-cox 24 \
  --json-out results/bench_baseline.json

TOTAL_ELAPSED=$(( $(date +%s) - OVERALL_START ))
log "=========================================="
log "ALL ${TOTAL} EXPERIMENTS COMPLETE in ${TOTAL_ELAPSED}s"
log "=========================================="
