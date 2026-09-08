#!/bin/bash
#PBS -N culinaryvlm_finetune
#PBS -q Gpu7-80g
#PBS -l walltime=24:00:00
#PBS -l select=1:naccelerators=1
#PBS -o /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/finetune_job.out
#PBS -e /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/finetune_job.err
#PBS -m ae
set -euo pipefail

PROJECT_ROOT="/storage/home/bagler/Shivam_Minde/CulinaryVLM"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TARGET_GPU_UUID="GPU-bfacee82-0a55-9ccf-245d-7ff819619515"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

RUN_TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/finetune_${RUN_TS}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

log "═══════════════════════════════════════════════════"
log "CulinaryVLM Fine-tuning — Job started"
log "PBS Job ID: ${PBS_JOBID:-interactive}"
log "Hostname:   $(hostname)"
log "═══════════════════════════════════════════════════"

export CUDA_VISIBLE_DEVICES="${TARGET_GPU_UUID}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if ! nvidia-smi -L 2>/dev/null | grep -q "${TARGET_GPU_UUID}"; then
    log "ERROR: Target GPU ${TARGET_GPU_UUID} not found on this node."
    nvidia-smi -L 2>&1 | tee -a "${MAIN_LOG}"
    exit 1
fi
log "GPU pinned: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "${TARGET_GPU_UUID}")"

source /storage/home/bagler/anaconda3/etc/profile.d/conda.sh
conda activate biryani_hv
log "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"
log "Python:    $(which python) — $(python --version 2>&1)"

cd "${PROJECT_ROOT}"

GPU_TELEM_LOG="${LOG_DIR}/gpu_telemetry_finetune_${RUN_TS}.csv"
log "Starting GPU telemetry logger → ${GPU_TELEM_LOG}"
(
    echo "timestamp,gpu_util_pct,mem_used_mib,mem_total_mib,temp_c,power_w"
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader,nounits 2>/dev/null
        sleep 60
    done
) >> "${GPU_TELEM_LOG}" &
TELEM_PID=$!
log "Telemetry PID: ${TELEM_PID}"

cleanup() {
    log "Cleanup: killing telemetry (PID ${TELEM_PID})"
    kill "${TELEM_PID}" 2>/dev/null || true
    wait "${TELEM_PID}" 2>/dev/null || true
    log "Cleanup done."
}
trap cleanup EXIT

if grep -q '"qa" / "train.json"' training/finetune_qlora.py; then
    log "ERROR: training/finetune_qlora.py still points at unfiltered files."
    exit 1
fi
log "Pre-flight OK: script reads filtered dataset files."

python -c "import unsloth" 2>&1 | tee -a "${MAIN_LOG}" || {
    log "ERROR: unsloth not importable in this env. Install with: pip install unsloth"
    exit 1
}

log "──────────────────────────────────────────────────"
log "STAGE: Stage 10 — QLoRA Fine-tuning"
log "CMD:   python training/finetune_qlora.py $*"
log "──────────────────────────────────────────────────"
python training/finetune_qlora.py "$@" 2>&1 | tee -a "${MAIN_LOG}"

log ""
log "═══════════════════════════════════════════════════"
log "  FINE-TUNING COMPLETE"
log "  Main log:      ${MAIN_LOG}"
log "  GPU telemetry: ${GPU_TELEM_LOG}"
log "═══════════════════════════════════════════════════"
