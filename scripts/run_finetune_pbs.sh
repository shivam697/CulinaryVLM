#!/bin/bash
#PBS -N culinaryvlm_finetune
#PBS -q Gpu7-80g
#PBS -l walltime=24:00:00
#PBS -l select=1:naccelerators=1
#PBS -o /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/finetune_job.out
#PBS -e /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/finetune_job.err
#PBS -m ae
# ═══════════════════════════════════════════════════════════════
# CulinaryVLM — Stage 10 (QLoRA Fine-tuning) PBS Job Script
# ═══════════════════════════════════════════════════════════════
#
# Submit with:   qsub scripts/run_finetune_pbs.sh
# Monitor with:  qstat -u $USER
# Cancel with:   qdel <jobid>
#
# Smoke test:    qsub scripts/run_finetune_pbs.sh  (passes --smoke-test by default)
# Full training: qsub -v FINETUNE_ARGS="--epochs 3" scripts/run_finetune_pbs.sh
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Constants ────────────────────────────────────────────────
PROJECT_ROOT="/storage/home/bagler/Shivam_Minde/CulinaryVLM"
CONDA_ENV="/storage/home/bagler/anaconda3/envs/culinary_finetune"
LOG_DIR="${PROJECT_ROOT}/logs"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/storage/home/bagler/hf_cache
export HF_HUB_DISABLE_XET=1

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

# ─── 1. Conda Environment ────────────────────────────────────
# Prepend conda env to PATH (this makes 'python' resolve to the env's python)
export PATH="${CONDA_ENV}/bin:$PATH"

# CRITICAL: Set LD_LIBRARY_PATH to ONLY the cu12 libraries that match
# torch 2.4.0+cu124. Do NOT include cu13 libraries — they cause:
#   libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4
# because the cu13 bundle ships a cusparse that expects nvJitLink 13.x
# symbols, but torch loads nvJitLink 12.x instead.
#
# We explicitly list only the cu12 pip-bundled NVIDIA libraries:
export LD_LIBRARY_PATH="${CONDA_ENV}/lib"
for lib in "${CONDA_ENV}"/lib/python3.10/site-packages/nvidia/*/lib; do
    # Skip cu13 directory — it contains CUDA 13 libraries incompatible with torch+cu124
    case "$lib" in
        *nvidia/cu13/lib) continue ;;
        *nvidia/nccl-cu13/lib) continue ;;
        *nvidia/cudnn-cu13/lib) continue ;;
        *nvidia/nvshmem-cu13/lib) continue ;;
        *nvidia/cusparselt-cu13/lib) continue ;;
        *-cu13*) continue ;;
    esac
    export LD_LIBRARY_PATH="$lib:$LD_LIBRARY_PATH"
done

log "Python:      $(which python) — $(python --version 2>&1)"
log "LD_LIBRARY_PATH set (cu13 excluded)"

cd "${PROJECT_ROOT}"

# ─── 2. GPU Detection & Pinning ───────────────────────────────
# This node hands out unrestricted GPU visibility (PBS does not
# isolate devices per job on this queue). Two of the three GPUs
# are MIG-partitioned A100s; the MIG PARENT device is not itself
# a valid CUDA compute target. We must pin to a specific MIG
# instance UUID before torch ever touches CUDA, or device 0 will
# resolve to the invalid parent and crash with:
#   device >= 0 && device < num_gpus INTERNAL ASSERT FAILED
#
# Pinned to the 40GB MIG slice on GPU 0. If contention with another
# job is suspected, the alternate 40GB slice on GPU 2 is:
#   MIG-4889f421-8c3c-57fc-9adf-707c476a0d63
TARGET_GPU_UUID="MIG-a2f0dfd1-875c-58ee-8928-3963fa6bceef"
export CUDA_VISIBLE_DEVICES="${TARGET_GPU_UUID}"

if ! nvidia-smi -L 2>/dev/null; then
    log "ERROR: nvidia-smi not available. Not on a GPU node."
    exit 1
fi

if ! nvidia-smi -L 2>/dev/null | grep -q "${TARGET_GPU_UUID}"; then
    log "ERROR: Target GPU ${TARGET_GPU_UUID} not found on this node."
    nvidia-smi -L 2>&1 | tee -a "${MAIN_LOG}"
    exit 1
fi
log "GPU pinned: $(nvidia-smi -L | grep "${TARGET_GPU_UUID}")"

# ─── 3. GPU Telemetry Logger (background) ─────────────────────
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

# ─── 4. Pre-flight checks ────────────────────────────────────
# 4a. Verify filtered dataset files
if [ ! -f "datasets/qa/train_filtered.json" ]; then
    log "ERROR: datasets/qa/train_filtered.json not found. Run Stage 9C first."
    exit 1
fi
if [ ! -f "datasets/qa/test_filtered.json" ]; then
    log "ERROR: datasets/qa/test_filtered.json not found. Run Stage 9C first."
    exit 1
fi
log "Pre-flight OK: filtered dataset files exist."

# 4b. Verify script reads filtered files
if grep -q 'train\.json' training/finetune_qlora.py | grep -v 'filtered' 2>/dev/null; then
    log "WARNING: finetune_qlora.py may reference unfiltered dataset. Check manually."
fi

# 4c. Verify torch + CUDA
log "Verifying torch + CUDA..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
" 2>&1 | tee -a "${MAIN_LOG}" || {
    log "ERROR: torch CUDA verification failed. Check CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH and CUDA drivers."
    exit 1
}

# 4d. Verify unsloth import
python -c "from unsloth import FastVisionModel; print('FastVisionModel import OK')" 2>&1 | tee -a "${MAIN_LOG}" || {
    log "ERROR: unsloth/FastVisionModel not importable."
    exit 1
}

# 4e. Verify trl import
python -c "from trl import SFTConfig, SFTTrainer; print('TRL import OK')" 2>&1 | tee -a "${MAIN_LOG}" || {
    log "ERROR: trl not importable."
    exit 1
}

log "All pre-flight checks passed."

# ─── 5. Determine run mode ───────────────────────────────────
# Default to smoke test if no arguments provided
FINETUNE_ARGS="${FINETUNE_ARGS:-${*:---smoke-test}}"

log "──────────────────────────────────────────────────"
log "STAGE: Stage 10 — QLoRA Fine-tuning"
log "CMD:   python training/finetune_qlora.py ${FINETUNE_ARGS}"
log "──────────────────────────────────────────────────"

python training/finetune_qlora.py ${FINETUNE_ARGS} 2>&1 | tee -a "${MAIN_LOG}"

log ""
log "═══════════════════════════════════════════════════"
log "  FINE-TUNING COMPLETE"
log "  Main log:      ${MAIN_LOG}"
log "  GPU telemetry: ${GPU_TELEM_LOG}"
log "═══════════════════════════════════════════════════"
