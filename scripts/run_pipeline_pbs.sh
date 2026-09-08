#!/bin/bash
#PBS -N culinaryvlm_pipeline
#PBS -q Gpu7-80g
#PBS -l walltime=48:00:00
#PBS -l select=1:naccelerators=1
#PBS -o /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/pipeline_job.out
#PBS -e /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/pipeline_job.err
#PBS -m ae
# ═══════════════════════════════════════════════════════════════
# CulinaryVLM — Full Pipeline PBS Job Script
# ═══════════════════════════════════════════════════════════════
#
# Submit with:   qsub scripts/run_pipeline_pbs.sh
# Monitor with:  qstat -u $USER
# Cancel with:   qdel <jobid>
#
# Safe to resubmit if killed by walltime — see per-stage notes below.
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Constants ────────────────────────────────────────────────
PROJECT_ROOT="/storage/home/bagler/Shivam_Minde/CulinaryVLM"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OLLAMA_BIN="/storage/home/bagler/ollama-new/bin/ollama"

# Unique port per job, derived from the PBS job ID (falls back to
# the shell PID for interactive runs). This means a manually-started
# Ollama instance, or another job's instance, can never collide with
# this job's own Ollama server again.
JOB_NUM="${PBS_JOBID%%.*}"
OLLAMA_PORT=$(( 20000 + (${JOB_NUM:-$$} % 10000) ))
OLLAMA_URL="http://127.0.0.1:${OLLAMA_PORT}"
OLLAMA_MODEL="qwen2.5:32b-instruct"
TARGET_GPU_UUID="GPU-bfacee82-0a55-9ccf-245d-7ff819619515"
LOG_DIR="${PROJECT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

# Timestamp for this run's log files
RUN_TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="${LOG_DIR}/pipeline_${RUN_TS}.log"

# ─── Logging helper ──────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

log "═══════════════════════════════════════════════════"
log "CulinaryVLM Pipeline — Job started"
log "PBS Job ID: ${PBS_JOBID:-interactive}"
log "Hostname:   $(hostname)"
log "Ollama port: ${OLLAMA_PORT}"
log "═══════════════════════════════════════════════════"

# ─── 1. Pin GPU by UUID ──────────────────────────────────────
# The cluster has 3 GPUs; two A100s have MIG enabled and must
# be avoided.  Pin to the L40S by its UUID.
export CUDA_VISIBLE_DEVICES="${TARGET_GPU_UUID}"
log "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Verify the pinned GPU is accessible
if ! nvidia-smi -L 2>/dev/null | grep -q "${TARGET_GPU_UUID}"; then
    log "ERROR: Target GPU ${TARGET_GPU_UUID} not found on this node."
    nvidia-smi -L 2>&1 | tee -a "${MAIN_LOG}"
    exit 1
fi
log "GPU pinned: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "${TARGET_GPU_UUID}")"

# ─── 2. Conda Environment ────────────────────────────────────
source /storage/home/bagler/anaconda3/etc/profile.d/conda.sh
conda activate biryani_hv
log "Conda env: $(conda info --envs | grep '*' | awk '{print $1}')"
log "Python:    $(which python) — $(python --version 2>&1)"

cd "${PROJECT_ROOT}"

# ─── 3. GPU Telemetry Logger (background) ─────────────────────
GPU_TELEM_LOG="${LOG_DIR}/gpu_telemetry_${RUN_TS}.csv"
log "Starting GPU telemetry logger → ${GPU_TELEM_LOG}"
(
    echo "timestamp,gpu_util_pct,mem_used_mib,mem_total_mib,temp_c,power_w"
    while true; do
        nvidia-smi \
            --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
            --format=csv,noheader,nounits \
            -i "${TARGET_GPU_UUID}" 2>/dev/null
        sleep 60
    done
) >> "${GPU_TELEM_LOG}" &
TELEM_PID=$!
log "Telemetry PID: ${TELEM_PID}"

# Ensure cleanup on exit (kill telemetry + Ollama)
cleanup() {
    log "Cleanup: killing telemetry (PID ${TELEM_PID}) and Ollama"
    kill "${TELEM_PID}" 2>/dev/null || true
    kill "${OLLAMA_PID:-}" 2>/dev/null || true
    wait "${TELEM_PID}" 2>/dev/null || true
    wait "${OLLAMA_PID:-}" 2>/dev/null || true
    log "Cleanup done."
}
trap cleanup EXIT

# ─── 4. Ollama lifecycle (start + health-check/restart) ───────
OLLAMA_LOG="${LOG_DIR}/ollama_${RUN_TS}.log"

start_ollama() {
    log "Starting Ollama server on port ${OLLAMA_PORT} → ${OLLAMA_LOG}"
    export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
    "${OLLAMA_BIN}" serve >> "${OLLAMA_LOG}" 2>&1 &
    OLLAMA_PID=$!
    log "Ollama PID: ${OLLAMA_PID}"

    for i in $(seq 1 15); do
        if kill -0 "${OLLAMA_PID}" 2>/dev/null && curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
            log "Ollama reachable after ${i}×2s"
            return 0
        fi
        if [ "$i" -eq 15 ]; then
            log "ERROR: Ollama not reachable after 30s. Tail of log:"
            tail -20 "${OLLAMA_LOG}" | tee -a "${MAIN_LOG}"
            exit 1
        fi
        sleep 2
    done
}

# Call this immediately before any stage that talks to Ollama.
# Checks the job's OWN process is alive (not just "something answers
# on the port") and transparently restarts it if it's gone.
ensure_ollama() {
    if kill -0 "${OLLAMA_PID:-0}" 2>/dev/null && curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
        return 0
    fi

    log "WARNING: Ollama (PID ${OLLAMA_PID:-none}) is down or unresponsive — restarting"
    kill "${OLLAMA_PID:-}" 2>/dev/null || true
    wait "${OLLAMA_PID:-}" 2>/dev/null || true
    start_ollama
}

start_ollama

# Verify target model is available
if ! curl -sf "${OLLAMA_URL}/api/tags" | python -c "
import sys, json
tags = [m['name'] for m in json.load(sys.stdin).get('models', [])]
if not any('${OLLAMA_MODEL}' == t or '${OLLAMA_MODEL}' == t.split(':')[0] for t in tags):
    print(f'ERROR: Model ${OLLAMA_MODEL} not found. Available: {tags}', file=sys.stderr)
    sys.exit(1)
print(f'Model ${OLLAMA_MODEL} confirmed available')
"; then
    log "ERROR: Required model ${OLLAMA_MODEL} not available on Ollama server"
    exit 1
fi
log "Ollama ready with model: ${OLLAMA_MODEL}"

# ─── 5. Pipeline Execution ────────────────────────────────────
# Each stage is wrapped in a function for clarity.  set -e ensures
# a failed stage aborts the whole script.

run_stage() {
    local stage_name="$1"
    shift
    log "──────────────────────────────────────────────────"
    log "STAGE: ${stage_name}"
    log "CMD:   python $*"
    log "──────────────────────────────────────────────────"
    python "$@" 2>&1 | tee -a "${MAIN_LOG}"
    # Check exit code from the python process (pipefail is set)
    log "✓ ${stage_name} — completed successfully"
}

# ── Phase 0A: Categorize ──────────────────────────────────────
# CLI: --config, --dry-run.  No --resume.
# RESUMABILITY: Cheap pure-Python categorization (~seconds).
#               Overwrites datasets/categorized/video_metadata.json.
#               Safe to fully rerun.
run_stage "Phase 0A: Categorize" pipeline/00a_categorize.py

# ── Phase 0B: Canonical Recipes ───────────────────────────────
# CLI: --config, --dry-run, --offline, --categories, --min-videos,
#      --force, --host, --model.  No explicit --resume.
# RESUMABILITY: Has skip-if-exists logic per category file.
#               Without --force, existing recipe JSONs are skipped.
#               Safe to resubmit — only generates missing ones.
ensure_ollama
run_stage "Phase 0B: Canonical Recipes" pipeline/00b_canonical_recipes.py \
    --host "${OLLAMA_URL}" --model "${OLLAMA_MODEL}"

# ── Phase 0C: Metadata Tagging ────────────────────────────────
# CLI: --config, --target N, --dry-run.  No --resume.
# RESUMABILITY: Cheap pure-Python metadata tagging (~seconds).
#               Overwrites pipeline_selection.json.
#               Safe to fully rerun.
run_stage "Phase 0C: Metadata Tagging" pipeline/00c_metadata_tagging.py

# ── Stage 1: Download ─────────────────────────────────────────
# CLI: --config, --limit, --category, --resume, --dry-run, --output-dir.
# RESUMABILITY: --resume checks download log + existing files.
#               Safe to resubmit.
run_stage "Stage 1: Download" pipeline/01_download.py --resume

# ── Stage 2: ASR + Translation ────────────────────────────────
# CLI: --config, --limit, --category, --batch-size, --model-size,
#      --resume, --dry-run, --host, --model.
# RESUMABILITY: --resume skips videos with existing transcript JSON.
#               Safe to resubmit.
# NOTE: Stage 1 (Download) can run a long time, which is exactly
# the window where Ollama has previously died from sitting idle.
# ensure_ollama here is the fix for that.
ensure_ollama
run_stage "Stage 2: ASR + Translation" pipeline/02_asr.py \
    --resume --host "${OLLAMA_URL}" --model "${OLLAMA_MODEL}"

# ── Stage 4: Segmentation (InternVL2-8B) ──────────────────────
# CLI: --config, --limit, --category, --resume, --dry-run, --batch-size.
# RESUMABILITY: --resume skips videos with existing segment JSON.
#               Safe to resubmit.
run_stage "Stage 4: Segmentation" pipeline/04_segment.py --resume

# ── Stage 5: Clustering ──────────────────────────────────────
# CLI: --config, --threshold, --dry-run.  No --resume.
# RESUMABILITY: ⚠ NO RESUME LOGIC.  Cheap CPU-only clustering
#               (~seconds).  Overwrites output. Safe to fully rerun.
run_stage "Stage 5: Clustering" pipeline/05_cluster.py

# ── Stage 6: Verification ────────────────────────────────────
# CLI: --config, --resume, --dry-run, --host, --model.
# RESUMABILITY: --resume loads existing verified_segments.json,
#               skips segments already verified (non-error status).
#               Checkpoints every 50 segments.  Safe to resubmit.
ensure_ollama
run_stage "Stage 6: Verification" pipeline/06_verify.py \
    --resume --host "${OLLAMA_URL}" --model "${OLLAMA_MODEL}"

# ── Stage 7: Alignment ───────────────────────────────────────
# CLI: --config, --resume, --dry-run, --similarity-threshold.
# RESUMABILITY: --resume skips videos with existing alignment JSON.
#               Safe to resubmit.
run_stage "Stage 7: Alignment" pipeline/07_align.py --resume

# ── Stage 8: VidDiff Comparison ──────────────────────────────
# CLI: --config, --pairs, --dry-run, --host, --model.
# RESUMABILITY: ⚠ NO RESUME LOGIC.  Overwrites comparison_results.json.
#               Moderate cost (one Ollama call per category pair).
#               Safe to fully rerun but wastes compute on redo.
ensure_ollama
run_stage "Stage 8: VidDiff Comparison" pipeline/08_viddiff.py \
    --host "${OLLAMA_URL}" --model "${OLLAMA_MODEL}"

# ── Stage 9: QA Generation ───────────────────────────────────
# CLI: --config, --tier, --limit, --dry-run, --host, --model.
# RESUMABILITY: ⚠ NO RESUME LOGIC.  Writes checkpoint_easy.json
#               during the Easy tier as a runtime safety net, but
#               main() does NOT read it back on restart.
#               Overwrites train.json/test.json on every run.
#               Medium/Hard/Expert tiers are template-only (no LLM,
#               instant).  Easy tier is the expensive part (one
#               Ollama call per verified segment).
ensure_ollama
run_stage "Stage 9: QA Generation" pipeline/09_generate_qa.py \
    --host "${OLLAMA_URL}" --model "${OLLAMA_MODEL}"

# ── Stage 9B: Fill QA Answers ─────────────────────────────────
# CLI: --tier (required), --config, --limit, --host, --model.
# RESUMABILITY: Filters on needs_generation=true, sets it to false
#               after filling.  Saves in-place to train.json/test.json.
#               Checkpoints every 50 entries.  Naturally resumable
#               on resubmit — already-filled entries are skipped.
ensure_ollama
run_stage "Stage 9B: Fill QA (medium)" pipeline/09b_fill_qa_answers.py \
    --tier medium --host "${OLLAMA_URL}" --model "${OLLAMA_MODEL}"

ensure_ollama
run_stage "Stage 9B: Fill QA (hard)" pipeline/09b_fill_qa_answers.py \
    --tier hard --host "${OLLAMA_URL}" --model "${OLLAMA_MODEL}"

ensure_ollama
run_stage "Stage 9B: Fill QA (expert)" pipeline/09b_fill_qa_answers.py \
    --tier expert --host "${OLLAMA_URL}" --model "${OLLAMA_MODEL}"

# ── Stage 9C: QA Quality Filter ──────────────────────────────
# CLI: No argparse (no flags at all).
# RESUMABILITY: Cheap pure-Python filter (~seconds).
#               Writes train_filtered.json/test_filtered.json.
#               Safe to fully rerun.
run_stage "Stage 9C: QA Filter" pipeline/09c_filter_qa.py

# ── Stage 11: Build FAISS Index ──────────────────────────────
# CLI: --config, --model, --dry-run.  No --resume.
# RESUMABILITY: ⚠ NO RESUME LOGIC.  Overwrites FAISS index.
#               Moderate cost (sentence embedding).
#               Safe to fully rerun.
run_stage "Stage 11: Build FAISS" pipeline/11_build_faiss.py

# ─── 6. Final Summary ────────────────────────────────────────
log ""
log "═══════════════════════════════════════════════════"
log "  ALL STAGES COMPLETED SUCCESSFULLY"
log "  Main log:      ${MAIN_LOG}"
log "  GPU telemetry: ${GPU_TELEM_LOG}"
log "  Ollama log:    ${OLLAMA_LOG}"
log "═══════════════════════════════════════════════════"
log ""
log "NOTE: There is NO pipeline/10_finetune.py."
log "      Fine-tuning script is at training/finetune_qlora.py"
log "      (separate job — not included in this pipeline run)."