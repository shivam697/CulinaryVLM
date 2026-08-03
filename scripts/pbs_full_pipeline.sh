#!/bin/bash
#PBS -N culinary_gpu_pipeline
#PBS -l nodes=1:ncpus=8
#PBS -l walltime=72:00:00
#PBS -q Gpu7-80g
#PBS -o /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/gpu_pipeline_out.txt
#PBS -e /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/gpu_pipeline_err.txt

# ═══════════════════════════════════════════════════════
# CulinaryVLM — Full GPU Pipeline (Stages 2 + 4)
# Then CPU stages 5–9, 11 sequentially
# Queue: Gpu7-80g (L40S 48GB)
# ═══════════════════════════════════════════════════════

set -e  # Exit on first error

PROJECT_DIR="/storage/home/bagler/Shivam_Minde/CulinaryVLM"
CONDA_BASE="/storage/home/bagler/anaconda3"
ENV_NAME="biryani_hv"
LOG_DIR="${PROJECT_DIR}/logs"

cd "$PROJECT_DIR"

# Activate conda
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Target L40S GPU (index 0) — avoids MIG enumeration issues on A100s
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$LOG_DIR"

echo "═══════════════════════════════════════════════════════"
echo "  CulinaryVLM — Full Pipeline"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════"
nvidia-smi
echo ""

# ── Stage 2: ASR + Translation (GPU) ──────────────────
echo "━━━ STAGE 2: ASR + Translation ━━━"
python pipeline/02_asr.py \
    --config configs/config.yaml \
    --resume \
    --batch-size 16 \
    2>&1 | tee "${LOG_DIR}/stage2_asr.log"
echo "Stage 2 done: $(date)"

# ── Stage 4: Video Segmentation (GPU) ────────────────
echo ""
echo "━━━ STAGE 4: Video Segmentation ━━━"
python pipeline/04_segment.py \
    --config configs/config.yaml \
    --resume \
    --batch-size 4 \
    2>&1 | tee "${LOG_DIR}/stage4_segment.log"
echo "Stage 4 done: $(date)"

# ── Stage 5: Action Clustering (CPU) ─────────────────
echo ""
echo "━━━ STAGE 5: Action Clustering ━━━"
python pipeline/05_cluster.py \
    --config configs/config.yaml \
    --threshold 0.3 \
    2>&1 | tee "${LOG_DIR}/stage5_cluster.log"
echo "Stage 5 done: $(date)"

# ── Stage 6: Verification (CPU + Gemini API) ─────────
echo ""
echo "━━━ STAGE 6: Segment Verification ━━━"
python pipeline/06_verify.py \
    --config configs/config.yaml \
    --resume \
    --rpm 14 \
    2>&1 | tee "${LOG_DIR}/stage6_verify.log"
echo "Stage 6 done: $(date)"

# ── Stage 7: Multimodal Alignment (CPU) ──────────────
echo ""
echo "━━━ STAGE 7: Multimodal Alignment ━━━"
python pipeline/07_align.py \
    --config configs/config.yaml \
    --resume \
    2>&1 | tee "${LOG_DIR}/stage7_align.log"
echo "Stage 7 done: $(date)"

# ── Stage 8: VidDiff Comparison (CPU + Groq API) ────
echo ""
echo "━━━ STAGE 8: VidDiff Comparison ━━━"
python pipeline/08_viddiff.py \
    --config configs/config.yaml \
    --pairs 50 \
    2>&1 | tee "${LOG_DIR}/stage8_viddiff.log"
echo "Stage 8 done: $(date)"

# ── Stage 9: QA Generation (CPU + Groq API) ─────────
echo ""
echo "━━━ STAGE 9: QA Generation ━━━"
python pipeline/09_generate_qa.py \
    --config configs/config.yaml \
    --tier all \
    2>&1 | tee "${LOG_DIR}/stage9_qa.log"
echo "Stage 9 done: $(date)"

# ── Stage 11: FAISS Index (CPU) ──────────────────────
echo ""
echo "━━━ STAGE 11: Build FAISS Index ━━━"
python pipeline/11_build_faiss.py \
    --config configs/config.yaml \
    2>&1 | tee "${LOG_DIR}/stage11_faiss.log"
echo "Stage 11 done: $(date)"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✓ FULL PIPELINE COMPLETE"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════"
