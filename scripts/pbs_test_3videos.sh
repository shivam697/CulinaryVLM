#!/bin/bash
#PBS -N culinary_test_3vid
#PBS -l nodes=1:ncpus=8
#PBS -l walltime=02:00:00
#PBS -q Gpu7-80g
#PBS -o /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/test3_out.txt
#PBS -e /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/test3_err.txt

# ═══════════════════════════════════════════════════════
# CulinaryVLM — 3-Video Test Job (Stages 2 + 4)
# Queue: Gpu7-80g (L40S 48GB)
# Expected runtime: ~15-30 min for 3 videos
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
echo "  CulinaryVLM — 3-Video Test Job"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "GPU:"
nvidia-smi
echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())')"
echo ""

# ── Stage 2: ASR (one video per category) ──────────────
echo "━━━ STAGE 2: ASR — Hyderabadi ━━━"
python pipeline/02_asr.py \
    --config configs/config.yaml --resume \
    --category Hyderabadi --limit 1 \
    --batch-size 4 \
    2>&1 | tee "${LOG_DIR}/test3_asr_hyderabadi.log"

echo ""
echo "━━━ STAGE 2: ASR — Kolkata ━━━"
python pipeline/02_asr.py \
    --config configs/config.yaml --resume \
    --category Kolkata --limit 1 \
    --batch-size 4 \
    2>&1 | tee "${LOG_DIR}/test3_asr_kolkata.log"

echo ""
echo "━━━ STAGE 2: ASR — Muradabadi ━━━"
python pipeline/02_asr.py \
    --config configs/config.yaml --resume \
    --category Muradabadi --limit 1 \
    --batch-size 4 \
    2>&1 | tee "${LOG_DIR}/test3_asr_muradabadi.log"

echo ""
echo "Stage 2 complete: $(date)"

# ── Stage 4: Segmentation (one video per category) ────
echo ""
echo "━━━ STAGE 4: Segmentation — Hyderabadi ━━━"
python pipeline/04_segment.py \
    --config configs/config.yaml --resume \
    --category Hyderabadi --limit 1 \
    --batch-size 2 \
    2>&1 | tee "${LOG_DIR}/test3_seg_hyderabadi.log"

echo ""
echo "━━━ STAGE 4: Segmentation — Kolkata ━━━"
python pipeline/04_segment.py \
    --config configs/config.yaml --resume \
    --category Kolkata --limit 1 \
    --batch-size 2 \
    2>&1 | tee "${LOG_DIR}/test3_seg_kolkata.log"

echo ""
echo "━━━ STAGE 4: Segmentation — Muradabadi ━━━"
python pipeline/04_segment.py \
    --config configs/config.yaml --resume \
    --category Muradabadi --limit 1 \
    --batch-size 2 \
    2>&1 | tee "${LOG_DIR}/test3_seg_muradabadi.log"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✓ 3-VIDEO TEST COMPLETE"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════════"

# Summarize outputs
echo ""
echo "── Output Summary ──"
echo "Transcripts:"
ls -la datasets/transcripts/*.json 2>/dev/null | tail -5 || echo "  (none)"
echo "Segments:"
ls -la datasets/segments/*.json 2>/dev/null | tail -5 || echo "  (none)"
