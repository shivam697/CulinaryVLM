#!/bin/bash
set -e

PROJECT_DIR="/storage/home/bagler/Shivam_Minde/CulinaryVLM"
CONDA_BASE="/storage/home/bagler/anaconda3"
LOG_DIR="${PROJECT_DIR}/logs"
L40S_UUID="GPU-bfacee82-0a55-9ccf-245d-7ff819619515"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR"

source "${CONDA_BASE}/etc/profile.d/conda.sh"

CATEGORIES=("Kolkata" "Muradabadi")

# ══════════════════════════════════════════════════
# Stage 2: ASR — requires biryani_asr (transformers>=4.48.0)
# ══════════════════════════════════════════════════
conda activate biryani_asr
echo "[env: biryani_asr] transformers: $(pip show transformers | grep Version)"

for cat in "${CATEGORIES[@]}"; do
    echo ""
    echo "━━━ STAGE 2: ASR — $cat ━━━"
    python pipeline/02_asr.py --config configs/config.yaml --resume \
        --category "$cat" --limit 1 --batch-size 4 \
        2>&1 | tee "${LOG_DIR}/asr_${cat}.log"
done
conda deactivate

# ══════════════════════════════════════════════════
# Stages 4-7 — require biryani_hv (transformers==4.44.2, InternVL2)
# ══════════════════════════════════════════════════
conda activate biryani_hv
echo ""
echo "[env: biryani_hv] transformers: $(pip show transformers | grep Version)"

export CUDA_VISIBLE_DEVICES="$L40S_UUID"
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "anaconda3/lib/python3.9" | tr '\n' ':')

echo "Confirming GPU target..."
python -c "import torch; assert torch.cuda.get_device_name(0) == 'NVIDIA L40S', 'WRONG GPU: ' + torch.cuda.get_device_name(0); print('GPU OK: NVIDIA L40S')"

# ── Stage 4: Segmentation ──
for cat in "${CATEGORIES[@]}"; do
    echo ""
    echo "━━━ STAGE 4: Segmentation — $cat ━━━"
    python pipeline/04_segment.py --category "$cat" --limit 1 --batch-size 2 \
        2>&1 | tee "${LOG_DIR}/seg_${cat}.log"
done

echo ""
echo "=== Segment file sizes (sanity check — should NOT be ~120 bytes) ==="
ls -la datasets/segments/*.json

# ── Stage 5: Clustering ──
echo ""
echo "━━━ STAGE 5: Clustering ━━━"
python pipeline/05_cluster.py 2>&1 | tee "${LOG_DIR}/cluster.log"

# ── Stage 6: Verification (Groq) ──
echo ""
echo "━━━ STAGE 6: Verification ━━━"
python pipeline/06_verify.py --rpm 25 2>&1 | tee "${LOG_DIR}/verify.log"

# ── Stage 7: Alignment ──
echo ""
echo "━━━ STAGE 7: Alignment ━━━"
python pipeline/07_align.py 2>&1 | tee "${LOG_DIR}/align.log"

conda deactivate

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✓ FULL PIPELINE COMPLETE"
echo "═══════════════════════════════════════════════════════"
echo "Transcripts:"; ls -la datasets/transcripts/*.json
echo "Segments:";    ls -la datasets/segments/*.json
echo "Verified:";    ls -la datasets/verified/*.json
echo "Alignments:";  ls -la datasets/alignments/*.json
