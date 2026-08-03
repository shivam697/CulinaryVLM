#!/bin/bash
#PBS -N culinary_segment
#PBS -l nodes=1:ncpus=8
#PBS -l walltime=72:00:00
#PBS -q Gpu7-80g
#PBS -o /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/segment_out.txt
#PBS -e /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/segment_err.txt

# ═══════════════════════════════════════════════════════
# CulinaryVLM — PBS Script: Video Segmentation (InternVL2-8B)
# Queue: Gpu7-80g (L40S 48GB)
# Expected runtime: ~6-12 hours for 172 videos
# ═══════════════════════════════════════════════════════

PROJECT_DIR="/storage/home/bagler/Shivam_Minde/CulinaryVLM"
CONDA_BASE="/storage/home/bagler/anaconda3"
ENV_NAME="biryani_hv"

cd "$PROJECT_DIR"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Target L40S GPU (index 0)
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${PROJECT_DIR}/logs"

echo "=== Segmentation Job Started: $(date) ==="
nvidia-smi

python pipeline/04_segment.py \
    --config configs/config.yaml \
    --resume \
    --batch-size 4 \
    2>&1 | tee "${PROJECT_DIR}/logs/segment_run.log"

echo "=== Segmentation Job Completed: $(date) ==="
