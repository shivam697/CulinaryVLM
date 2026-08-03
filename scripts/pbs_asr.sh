#!/bin/bash
#PBS -N culinary_asr
#PBS -l nodes=1:ncpus=8
#PBS -l walltime=48:00:00
#PBS -q Gpu7-80g
#PBS -o /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/asr_out.txt
#PBS -e /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/asr_err.txt

# ═══════════════════════════════════════════════════════
# CulinaryVLM — PBS Script: WhisperX ASR
# Queue: Gpu7-80g (L40S 48GB full card)
# Expected runtime: ~2-4 hours for 172 videos
# ═══════════════════════════════════════════════════════

# Project paths
PROJECT_DIR="/storage/home/bagler/Shivam_Minde/CulinaryVLM"
CONDA_BASE="/storage/home/bagler/anaconda3"
ENV_NAME="biryani_hv"

cd "$PROJECT_DIR"

# Activate conda environment
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Target L40S GPU (index 0) — avoids MIG enumeration issues on A100s
export CUDA_VISIBLE_DEVICES=0

# Create log directory
mkdir -p "${PROJECT_DIR}/logs"

echo "=== ASR Job Started: $(date) ==="
echo "Project: ${PROJECT_DIR}"
echo "Python: $(which python)"
echo "GPU:"
nvidia-smi

# Run ASR pipeline with resume support
python pipeline/02_asr.py \
    --config configs/config.yaml \
    --resume \
    --batch-size 16 \
    2>&1 | tee "${PROJECT_DIR}/logs/asr_run.log"

echo "=== ASR Job Completed: $(date) ==="
