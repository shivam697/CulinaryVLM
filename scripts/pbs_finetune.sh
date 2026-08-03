#!/bin/bash
#PBS -N culinary_finetune
#PBS -l nodes=1:ncpus=8
#PBS -l walltime=48:00:00
#PBS -q Gpu5-40g
#PBS -o /storage/iiitd/culinary_vlm/logs/finetune_out.txt
#PBS -e /storage/iiitd/culinary_vlm/logs/finetune_err.txt

cd $PBS_O_WORKDIR
conda activate biryani
mkdir -p /storage/iiitd/culinary_vlm/logs

echo "=== QLoRA Fine-tuning Started: $(date) ==="
nvidia-smi

python training/finetune_qlora.py \
    --epochs 3 \
    --lr 2e-4 \
    --batch-size 4 \
    --lora-r 16 \
    --lora-alpha 32 \
    2>&1 | tee /storage/iiitd/culinary_vlm/logs/finetune_run.log

echo "=== QLoRA Fine-tuning Completed: $(date) ==="
