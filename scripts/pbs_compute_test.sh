#!/bin/bash
#PBS -N gpu_compute_test
#PBS -l nodes=1:ncpus=4
#PBS -l walltime=00:05:00
#PBS -q Gpu7-80g

OUTFILE="/storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/gpu_compute_test.txt"
CONDA_BASE="/storage/home/bagler/anaconda3"
PYTHON="${CONDA_BASE}/envs/biryani_hv/bin/python"

mkdir -p "$(dirname $OUTFILE)"
exec > "$OUTFILE" 2>&1

export CUDA_VISIBLE_DEVICES=0

echo "GPU Compute Test — $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

$PYTHON -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'Available: {torch.cuda.is_available()}')
print(f'Devices: {torch.cuda.device_count()}')
print(f'Device 0: {torch.cuda.get_device_name(0)}')

# Actual compute test
x = torch.randn(1000, 1000, device='cuda:0')
y = x @ x.T
print(f'Matmul test: PASSED (result shape: {y.shape})')

# Memory test
mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f'Memory: {mem_alloc:.1f} MB used / {mem_total:.1f} GB total')

print('GPU COMPUTE TEST: ALL PASSED')
"

echo "Done: $(date)"
