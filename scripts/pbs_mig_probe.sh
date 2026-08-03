#!/bin/bash
#PBS -N gpu_mig_probe
#PBS -l nodes=1:ncpus=4
#PBS -l walltime=00:10:00
#PBS -q Gpu7-80g

OUTFILE="/storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/gpu_mig_probe.txt"
CONDA_BASE="/storage/home/bagler/anaconda3"
PYTHON="${CONDA_BASE}/envs/biryani_hv/bin/python"

mkdir -p "$(dirname $OUTFILE)"
exec > "$OUTFILE" 2>&1

echo "═══════════════════════════════════════════════════"
echo "  MIG DEVICE PROBE — $(date)"
echo "═══════════════════════════════════════════════════"

echo ""
echo "1. All GPU UUIDs:"
nvidia-smi -L

echo ""
echo "2. MIG device UUIDs:"
nvidia-smi -L | grep -i MIG

echo ""
echo "3. GPU UUIDs (query format):"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv

echo ""
echo "4. Testing each GPU index with CUDA_VISIBLE_DEVICES:"

for GPU_ID in 0 1 2; do
    echo ""
    echo "--- Testing CUDA_VISIBLE_DEVICES=$GPU_ID ---"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -c "
import torch
try:
    print(f'  cuda.is_available() = {torch.cuda.is_available()}')
    print(f'  device_count()      = {torch.cuda.device_count()}')
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f'  device name         = {torch.cuda.get_device_name(0)}')
        props = torch.cuda.get_device_properties(0)
        print(f'  memory              = {props.total_mem/1024**3:.1f} GB')
        x = torch.randn(100,100,device='cuda:0')
        y = x @ x.T
        print(f'  compute test        = PASSED')
except Exception as e:
    print(f'  ERROR: {e}')
" 2>&1
done

echo ""
echo "5. Testing MIG UUIDs (if any):"
# Extract MIG UUIDs and test each
nvidia-smi -L | grep "MIG" | while read -r line; do
    UUID=$(echo "$line" | grep -oP 'MIG-[a-f0-9-]+')
    if [ -n "$UUID" ]; then
        echo ""
        echo "--- Testing CUDA_VISIBLE_DEVICES=$UUID ---"
        CUDA_VISIBLE_DEVICES=$UUID $PYTHON -c "
import torch
try:
    print(f'  cuda.is_available() = {torch.cuda.is_available()}')
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f'  device name         = {torch.cuda.get_device_name(0)}')
        props = torch.cuda.get_device_properties(0)
        print(f'  memory              = {props.total_mem/1024**3:.1f} GB')
        x = torch.randn(100,100,device='cuda:0')
        y = x @ x.T
        print(f'  compute test        = PASSED')
except Exception as e:
    print(f'  ERROR: {e}')
" 2>&1
    fi
done

echo ""
echo "6. Testing L40S specifically (GPU index 1):"
CUDA_VISIBLE_DEVICES=1 $PYTHON -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'  Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>&1

echo ""
echo "═══════════════════════════════════════════════════"
echo "  DONE: $(date)"
echo "═══════════════════════════════════════════════════"
