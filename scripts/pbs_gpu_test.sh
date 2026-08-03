#!/bin/bash
#PBS -N gpu_test
#PBS -l nodes=1:ncpus=4
#PBS -l walltime=00:10:00
#PBS -q Gpu7-80g
#PBS -o /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/gpu_test_out.txt
#PBS -e /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/gpu_test_err.txt

# ═══════════════════════════════════════════════════════
# GPU Diagnostic Test Job
# Purpose: Verify GPU access on compute node
# ═══════════════════════════════════════════════════════

PROJECT_DIR="/storage/home/bagler/Shivam_Minde/CulinaryVLM"
CONDA_BASE="/storage/home/bagler/anaconda3"
ENV_NAME="biryani_hv"
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

mkdir -p "${PROJECT_DIR}/logs"

echo "═══════════════════════════════════════════════════"
echo "  GPU DIAGNOSTIC TEST"
echo "  Time: $(date)"
echo "═══════════════════════════════════════════════════"

echo ""
echo "1. HOSTNAME:"
hostname

echo ""
echo "2. CUDA_VISIBLE_DEVICES:"
echo "   Value: '${CUDA_VISIBLE_DEVICES:-NOT SET}'"

echo ""
echo "3. nvidia-smi:"
nvidia-smi 2>&1 || echo "   nvidia-smi NOT FOUND"

echo ""
echo "4. NVIDIA driver / CUDA version from nvidia-smi:"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader 2>/dev/null || echo "   Could not query GPU"

echo ""
echo "5. Check ldconfig for CUDA libraries:"
ldconfig -p 2>/dev/null | grep -i cuda | head -5 || echo "   No CUDA libs in ldconfig"

echo ""
echo "6. Check CUDA toolkit:"
nvcc --version 2>/dev/null || echo "   nvcc NOT FOUND"
ls /usr/local/cuda*/bin/nvcc 2>/dev/null || echo "   No /usr/local/cuda"

echo ""
echo "7. Environment PATH and LD_LIBRARY_PATH:"
echo "   PATH: $PATH"
echo "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-NOT SET}"

echo ""
echo "8. Conda environment activation test:"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "   Python: $(which python)"
echo "   Python version: $(python --version)"

echo ""
echo "9. PyTorch CUDA test:"
python -c "
import torch
print(f'   torch.__version__    = {torch.__version__}')
print(f'   torch.version.cuda   = {torch.version.cuda}')
print(f'   cuda.is_available()  = {torch.cuda.is_available()}')
print(f'   cuda.device_count()  = {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'   GPU {i}: {props.name}, {props.total_mem / 1024**3:.1f} GB')
    # Quick compute test
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'   Compute test: PASSED (matmul on GPU)')
else:
    print('   WARNING: CUDA not available!')
    print(f'   torch built with CUDA: {torch.backends.cuda.is_built()}')
" 2>&1

echo ""
echo "10. Internet connectivity test:"
python -c "
import urllib.request
try:
    urllib.request.urlopen('https://huggingface.co', timeout=5)
    print('   Internet: AVAILABLE')
except Exception as e:
    print(f'   Internet: NOT AVAILABLE ({e})')
" 2>&1

echo ""
echo "11. Disk space:"
df -h /storage/home/bagler/ 2>/dev/null | tail -1

echo ""
echo "═══════════════════════════════════════════════════"
echo "  GPU DIAGNOSTIC COMPLETE: $(date)"
echo "═══════════════════════════════════════════════════"
