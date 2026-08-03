#!/bin/bash
#PBS -N gpu_test2
#PBS -l nodes=1:ncpus=4
#PBS -l walltime=00:10:00
#PBS -q Gpu7-80g

# Write directly to shared storage (bypass PBS output staging)
OUTFILE="/storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/gpu_test_direct.txt"
CONDA_BASE="/storage/home/bagler/anaconda3"
ENV_NAME="biryani_hv"
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

mkdir -p "$(dirname $OUTFILE)"

exec > "$OUTFILE" 2>&1

echo "═══════════════════════════════════════════════════"
echo "  GPU DIAGNOSTIC TEST (Direct Output)"
echo "  Time: $(date)"
echo "═══════════════════════════════════════════════════"

echo ""
echo "1. HOSTNAME: $(hostname)"

echo ""
echo "2. CUDA_VISIBLE_DEVICES: '${CUDA_VISIBLE_DEVICES:-NOT SET}'"

echo ""
echo "3. nvidia-smi:"
nvidia-smi 2>&1 || echo "   nvidia-smi NOT FOUND"

echo ""
echo "4. GPU details:"
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader 2>/dev/null || echo "   Could not query GPU"

echo ""
echo "5. CUDA libraries:"
ldconfig -p 2>/dev/null | grep -i cuda | head -5 || echo "   No CUDA libs"

echo ""
echo "6. nvcc:"
nvcc --version 2>/dev/null || echo "   nvcc not found"
ls /usr/local/cuda*/bin/nvcc 2>/dev/null || echo "   No /usr/local/cuda"

echo ""
echo "7. PATH: $PATH"
echo "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-NOT SET}"

echo ""
echo "8. Conda env:"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "   Python: $(which python)"
echo "   Version: $(python --version)"

echo ""
echo "9. PyTorch CUDA:"
python -c "
import torch
print(f'   torch version     = {torch.__version__}')
print(f'   torch.version.cuda= {torch.version.cuda}')
print(f'   cuda available    = {torch.cuda.is_available()}')
print(f'   device count      = {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'   GPU {i}: {p.name}, {p.total_mem/1024**3:.1f} GB')
    x = torch.randn(100,100,device='cuda')
    y = x @ x.T
    print(f'   Compute test: PASSED')
else:
    print(f'   cuda built: {torch.backends.cuda.is_built()}')
"

echo ""
echo "10. Internet:"
python -c "
import urllib.request
try:
    urllib.request.urlopen('https://huggingface.co', timeout=5)
    print('   Internet: AVAILABLE')
except Exception as e:
    print(f'   Internet: NOT AVAILABLE ({e})')
"

echo ""
echo "11. Disk: $(df -h /storage/home/bagler/ 2>/dev/null | tail -1)"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  DONE: $(date)"
echo "═══════════════════════════════════════════════════"
