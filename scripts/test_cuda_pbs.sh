#!/bin/bash
#PBS -N culinary_cuda_test
#PBS -q Gpu7-80g
#PBS -l walltime=00:05:00
#PBS -l select=1:naccelerators=1
#PBS -o /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/cuda_test.out
#PBS -e /storage/home/bagler/Shivam_Minde/CulinaryVLM/logs/cuda_test.err

set -u

ENV="/storage/home/bagler/anaconda3/envs/culinary_finetune"

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib"

for lib in "$ENV"/lib/python3.10/site-packages/nvidia/*/lib; do
    case "$lib" in
        *nvidia/cu13/lib) continue ;;
        *nvidia/nccl-cu13/lib) continue ;;
        *nvidia/cudnn-cu13/lib) continue ;;
        *nvidia/nvshmem-cu13/lib) continue ;;
        *nvidia/cusparselt-cu13/lib) continue ;;
        *-cu13*) continue ;;
    esac
    export LD_LIBRARY_PATH="$lib:$LD_LIBRARY_PATH"
done

echo "===== PBS ====="
echo "JOBID=${PBS_JOBID:-UNSET}"
echo "HOST=$(hostname)"
echo

echo "===== CUDA ENV ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-UNSET}"
echo

echo "===== NVIDIA ====="
nvidia-smi -L
echo
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv
echo

echo "===== PYTORCH ====="
python - <<'PY'
import os
print("Python:", os.sys.executable)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA_DEVICE_ORDER:", os.environ.get("CUDA_DEVICE_ORDER"))

import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

try:
    print("current_device:", torch.cuda.current_device())
except Exception as e:
    print("current_device ERROR:", repr(e))

for i in range(torch.cuda.device_count()):
    try:
        print("device", i, ":", torch.cuda.get_device_name(i))
    except Exception as e:
        print("device", i, "ERROR:", repr(e))
PY

echo
echo "===== END ====="
