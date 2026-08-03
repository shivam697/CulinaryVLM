#!/bin/bash
# ═══════════════════════════════════════════════════════
# CulinaryVLM — Setup GPU Environment
# Installs missing packages into biryani_hv conda env
# Run on: cluster head node
# ═══════════════════════════════════════════════════════

set -e

CONDA_BASE="/storage/home/bagler/anaconda3"
ENV_NAME="biryani_hv"
PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

echo "═══════════════════════════════════════════════════"
echo "  CulinaryVLM — Environment Setup"
echo "═══════════════════════════════════════════════════"

# Verify env exists
if [ ! -f "$PYTHON" ]; then
    echo "ERROR: Conda env '${ENV_NAME}' not found at ${CONDA_BASE}/envs/${ENV_NAME}"
    exit 1
fi

echo "Python: $($PYTHON --version)"
echo "Pip: $($PIP --version)"
echo ""

# ── Already installed: torch, torchvision, torchaudio, transformers,
#    numpy, pillow, scipy, PyYAML, sentencepiece

# ── ASR ──
echo "Installing ASR dependencies..."
$PIP install --no-cache-dir "whisperx>=3.1.1" "faster-whisper>=0.10.0" 2>&1 | tail -1

# ── NLP / Embeddings ──
echo "Installing NLP & embedding packages..."
$PIP install --no-cache-dir "sentence-transformers>=2.2.2" "scikit-learn>=1.3.0" 2>&1 | tail -1

# ── Video Download ──
echo "Installing video download tools..."
$PIP install --no-cache-dir "yt-dlp>=2023.11.0" 2>&1 | tail -1

# ── Image / Video Processing ──
echo "Installing image/video processing..."
$PIP install --no-cache-dir "opencv-python-headless>=4.8.0" "imagehash>=4.3.1" 2>&1 | tail -1

# ── API Clients ──
echo "Installing API clients..."
$PIP install --no-cache-dir "groq>=0.4.0" "google-generativeai>=0.3.0" "huggingface-hub>=0.19.0" 2>&1 | tail -1

# ── Vector Store ──
echo "Installing FAISS..."
$PIP install --no-cache-dir "faiss-cpu>=1.7.4" 2>&1 | tail -1

# ── Environment & Data ──
echo "Installing data processing..."
$PIP install --no-cache-dir "python-dotenv>=1.0.0" "openpyxl>=3.1.2" "pandas>=2.1.0" 2>&1 | tail -1

# ── GPU-specific (transformers extras) ──
echo "Installing GPU acceleration..."
$PIP install --no-cache-dir "accelerate>=0.25.0" "bitsandbytes>=0.41.0" 2>&1 | tail -1

# ── HTTP ──
echo "Installing HTTP tools..."
$PIP install --no-cache-dir "httpx>=0.25.0" "requests>=2.31.0" 2>&1 | tail -1

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Verifying key packages..."
echo "═══════════════════════════════════════════════════"

$PYTHON -c "
packages = [
    'torch', 'whisperx', 'sentence_transformers', 'sklearn',
    'cv2', 'groq', 'google.generativeai', 'faiss', 'yt_dlp',
    'dotenv', 'openpyxl', 'pandas', 'accelerate', 'bitsandbytes',
]
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  ✓ {pkg}')
    except ImportError:
        print(f'  ✗ {pkg} — MISSING')
"

echo ""
echo "✓ Environment setup complete!"
echo "  Next: python pipeline/01_download.py --resume"
