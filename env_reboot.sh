#!/bin/bash

set -euo pipefail

PROJECT_DIR=/lustre/uc3m/gts_c3_cluster_1-12/NFattention
VENV_DIR="$PROJECT_DIR/.venv"
TMP_DIR=/lustre/uc3m/gts_c3_cluster_1-12/tmp
CACHE_DIR=/lustre/uc3m/gts_c3_cluster_1-12/.cache/pip

echo "==> Loading Python module"
module purge
module load python/3.11.14

echo "==> Checking python3"
which python3
python3 --version

# Seguridad: aborta si python3 no es 3.10+
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PY_VER" < "3.10" ]]; then
    echo "ERROR: python3 is too old ($PY_VER). Aborting."
    exit 1
fi

echo "==> Removing old venv"
rm -rf "$VENV_DIR"

echo "==> Creating new venv with python3"
python3 -m venv "$VENV_DIR"

echo "==> Activating venv"
source "$VENV_DIR/bin/activate"
hash -r

echo "==> Verifying venv python"
which python
python --version
python -c "import sys; print(sys.executable)"

echo "==> Preparing dirs"
mkdir -p "$TMP_DIR"
mkdir -p "$CACHE_DIR"

echo "==> Upgrading pip"
TMPDIR="$TMP_DIR" PIP_CACHE_DIR="$CACHE_DIR" \
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing base dependencies"
TMPDIR="$TMP_DIR" PIP_CACHE_DIR="$CACHE_DIR" \
python -m pip install numpy scikit-learn PyYAML "datasets>=2.14.0,<4.0.0" tqdm normflows matplotlib pandas

echo "==> Installing PyTorch (CPU fallback)"
TMPDIR="$TMP_DIR" PIP_CACHE_DIR="$CACHE_DIR" \
python -m pip install torch torchvision || echo "⚠️ Torch install failed, will need manual install"

echo "==> Testing environment"
python - <<EOF
import sys
print("Python:", sys.version)
import numpy, sklearn, yaml, datasets, tqdm, matplotlib, pandas
print("Core deps OK")
try:
    import torch
    print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
except Exception as e:
    print("Torch not available:", e)
EOF

echo "==> DONE"