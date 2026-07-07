#!/usr/bin/env bash
# One-time creation of the `hnet-asr` conda env on Babel. Validated 2026-07-07
# on babel-o9-32 (NVIDIA L40S, driver 575.51 => CUDA <= 12.9).
#
# Rules this script encodes:
#   * The env lives on /data (via ~/.condarc envs_dirs), never on /home.
#   * torch MUST come from the cu129 wheel index — default PyPI torch ships
#     CUDA-13.0 builds that the Babel driver cannot run.
#   * mamba-ssm / causal-conv1d are installed --no-deps so pip cannot swap
#     torch for an incompatible build during dependency resolution.
set -euo pipefail

source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"

# Heavy conda dirs on the data partition (idempotent; ignore "already exists")
conda config --prepend envs_dirs /data/user_data/anshulk/envs 2>/dev/null || true
conda config --prepend pkgs_dirs /data/user_data/anshulk/conda_pkgs 2>/dev/null || true

conda create -y -n hnet-asr python=3.10
conda activate hnet-asr

pip install "torch==2.12.1+cu129" "torchaudio==2.11.0+cu129" --index-url https://download.pytorch.org/whl/cu129
pip install packaging ninja wheel setuptools einops

export CUDA_HOME=/usr/local/cuda-12.9
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS=8
pip install --no-build-isolation --no-deps --no-cache-dir causal-conv1d
pip install --no-build-isolation --no-deps --no-cache-dir mamba-ssm

pip install -r "$(cd "$(dirname "$0")/../.." && pwd)/requirements.txt"

# Smoke test: real Mamba-2 forward on the GPU
python - <<'EOF'
import torch
from mamba_ssm import Mamba2
m = Mamba2(d_model=256).cuda()
x = torch.randn(2, 64, 256, device="cuda")
print("Mamba2 forward OK:", tuple(m(x).shape), "| torch", torch.__version__)
EOF
