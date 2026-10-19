#!/usr/bin/env bash
# Inference-only install of the behavior_prompting policy server on a Jetson Orin Nano (JetPack 7.2.x, Ubuntu 24.04, Python 3.12; JetPack 6.x also works by overriding TORCH_INDEX).
# Run ON THE JETSON, from anywhere:  bash controlSW/jetson/setup_jetson.sh
# Does NOT use behavior_prompting/environment.yml: it pins x86 + CUDA 11.8 wheels and training/sim-only packages.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-$HOME/bp-venv}"
# Wheel index for CUDA torch. Default = upstream PyTorch cu132 build, reported working on JetPack 7.2 + Orin (sm_87) by NVIDIA forum users.
# Not yet verified on THIS Nano. For JetPack 6.x use: TORCH_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu132}"

sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential cmake git libopenblas-dev ffmpeg libgl1 libglib2.0-0

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip "setuptools<70" wheel

# 1) CUDA torch FIRST, from $TORCH_INDEX. A plain `pip install torch` gives a CPU-only aarch64 build.
#    Skipped if this venv already has a CUDA torch.
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA torch already present in $VENV, skipping install"
else
  pip install torch torchvision --index-url "$TORCH_INDEX"
fi
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "torch has no CUDA: wrong wheel/index for this JetPack, fix before continuing"
PY

# 2) Pure-Python / small deps, versions copied from behavior_prompting/environment.yml (plus its deps).
#    Installed from PyPI; none of these should replace the torch above (checked at the end).
pip install \
  numpy==1.26.4 scipy==1.15.3 pyyaml==6.0.3 click==8.3.2 tqdm==4.67.3 pandas==2.3.3 matplotlib==3.10.8 \
  transformers==4.57.1 diffusers==0.35.1 accelerate==1.10.1 huggingface_hub==0.35.3 safetensors==0.6.2 \
  hydra-core==1.2.0 dill==0.3.7 einops==0.6.1 timm==1.0.20 \
  zarr==2.16.0 numcodecs==0.13.1 imagecodecs==2023.9.18 lmdb==2.2.0 \
  opencv-python-headless imageio imageio-ffmpeg av pillow transforms3d==0.4.2 pyzmq numba \
  cloudpickle==2.1.0 easydict==1.9 threadpoolctl==3.2.0 colorama==0.4.6 atomics==1.0.3 quaternion || \
  echo "WARNING: some package failed to install; see error above (numpy-quaternion / numba may need a different version on aarch64)"

# 3) robotmq (RMQ transport). May have no aarch64 wheel; if so it must be built from source.
python -c "import robotmq" 2>/dev/null || pip install robotmq==0.1.14 || { echo "ERROR: robotmq 0.1.14 has no aarch64 wheel; see README step 4"; }

# 4) The policy package itself (no deps: we installed what we need above)
pip install --no-deps -e "$REPO/behavior_prompting"

# 5) Smoke test: does the inference server import, and is torch still the CUDA one?
python - <<'PY'
import torch, robotmq
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
assert torch.cuda.is_available(), "a dependency replaced the CUDA torch with a CPU one; reinstall torch from the Jetson index"
import behavior_prompting.deployment.detached_inference  # noqa: F401
print("OK: detached_inference imports")
PY
echo "Done. Activate with: source $VENV/bin/activate"
