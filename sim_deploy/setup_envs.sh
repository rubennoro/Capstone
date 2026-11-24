#!/usr/bin/env bash
# Build the three sim_deploy environments (sim_deploy/DESIGN.md §7, §9; PLAN.md D5). Idempotent: an env that exists
# gets the same pinned packages again; delete its folder for a clean rebuild.
#   ~/envs/sim_policy_deploy        the copied policy_deploy, light (launcher, frames, manager, policy hold_still/replay)
#   ~/envs/sim_policy_deploy_torch  the copied policy_deploy + torch + behavior_prompting (policy kind diffusion)
#   ~/envs/sim_deploy               the sim: MuJoCo 3.3.0, franka_sim, deploy_contracts (E6 adds sim_deploy itself)
# The first two replicate ~/envs/policy_deploy and ~/envs/policy_deploy_torch (pip freeze of 2026-11-19, same Python
# 3.10.12) with the local packages installed editable from THIS checkout, never from ~/Capstone-final-deployment.
# Usage: sim_deploy/setup_envs.sh [light] [torch] [sim]      (no argument: all three)
set -euo pipefail
unset PYTHONPATH
export PYTHONNOUSERSITE=1                 # pip must not count ~/.local packages as installed
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PYTHON310:-/usr/bin/python3.10}
ENVS=$HOME/envs
CU118=https://download.pytorch.org/whl/cu118

case $REPO in
  *Capstone-final-deployment*) echo "setup_envs: refusing to install from $REPO" >&2; exit 1 ;;
esac

make_env() {   # make_env DIR
  if [[ -x $1/bin/python ]]; then
    echo "setup_envs: $1 exists; reinstalling its pinned packages"
  else
    "$PY" -m venv "$1"
  fi
}

LIGHT_PINS=(
  av==17.1.0 exceptiongroup==1.3.1 iniconfig==2.3.1 msgpack==1.1.0 numpy==2.2.6 opencv-python-headless==4.12.0.88
  packaging==26.3 pluggy==1.7.0 pytest==8.3.5 PyYAML==6.0.3 pyzmq==27.1.0 scipy==1.15.3 tomli==2.5.0
  typing_extensions==4.16.0
)

TORCH_PINS=(
  accelerate==1.10.1 antlr4-python3-runtime==4.9.3 asciitree==1.0.0 certifi==2026.10.23 charset-normalizer==3.5.2
  click==8.3.2 contourpy==1.3.2 cycler==0.12.1 diffusers==0.35.1 dill==0.3.7 einops==0.6.1 exceptiongroup==1.3.1
  fasteners==0.20 filelock==4.1.0 fonttools==4.65.0 fsspec==2026.9.0 hf-xet==1.8.0 huggingface-hub==0.35.3
  hydra-core==1.2.0 idna==3.20 imagecodecs==2023.9.18 ImageIO==2.38.0 importlib_metadata==9.0.1 iniconfig==2.3.1
  Jinja2==3.1.6 kiwisolver==1.5.1 MarkupSafe==3.0.3 matplotlib==3.10.8 mpmath==1.3.0 msgpack==1.1.0 networkx==3.4.2
  numcodecs==0.13.1 numpy==1.26.4
  nvidia-cublas-cu11==11.11.3.6 nvidia-cuda-cupti-cu11==11.8.87 nvidia-cuda-nvrtc-cu11==11.8.89
  nvidia-cuda-runtime-cu11==11.8.89 nvidia-cudnn-cu11==9.1.0.70 nvidia-cufft-cu11==10.9.0.58
  nvidia-curand-cu11==10.3.0.86 nvidia-cusolver-cu11==11.4.1.48 nvidia-cusparse-cu11==11.7.5.86
  nvidia-nccl-cu11==2.21.5 nvidia-nvtx-cu11==11.8.86
  omegaconf==2.4.0 opencv-python==4.11.0.86 packaging==26.3 pillow==12.3.0 pluggy==1.7.0 psutil==7.2.2
  pyparsing==3.3.3 pytest==8.3.5 python-dateutil==3.0.0 PyYAML==6.0.3 pyzmq==27.1.0 regex==2026.9.29
  requests==2.34.2 robotmq==0.1.14 safetensors==0.6.2 scipy==1.15.3 six==1.17.0 sympy==1.14.0 timm==1.0.20
  tokenizers==0.22.2 tomli==2.5.0 torch==2.7.1+cu118 torchvision==0.22.1+cu118 tqdm==4.70.1 transformers==4.57.1
  triton==3.3.1 typing_extensions==4.16.0 tzdata==2026.6 urllib3==2.8.0 zarr==2.16.0 zipp==4.1.2
)

# deploy_contracts' pins (deployment/contracts/pyproject.toml) + MuJoCo 3.3.0 (franka_sim, and the C library E4 links)
SIM_PINS=(
  mujoco==3.3.0 numpy==2.2.6 scipy==1.15.3 PyYAML==6.0.3 msgpack==1.1.0 pyzmq==27.1.0
  opencv-python-headless==4.12.0.88 pexpect==4.9.0 ptyprocess==0.7.0
  absl-py==2.5.1 etils==1.13.0 fsspec==2026.9.0 glfw==2.11.0 importlib_resources==7.1.0 PyOpenGL==3.1.10
  typing_extensions==4.16.0 zipp==4.1.2
  # sim_deploy/ui (python -m sim_deploy.ui): aiohttp and its deps, the versions of ~/envs/teleop_collect
  aiohttp==3.10.10 aiohappyeyeballs==2.7.1 aiosignal==1.4.0 async-timeout==4.0.3 attrs==26.1.0 frozenlist==1.8.0
  idna==3.20 multidict==6.9.1 propcache==0.5.4 yarl==1.25.1
)

light() {
  local env=$ENVS/sim_policy_deploy
  make_env "$env"
  "$env/bin/pip" install "${LIGHT_PINS[@]}"
  "$env/bin/pip" install --no-deps -e "$REPO/deployment/contracts" -e "$REPO/deployment/policy_deploy"
  (cd /tmp && "$env/bin/python" -c "import cv2, scipy, zmq, av, deploy_contracts, policy_deploy; \
print('sim_policy_deploy OK:', policy_deploy.__file__)")
}

torch_env() {
  local env=$ENVS/sim_policy_deploy_torch
  make_env "$env"
  "$env/bin/pip" install pip==26.3
  "$env/bin/pip" install --extra-index-url "$CU118" "${TORCH_PINS[@]}"
  "$env/bin/pip" install --no-deps -e "$REPO/behavior_prompting" -e "$REPO/deployment/contracts" \
    -e "$REPO/deployment/policy_deploy"
  (cd /tmp && "$env/bin/python" -c "import torch, dill, hydra, timm, diffusers, behavior_prompting, zarr, \
deploy_contracts, policy_deploy; print('sim_policy_deploy_torch OK: torch', torch.__version__, \
behavior_prompting.__file__, policy_deploy.__file__)")
}

sim() {
  local env=$ENVS/sim_deploy
  make_env "$env"
  "$env/bin/pip" install "${SIM_PINS[@]}"
  "$env/bin/pip" install --no-deps -e "$REPO/deployment/contracts"
  # sim_deploy/ui/live.py builds the policy's view with the copied policy_deploy's ImagePrep / PacketSelector: the
  # copy installed editable without its deps (its light pins numpy 2.2.6, opencv-headless 4.12.0.88, scipy 1.15.3,
  # PyYAML 6.0.3 are the same as SIM_PINS)
  "$env/bin/pip" install --no-deps -e "$REPO/deployment/policy_deploy"
  # controlSW/franka_sim has no pyproject: it is run from its folder (python -m franka_sim). A .pth line puts that
  # folder on sys.path (franka_sim itself appends controlSW/franka_deploy), the same as an editable install.
  local site
  site=$("$env/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  echo "$REPO/controlSW/franka_sim" > "$site/sim_deploy_franka_sim.pth"
  (cd /tmp && "$env/bin/python" -c "import mujoco, zmq, cv2, yaml, msgpack, pexpect, scipy, aiohttp, deploy_contracts, \
franka_sim.model, policy_deploy.frames._dataproc.imageprep; print('sim_deploy OK: mujoco', mujoco.__version__, \
franka_sim.model.__file__)")
}

targets=("$@")
[[ ${#targets[@]} -eq 0 ]] && targets=(light torch sim)
for t in "${targets[@]}"; do
  case $t in
    light) light ;;
    torch) torch_env ;;
    sim) sim ;;
    *) echo "setup_envs: unknown target $t (light, torch, sim)" >&2; exit 2 ;;
  esac
done
