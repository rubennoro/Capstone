#!/usr/bin/env bash
# Build the sim environment ~/envs/env_sim (closed-loop MuJoCo sim, launcher, scripted policy; no torch).
# Usage: controlSW/franka_sim/setup_env.sh [ENV_PREFIX]     (default ~/envs/env_sim)
# PYTHONNOUSERSITE=1 keeps pip from counting ~/.local packages as installed.
set -euo pipefail
ENV=${1:-$HOME/envs/env_sim}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CONDA=${CONDA_EXE:-$HOME/anaconda3/bin/conda}
export PYTHONNOUSERSITE=1

"$CONDA" create -y -p "$ENV" -c conda-forge --override-channels python=3.10 pip
"$ENV/bin/pip" install --no-cache-dir numpy==2.2.6 scipy==1.15.2 pyyaml==6.0.2 msgpack==1.1.1 zarr==2.18.3 \
  opencv-python==4.12.0.88 omegaconf==2.3.0 hydra-core==1.3.2 robologger==0.1.7 robotmq==0.1.14 robot-utils==0.1.4 \
  transforms3d==0.4.2 flask==3.1.1 flask-socketio==5.5.1 python-socketio==5.13.0 python-engineio==4.12.2 \
  simple-websocket websocket-client mujoco==3.3.0 loguru pyzmq imageio==2.37.0 tqdm pytest pygments av
"$ENV/bin/pip" install --no-cache-dir --no-deps -e "$REPO/behavior_prompting/deps/real-env" -e "$REPO/sensor_rig"

"$ENV/bin/python" -c "import mujoco, robotmq, robot_utils, flask_socketio, cv2, real_env, sensor_rig" \
  && echo "env_sim OK: $ENV"
