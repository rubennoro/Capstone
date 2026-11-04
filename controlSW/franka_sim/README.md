# franka_sim

MuJoCo simulation of the Franka arm (Panda + Franka Hand, FR3) used to evaluate policies before hardware day.
Package: official [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) models, MuJoCo Panda/FR3 + Franka Hand model, kinematics, SimCore/MuJoCoBackend, scene, renderer, Stage 1/1b campaign, and the eval CLI (`python -m franka_sim --help`: check-assets, model, stage1, campaign, camera-check, camera-gap, finger-kit, characterize, eval).

- Assets: `franka_sim/assets/` (pinned commit + sha256 per file in `assets/SOURCE.md`, Apache-2.0 licences kept).
  Re-vendor with `python -I franka_sim/tools/vendor_menagerie.py --commit <sha|HEAD> --scratch <empty dir>`.
- Environment: `env_realenv` has mujoco 3.3.0. `import franka_sim` never imports mujoco, so `--help` works elsewhere.
  On the laptop it lives at `~/envs/env_realenv` (lean: no open3d/rerun/ur-rtde/arx5). Rebuild with `PYTHONNOUSERSITE=1`
  set, or pip counts `~/.local` packages as installed:
  ```
  export PYTHONNOUSERSITE=1
  conda create -y -p ~/envs/env_realenv -c conda-forge --override-channels python=3.10 pip
  ~/envs/env_realenv/bin/pip install numpy==2.2.6 scipy==1.15.2 pyyaml==6.0.2 msgpack==1.1.1 zarr==2.18.3 \
    opencv-python==4.12.0.88 omegaconf==2.3.0 hydra-core==1.3.2 robologger==0.1.7 robotmq==0.1.14 robot-utils==0.1.4 \
    transforms3d==0.4.2 flask==3.1.1 flask-socketio==5.5.1 python-socketio==5.13.0 python-engineio==4.12.2 \
    simple-websocket websocket-client mujoco==3.3.0 loguru pyzmq imageio==2.37.0 tqdm pytest pygments av
  ~/envs/env_realenv/bin/pip install --no-deps -e ../../behavior_prompting/deps/real-env -e ../../sensor_rig
  ```
  `~/envs/env_bp` (inference) reuses torch 2.9.1 from `~/.local/lib/python3.10/site-packages`. Do NOT set
  PYTHONNOUSERSITE for this install (pip would download a 3 GB CUDA torch). The launcher starts it with user site off, so a
  `.pth` adds `~/.local` at the end of sys.path, and a `sitecustomize.py` moves the user site behind site-packages when it
  is on (the env's numpy 1.26.4 must win over ~/.local's 2.2.6):
  ```
  conda create -y -p ~/envs/env_bp -c conda-forge --override-channels python=3.10 pip
  SP=~/envs/env_bp/lib/python3.10/site-packages
  echo ~/.local/lib/python3.10/site-packages > $SP/zz_user_torch.pth
  printf 'import site, sys\n_u = site.getusersitepackages()\nif _u in sys.path:\n    sys.path.remove(_u)\n    sys.path.append(_u)\n' > $SP/sitecustomize.py
  ~/envs/env_bp/bin/pip install --no-cache-dir -r ../franka_deploy/requirements-inference.txt \
    joblib threadpoolctl seaborn open3d transforms3d pytest
  ~/envs/env_bp/bin/pip install --no-cache-dir --no-deps -e ../../behavior_prompting
  ```
  GUI run (MuJoCo viewer, iPhone, real policy): `~/sim_runs/run_viewer.sh [POSES] [EPISODES_PER_POSE]`.
- Headless rendering: `MUJOCO_GL=egl`; call `renderer.close()` explicitly.

```
cd controlSW/franka_sim
python -m franka_sim --help
python -m franka_sim check-assets
env -u PYTHONPATH python -m pytest -q tests     # mujoco tests skip when mujoco is missing
```

Finger model (modes stock/box/kit, frames, numbers with sources, validation, known limits): [docs/finger_model.md](docs/finger_model.md).
