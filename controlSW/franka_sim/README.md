# franka_sim

MuJoCo simulation of the Franka arm (Panda + Franka Hand, FR3) used to evaluate policies before hardware day.
Package: official [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) models, MuJoCo Panda/FR3 + Franka Hand model, kinematics, SimCore/MuJoCoBackend, scene, renderer, Stage 1/1b campaign, and the eval CLI (`python -m franka_sim --help`: check-assets, model, stage1, campaign, camera-check, camera-gap, finger-kit, characterize, eval).

- Assets: `franka_sim/assets/` (pinned commit + sha256 per file in `assets/SOURCE.md`, Apache-2.0 licences kept).
  Re-vendor with `python -I franka_sim/tools/vendor_menagerie.py --commit <sha|HEAD> --scratch <empty dir>`.
- Environment: `env_realenv` has mujoco 3.3.0. `import franka_sim` never imports mujoco, so `--help` works elsewhere.
- Headless rendering: `MUJOCO_GL=egl`; call `renderer.close()` explicitly.

```
cd controlSW/franka_sim
python -m franka_sim --help
python -m franka_sim check-assets
env -u PYTHONPATH python -m pytest -q tests     # mujoco tests skip when mujoco is missing
```

Finger model (modes stock/box/kit, frames, numbers with sources, validation, known limits): [docs/finger_model.md](docs/finger_model.md).
