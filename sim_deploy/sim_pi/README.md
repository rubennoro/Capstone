# sim_deploy/sim_pi: the simulated Raspberry Pi 5 camera sender (E5)

A live stand-in for the Pi on the hand. It mirrors the sim PC's physics state and streams both rig cameras as
`pi_stream/v1` on ZMQ PUB :5555, so the copied `policy_deploy` frames process (`launcher.fake_pi: false`) receives
it as it would from the real Pi. Design and numbers: [../DESIGN.md](../DESIGN.md) section 6 and "E5: as built".

```
sim PC  ── sim.state/v1, PUB :5650 (qpos every 5 ms) ──►  sim_pi  ── pi_stream/v1, PUB :5555 ──►  policy_deploy frames
                                                          (GL render, lens, 640x480, JPEG q90, Pi clock)
```

## Run

Env `~/envs/sim_deploy` (mujoco 3.3.0, pyzmq, opencv, numpy; `sim_deploy` installed editable). Always `unset PYTHONPATH`.

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
~/envs/sim_deploy/bin/python -m sim_deploy.sim_pi --station sim_deploy/stations/sim_rig.yaml
```

Options (they override the station's optional `sim_pi:` block, whose keys are the fields of `settings.py`):
`--model scene.xml` (default `sim_deploy/build/<station id>/scene.xml`; it waits for `model.json`), `--state
tcp://127.0.0.1:5650`, `--bind tcp://127.0.0.1:5555`, `--rig-file` (default the copy's
`deployment/contracts/rigs/rig_pi_v3_640.yaml`), `--seconds N`, `--save-dir DIR --save-every N` (raw 640x480 PNGs),
`--stats-json FILE`, `--no-shadows`, `--send-delay-ms`. One status line per second:

```
[sim_pi] packets 1370 | skipped 0 | uw_reused 3 | stale 0 | GL 24.0 ms | encode 9.0 ms | state age 4.4 ms (9180 msgs)
```

`stale` counts frames whose state was held more than 10 ms (the sim PC is slow or gone); `skipped` ticks that could
not start in time; `uw_reused` packets that repeat the last ultrawide frame (an overrun of more than half a period).
A `model_sha256` or `nq` mismatch with the state export stops it with `sim_pi: FAILED ...`.

**Smoke run without the sim PC** (ports 5660, 5555, 5610):

```bash
M=sim_deploy/build/sim_rig/scene.xml
~/envs/sim_deploy/bin/python -m sim_deploy.sim_pi.state_stub --model $M --bind tcp://127.0.0.1:5660 &
~/envs/sim_deploy/bin/python -m sim_deploy.sim_pi --station sim_deploy/stations/sim_rig.yaml --state tcp://127.0.0.1:5660 &
cd deployment/policy_deploy
~/envs/sim_policy_deploy/bin/python -m policy_deploy.frames --mock --station ../../sim_deploy/stations/policy_sim.yaml &
~/envs/sim_policy_deploy/bin/python -m policy_deploy.frames.probe --station ../../sim_deploy/stations/policy_sim.yaml --mock --save /tmp/probe
~/envs/sim_policy_deploy/bin/python -m policy_deploy.frames.pi_collect --endpoint tcp://127.0.0.1:5555 --seconds 30
```

**Check G3** (cameras vs reality, PLAN section 6): renders the handheld gripper (kit `iphumi_umi_soft_v1` with real
ArUco tags, training's camera poses) through the full chain and compares tag corners with training's images of
PICK_TEST1 `pick1-v1` index 14 (= episode_000152). It exports the reference with the torch env once (zarr + JPEG-XL).

```bash
~/envs/sim_deploy/bin/python -m sim_deploy.sim_pi.handheld_check --out /tmp/g3 [--zarr ZIP --index 14 --frames N]
```

## Files

| File | What it does |
|---|---|
| `__main__.py` | CLI: settings, `SimPi`, signals |
| `settings.py` | `SimPiSettings` (defaults = the live Pi's timing) + the station's `sim_pi:` block + CLI |
| `sim_pi.py` | `SimPi`: 30 Hz schedule, state interpolation per camera instant, GL stage, 2 encoder threads, sender with the Pi clock |
| `cameras.py` | `CameraChain`: the views to render, remap to the native sensor image, INTER_AREA to 640x480 |
| `render.py` | `GLRenderer` (EGL on the NVIDIA GPU, asymmetric pinhole views via `mjvGLCamera`), lighting (`prepare_spec`) |
| `model_file.py` | loads `scene.xml` + `model.json`, recomputes the sha with E6's method, render-only model edits |
| `state.py` | `sim.state/v1` SUB thread, `StateBuffer.at(t)` (interpolation, never extrapolation) |
| `state_stub.py` | stand-in state export for smoke runs (home qpos, slow joint swing, fingers opening) |
| `_deploy.py` | imports from the copied `deployment/policy_deploy` (rig loader, camera models, `pi_stream` contract) |
| `handheld_check.py`, `handheld_reference.py` | check G3 (above) |

## How the images are made

- **Constants:** every lens number comes from the copy's rig file, loaded with the copy's `load_live_rig`; camera
  models are the copy's `PinholeRadtan` / `FisheyeEquidistant` (`physical_model`), scaled with `.scaled()` to the
  native sizes 1600x1200 / 1280x960 (training's recorded sizes, `rig_pi_v3.yaml`).
- **IMX519 (`main`):** one undistorted pinhole render at 1600x1200 (f 1231.25, widened to 1621x1217 to hold every ray
  of the distorted image), the rig's radtan applied by remap (`unproject` per native pixel), INTER_AREA to 640x480.
- **IMX219 (`ultrawide`):** a cube map around the lens, only the face rectangles the sensor sees (5 views, 2.45 Mpx,
  f 449.4 = the fisheye's centre density), remapped into the 1280x960 equidistant fisheye with the measured lens,
  INTER_AREA to 640x480. The measured lens reaches about 98° in the sensor corners, past the copy model's 89° cap;
  the same polynomial is used there (monotonic up to 106.7°), so the whole sensor shows the scene, as in the real
  frames (no black image circle on the real IMX219 either).
- Each camera's own board and lens barrel (`rig_imx519_board`, `rig_imx219_board`) is not drawn in its own views.
- **Light:** the model's lights (made directional, no spot cone) with shadows; the headlight keeps only its ambient
  part (its diffuse would follow each cube face and draw seams). Sites are not drawn.
- JPEG quality 90, RGB to BGR only for the encoder. Nothing the real Pi does not do (no unwarp, crop or letterbox).

## Timing (`pi_stream/v1`, PI_PAYLOAD.md section 0)

Pi clock = `CLOCK_MONOTONIC + 1000 s + 20 ppm`. Tick k: main mid-exposure `t0 + k/30 s`, ultrawide 16.3 ms later
(the live pair gap). Each camera is rendered from the state interpolated at its own mid-exposure (waiting up to
10 ms for the state to cover it). `t_sensor_ns = t_mid_ns + exposure/2` (exposures 25 640 / 33 287 us, the live
Pi's), `packet_no = main.frame_no = k`, `skew_ns = ultrawide.t_mid − main.t_mid` (the live sign), `uw_reused` repeats
the previous ultrawide stamps, and each packet leaves 45.8 ms after its main `t_sensor_ns` (the live Pi), with
`t_send_ns` read just before sending. Every packet is checked with `PiStreamPacket.from_parts` before it is sent.
