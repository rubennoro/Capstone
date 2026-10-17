# Lead's reconciliation of P1 and P2 (binding; overrides either plan where they conflict)

Files: BRIEF.md, P1_data_collection_plan.md (P1), P2_camera_calib_deploy_plan.md (P2), all in this directory.

## R1. Layout
Project root `/home/ruben/Capstone/sensor_rig/` (holds `pyproject.toml`, `README.md`, `tests/`), Python package in
`/home/ruben/Capstone/sensor_rig/sensor_rig/`. Tests: `sensor_rig/tests/`. Run tests with
`<scratchpad>/env_iphumi/bin/python -m pytest sensor_rig/tests` from the repo root (pytest + pyyaml installed there;
the package is importable by adding `sensor_rig/` to `sys.path` in `tests/conftest.py`, or `pip install -e` into the
scratch env — WP-A decides and documents). `sensor_rig/tests/conftest.py` is owned by WP-A; WP-C may add fixtures in
`tests/fixtures_sim.py` and import them, not edit conftest.

## R2. P2's API is canonical for geometry/config
P1's "Assumed interfaces" §1 is superseded by P2 §3.1/§4.1/§8. Mapping for P1 code:
| P1 assumed | Use instead (P2) |
|---|---|
| `load_rig_from_text(text)`, `rig.text`, `rig.sha256` | raw episode stores `rig.dump(path)` output as `rig.yaml`; read back with `load_rig(path)`; provenance = `rig.fingerprint()` |
| `rig.time_offset_ns(cam)` | `rig.capture_time_ns(cam, t)` for cameras; `rig.imu.time_offset_ns` for the IMU |
| `rig.T(a,b)` frame `"imu0"` | `rig.T(a,b)` with frames `imu, main, ultrawide, virtual_main, virtual_ultrawide` (OpenCV axes) |
| `rig.world_offset` | `rig.export.T_world_offset` |
| `rig.imu_noise` | `rig.imu.noise` (`.for_vio()`) |
| `virtual_from_physical(rig, name, scale)` | `rig.remap(name, "export" | "deploy" | (w,h))` |
| `build_remap` used inversely in sim | `rig.inverse_remap(name, virt)` / `camera_model.build_inverse_remap` |
| `RemapTable.map_x/map_y` | use only `.apply()`, `.valid`, `.coverage`, `.src_size`, `.dst_size` |
| `rig.device_identifier/device_name` | `rig.export.device_identifier / device_name / gripper_id` |
| pose chain | `T_out = rig.export.T_world_offset @ T_YUP_ZUP @ T_W_S(t) @ rig.T_S_virtual_main_arkit()` (P2 §8) |
Time convention (both agree): `t_capture = t_stamp + offset`. Raw files store unmodified stamps.

## R3. rig.yaml: no duplicated settings
P2's schema owns sensor identity and settings: `cameras.<c>.{sensor, sensor_id, sensor_mode, resolution, fps, argus{...}}`
and `imu.{type, spi_device, spi_speed_hz, spi_mode, odr_hz, accel_range_g, gyro_range_dps, fifo{...}, ...}`.
P1's `capture:` section shrinks to recorder-only settings and is added to P2's schema as an optional top-level mapping
passed through as `RigConfig.capture: dict` (not validated by P2 beyond being a mapping):
```yaml
capture:
  backends: {main: argus, ultrawide: argus, imu: icm42688_spi}   # or mock
  mock: {pattern: moving_checker}                                # optional mock settings
  jpeg: {encoder: opencv, quality: 90, workers_per_camera: 2, queue_seconds: 0.75}
  live_abort: {min_fps_ratio: 0.52, after_s: 1.0, min_disk_free_gb: 2.0}
```
Drivers read width/height/fps/sensor_id/mode/exposure from P2's camera section (`argus` dict holds exposure_us, gain,
wbmode, aelock, awblock, etc. — P2 defines the keys, WP-B consumes them; WP-B may ask the lead for extra keys).

## R4. Camera backend interface (one interface for recorder AND deployment)
Use P1 `sensor_rig/drivers/base.py` (`Frame{image,t_ns,seq,t_arrival_ns,meta}`, `CameraBackend{start,read,latest,stop}`,
`make_camera`, `make_imu`) and ADD `read_recent(n) -> list[Frame]` (newest last, ring ≥ 8) for deployment pairing.
P2's `sensor_rig.capture`/`open_camera_source`/`wait_next` names are dropped; `RigFrameSource` uses
`make_camera(rig, name, backend="mock"|"argus", **opts)` and `read()`/`read_recent()`. `Frame.t_ns` is the stamp the
recorder writes to data.csv (CLOCK_MONOTONIC, before rig offsets; Argus MONOTONIC_RAW converted in the backend).
`MockCamera.replay(raw_episode_dir, cam, realtime=True, loop=False, preserve_timestamps=False)` is frozen (D10d uses it).

## R5. Accepted refinements
- UW export decimated to ~10 Hz by default (P1 §8.1). yuv420p MP4 (P1 §8.3), verified by the e2e test.
- Calibration episodes record main frames too; export writes only UW mp4 + JSON (P1 §8.4).
- Native grid when main fps ≥ 50, else 60 Hz fill grid (P1 §8.7).
- Default lenses in `rig_default.yaml` = covering lenses per P2 §11.1 (main ≥ ~90° diag, UW fisheye ~140–160°), placeholders.
  A test pins the 77°/120° coverage shortfall. Default main = 30 fps full-FOV 4:3 mode (P2 §11.2); export fills to 60 Hz.
- Half-pixel scaling convention (P2 §2). Sim uses each iPhone episode's own K (P2 §11.8).
- `T_S_main` derived, not stored (P2 §4.2).

## R6. Deployment upstream issues (P2 §10)
Only rig-local mitigations are approved now: rig config `camera_latency_s: 0` with latency applied once via
`capture_time_ns`, and the libx264 override inside RigCamera. Do NOT edit other upstream real-env/behavior_prompting
files beyond what P2 §1 lists (new rig_camera.py, two yaml configs, +1 pyproject line). The rest go to the user.

## R7. Shared files
`sensor_rig/pyproject.toml`, `sensor_rig/sensor_rig/__init__.py`, `__main__.py`, `cli.py`, `tests/conftest.py`: WP-A.
`cli.py` registers subcommand groups by lazily importing `sensor_rig.calib.cli.register` (WP-G),
`sensor_rig.record.cli.register` (WP-B), `sensor_rig.sim.cli.register` (WP-C) inside try/except ImportError so a
missing group doesn't break the others. `__init__.py` imports nothing heavy.

## R8. Rules for every builder
- Only create/edit files you own. If you need a change in someone else's file, write it under "Requests" in your final
  report (or, if blocked, code against the documented interface and stub locally inside your own test).
- Other builders are working in the same tree at the same time; their files may appear or change while you work.
- No git commits, no installs into the system Python. You may `pip install` pure-Python test deps into the scratch env
  only if truly needed (say so in the report).
- Never write into `iPhUMI/.../tmp_demonstrations`, `tmp_sessions`, or `data/raw/`. Treat `data/raw/` as read-only input.
- Definition of done includes: your tests pass in the scratch env; report exact test command + result, deviations from
  the plans, and open TODOs that need hardware.

Work-package naming: WP-G = P2 WP2.1 + WP2.2 (geometry core + rig config + calibration); WP-D = P2 WP2.3 (deployment);
WP-A, WP-B, WP-C as in P1 §6.
Phase 1 (parallel): WP-G, WP-A, WP-B. Phase 2 (parallel): WP-C, WP-D.
