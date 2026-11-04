# Station and run configuration

One home for every constant of the Franka deployment and its simulator. The loader is
`controlSW/franka_deploy/franka_deploy/station.py` (stdlib + numpy + yaml only, so it runs on the Jetson).
Consumed by the launcher, `run_task.py`, `fdcal` and `franka_sim`.

## Layout

```
station/                      what exists physically (changes when hardware changes)
  robot/    panda, fr3        joints, Cartesian limits, impedance, FCI, Desk end effector, bridge watchdog/slack, home
  tooling/  franka_hand_*     hand, finger adapter, width map, F_T_TCP, C_T_TCP, payload, collision boxes, sensor_compat
  sensor/   iphone16pro_wifi, rig1_wifi   device, transport, streams (K), clock, health, task image-age
  scene/    block_raise_table table, eraser, safety workspace box + cone, placement grid
  host/     jetson_orin_nano, pc_laptop   paths: work_root, python envs, checkpoint, prompt zarr, device
run/                          what one experiment chooses
  policy/   bp_run001, dp_template, demo_replay, fake   server kind, checkpoint, contract, task timing
  link/     ideal, wifi_good|busy|bad[_skew]   latency, jitter, stalls, bandwidth, clock skew
  criteria/ block_raise       success rule, stage gates, v_max / w_max
  sim/      physics, render, video_replay, rig_raw_replay, stage1
  sweeps/   block_raise_default, block_raise_smoke
```

Every file starts with `schema: <kind>.<group>/v1` and `id: <file stem>`.

## Where the safety numbers live

- Safety box and approach cone: `scene.workspace` (min_m, max_m, cone_axis, cone_max_deg).
- Speed limits `v_max_m_s` / `w_max_rad_s`: `criteria.limits` (a run choice; 0.5 m/s only after B3-hw).
- Watchdog (`watchdog_hold_s`, `watchdog_stop_s`) and `slack_s`: the robot's `bridge:` block (robot independent).
- `bridge_fragment()` assembles all of them into the shape of `franka_deploy/configs/bridge.yaml`.

## Adding a value

1. Put it in the right group file. The key name carries the unit: `_m _rad _deg _s _hz _kg _n _nm _m_s _rad_s _px _ppm`
   (plus the extended list `UNIT_SUFFIXES` in station.py). Trailing `_min/_max/_p95/_p99/_p999` are ignored, and a
   unit on any ancestor key counts (`latency_s: {main: 0.08}`).
2. Numeric values without a unit are only allowed under the keys in `NON_SUFFIXED_ALLOW`: `K`, `T_*`, `F_T_*`,
   `C_T_*`, `*_T_*`, `width_map.a/b`, `n_*`, `seed`, `*_version`, `*_index`, `ports`, `exec_h`, `jpeg_quality`,
   `friction`, `max_queue`, `max_runs`, `nullspace_stiffness`, `filter_coeff`, `impratio`, `cone_axis`, a few criteria
   counts. Add to the list (and justify it) rather than dropping a unit.
3. Give it provenance (below). The validator fails on a value without one.
4. If a real run must not start while the value is a guess, add its path to `critical:`.
5. Run `python -m franka_deploy.station validate --all` and `pytest tests/test_station.py`.

## Provenance

`provenance:` maps a dotted path to `{status, source, note?}`. Entries are prefix-inherited: the longest matching key
wins, and `""` covers the whole file. Every leaf (a scalar, a list of scalars, or `null`) must resolve to one, and a
key that matches no value is an error.

| status | meaning |
|---|---|
| measured | measured on this setup or from the training data |
| datasheet | vendor documentation |
| derived | computed from other values or files |
| decision | a design choice (constants copied from today's configs are decisions with the file:line) |
| placeholder | a guess to be replaced |
| hw | must be read or set on the real hardware (Desk, libfranka, CAD, calibration) |

## Finger kit keys (tooling.finger_adapter)

Optional; absent keys keep old files valid. `kit` (null = box fingers, or a kit dir relative to controlSW) and
`kit_sha256` (64 lowercase hex, required when `kit` is set, must equal `kit_digest(kit dir)`) select an STL finger kit;
`adapter_length_m` (>= 0, L), `lateral_offset_m` (`auto` or a number, dy), `adapter_yaw_deg` and `adapter_section_m`
(two positive numbers, placeholder block cross-section) place it. station.py checks types only; franka_sim checks
that the kit exists and matches the digest.
`mode` (`stock` | `box` | `kit`; absent = `kit` if `kit` is set, else `box`) picks the fingers; `mode: kit` requires `kit`.
`kit_variant` (`[a-z0-9_]+`, default `full`) picks a variant defined in the kit's `kit.yaml` (`full` = soft finger +
holder with tags, `finger_only` = soft fingers only). Number keys must be finite (no nan/inf); `adapter_yaw_deg` too.

## Critical

`critical:` lists paths (or subtrees) that real profiles must not run with. A critical path whose status is
`placeholder` or `hw` is reported by `Station.placeholders(critical_only=True)`. Real profiles refuse these (B2),
sim profiles print a warning, and `allow_placeholders` in the deploy config overrides it (logged).

## Interpolation, overrides, profiles

- `${group.path}` (for example `${host.checkpoint}`) is resolved after all groups load. A reference to a group that is
  not loaded, or a cycle, is an error.
- Overrides: `sets=["link.latency_s=0.05", "tooling.width_map.b=0.01"]`. The value is parsed as YAML, the path must
  exist, and applied overrides are recorded in `Station.overrides` and `to_yaml()`.
  Precedence: group file < `station_set` < explicit `bridge.override` / `controller.override`.
- `load_station({"station": {robot, tooling, sensor, scene, host}, "run": {policy, link, criteria, sim}})` takes any
  subset. Profiles (`PROFILES` in station.py): `sim-iphone`, `sim-rig`, `sim-iphone-fake`, `sim-rig-fake` (host
  `pc_laptop`), `real-iphone`, `real-rig` (host `jetson_orin_nano`, no link or sim).
- `Station.fingerprint()` is the first 16 hex characters of sha256 over the canonical JSON of all group values
  (not provenance); it names run directories.

## CLI

```
python -m franka_deploy.station validate --all                  # every file + every profile; prints critical placeholders
python -m franka_deploy.station validate --spec station.robot=fr3 --spec station.host=pc_laptop --set link.seed=3
python -m franka_deploy.station validate --all -v               # also list non-critical placeholders
```

Exit code 0 (OK) or 1 (errors on stderr).
