# physicalSetup status

The single source of truth for implementation progress. Plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Pause
and resume protocol: plan section 8. Design: [PIPELINE.md](PIPELINE.md).

Last gate commit: see `git log` (Wave 0 = T0). Sensors arrive 2026-10-29.

## Tasks

| Wave | Task | State | Worklog | Notes |
|---|---|---|---|---|
| 0 | T0 build skeleton + frozen interfaces | done | [worklogs/T0.WORKLOG](worklogs/T0.WORKLOG) | gate passed 2026-10-27: 14/14 tests default, ASan+UBSan, TSan |
| 1 | T1 LSM6DSOX driver + simulator | todo | | |
| 1 | T2 camera backends + IMX519 focus | todo | | |
| 1 | T3 time/config/capture/shm/package | todo | | |
| 1 | T4 camera models + image prep + bindings | todo | | |
| 1 | R1 reviewer | todo | | |
| 2 | T5 ec-record | todo | | |
| 2 | T6 ec-viz + ec-slam-live | todo | | |
| 2 | T7 ec-calibrate + tag detector | todo | | |
| 2 | T8 bring-up kit + ec-doctor | todo | | |
| 2 | R2 reviewer | todo | | |
| H | hardware day H1–H7 | todo | | |
| 3–5 | T9–T12, calibration, deployment | todo | | |

## Open reviewer findings

(none)

## Contract notes from T0 (binding for T1–T4)

- Factories take the config: `make_camera(cfg)`, `make_imu(cfg, bus=nullptr)`; `start()` takes no argument.
- Sources, tests (`tests/<area>/test_*.cpp`) and apps (`apps/<name>/`) are globbed, so no CMakeLists edits are needed. `argus/`, `gst/`, `orbslam3/` and `ceres/` dirs build only with their option on.
- Session names are `[A-Za-z0-9-]+` (iPhUMI group.py:36-37 splits on `_`); the package kind is `rigcalibration`.
- Adding a field with a default, or a new header in your own module: worklog note. Changing a frozen declaration: tell the orchestrator.

## Next actions

1. ~~T0~~ done.
2. After the T0 gate: **launch Wave 1 (T1–T4) automatically** (user, 2026-10-27), then R1 → gate → commit.
3. **Stop after the Wave 1 gate.** Wave 2 (T5–T8) starts only after the user confirms.
