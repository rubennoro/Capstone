# physicalSetup status

The single source of truth for implementation progress. Plan: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Pause
and resume protocol: plan section 8. Design: [PIPELINE.md](PIPELINE.md).

Last gate commit: `374d3dc` (docs only). Sensors arrive 2026-10-29.

## Tasks

| Wave | Task | State | Worklog | Notes |
|---|---|---|---|---|
| 0 | T0 build skeleton + frozen interfaces | running | [worklogs/T0.WORKLOG](worklogs/T0.WORKLOG) | started 2026-10-27 |
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

## Next actions

1. T0 running → orchestrator review → gate → commit.
2. Then launch Wave 1 (T1–T4 in parallel).
