# Real-arm trials

Where each trial run on the arm is, and what happened. Add a row per trial. How to record one: [DIRECTIONS.md](DIRECTIONS.md),
"Running trials", step 5.

## Where things are (on the laptop)

- **Recordings:** `~/Capstone/data/<task>/policy/<run>/deploy/<date>_<HH-MM>_trialN_probe_live/`.
  - One folder per trial, inside the folder of the policy that ran.
  - Trial numbers count per policy.
  - Each folder holds `NOTES.md` (policy, session, outcome, clamps) and `NNNNNN_{main,ultrawide}_{0,1}.png`: the policy's four 224×224 inputs, about 10 sets a second.
  - Each folder also has `<folder name>.mp4`, a video of all the sets: a 2×2 grid (main on top, ultrawide below; observation steps 0 and 1), played at 10 fps so it runs in roughly real time. Made by `python -m policy_deploy.frames.trial_video` (DIRECTIONS.md "Running trials" step 5.5); the PNGs are kept.
- **Logs:** the session folder named in `NOTES.md`, under `~/policy_deploy_sessions/<date>_<time>/`. It holds `episodes.jsonl`, `manager.log` (clamps, faults) and `manager/steps.jsonl`.
- **Not in git:** `~/Capstone/data/` is local only. `train/fetch.sh` never deletes local files, so the `deploy/` folders survive a re-fetch.

## 2026-11-20: first day on the arm

Settings for every trial: bf16, 8 diffusion steps, exec_h 12. The start pose was the sim's `q_start_rad`, not yet checked
on the arm.

| Time  | Policy (`$D/...`)                                  | What it is                          | Trial | Sets | Outcome |
|-------|----------------------------------------------------|-------------------------------------|-------|------|---------|
| 17:22 | `PICK_TEST1/policy/run_022_2026-11-19`             | plain diffusion, 30 eps (ep 50)     | 2     | 701  | no clamp yet; no notes on the outcome |
| 17:26 | `PICK_TEST1/policy/run_022_2026-11-19`             | same                                | 3     | 357  | same session as trial 4 |
| 17:28 | `PICK_TEST1/policy/run_022_2026-11-19`             | same                                | 4     | 155  | |
| 17:31 | `PICK_TEST1/policy/run_006_2026-11-18`             | behavior prompting, 30 eps (ep 59), prompt 0 | 1 | 238 | the rot6d check failed once (\|cos\| 0.13 > 0.1) |
| 17:34 | `PICK_DOUBLE/policy/run_003_2026-11-19`            | plain, "15-15" (ep15-v1, ep 59)     | 1     | 244  | around here the PC's gate rejected z −0.019, so the laptop clamp was added |
| 17:38 | `PICK_DOUBLE/policy/run_003_2026-11-19`            | same                                | 2     | 109  | first with the clamp: 11 chunks clamped; user stop |
| 17:41 | `PICK_DOUBLE/policy/run_002_2026-11-19`            | plain, double-62 (ep 40), candidate | 1     | 551  | 46 chunks clamped; user stop |
| 17:45 | `PICK_DOUBLE/policy/run_002_2026-11-19`            | same                                | 2     | 219  | 29 clamped; 16.1 s; user stop |
| 17:52 | `PICK_DOUBLE/policy/run_002_2026-11-19`            | same                                | 3     | 330  | 81 clamped; 27.5 s; user stop |

`run_022` trial 1 was not recorded.

**What it showed:** every PICK_DOUBLE policy aims 2–8 cm below the table (z down to −0.079 m), so the laptop clamps
dozens of chunks and the episodes end on a user stop. The suspects:
- the start pose: the sim's `q_start_rad` was never checked on the arm, and the policies act relative to it;
- the table may not be at base z ≈ 0.

Next steps:
1. With the arm low, measure fingertips to table with a ruler.
2. Hand-guide a start about 20 cm up and tilted about 48°.
3. Paste the `q_start_rad` that `pc_probe --start-pose` prints into the PC's `control_deploy/station.yaml`.
