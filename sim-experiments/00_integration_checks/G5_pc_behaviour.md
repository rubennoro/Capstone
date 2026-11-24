# G5: PC behaviour (hold still, out-of-box fault, watchdog, homing, start-pose refusals)

**Status:** done (pass, E7 and P3; "scripted waypoints tracked within a set error" not run as such) · **Date:** 2026-11-20 ·
**Who:** agent (E7, P3), documented by agent · **Plan ref:** sim_deploy/PLAN.md §6 (G5 row, P3 table), §9;
DESIGN.md "E4: as built", "E7: integration", "P1: start-pose port", "P3/P4"

`$S` below = `/tmp/claude-1000/-home-ruben-Capstone/20deb2cf-0c82-44ce-870d-39667f0b1130/scratchpad`, the session
scratchpad of the E7 and P3 agents: **temporary, not in sim_evals**. The small files from it that this record uses
are copied into [data/g5/](data/g5/).

## Question

Does the sim PC (the copied C++ `franka_control` with a MuJoCo arm) behave like the real PC: no motion under
`hold_still`, a latched `fault` for a waypoint outside the workspace box, `stopped` after 1 s without C3, homing to
each home, and (P3) the start-pose key `p` and its refusals?

## Why it matters for the real robot

Every later check and benchmark trusts the PC side. If the sim PC drifted, ignored the gate or the watchdog, a
policy result in the sim would say nothing about the Franka. The gate, supervisor and watchdog are the real C++
code, so a pass here also exercises the real deployment's safety logic (with the sim's box values, see Setup).

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (eval.yaml `git.sim_deploy` `3633fcb3…-dirty`; + uncommitted `sim_deploy/`, `deployment/`) |
| Model | E7 runs: `model.sha256` `701815be08cd…` (old table, top O z 0.05, pedestal). P3 0939 (superseded): `7746a4391b50…` (old table). P3 0943 / 0945: `166a0649571a…` (table top O z 0) |
| Homes | E7: `--home deploy` (TCP O (0.3069, 0.0000, 0.3542), 30.4 cm above the old table) and `--home trainstart` (TCP O (0.4384, -0.1640, 0.2113), q `[-0.6225485, 0.3138391, -0.1643961, -1.7825973, 0.5934631, 1.3978677, -1.6275796]`). P3: `--home start` with the P3 stand-in q `[-0.6419329, 0.3706554, -0.1347388, -1.8275169, 0.5918946, 1.4884701, -1.6761742]` (TCP O (0.4385, -0.1641, 0.1618), 16.2 cm above the table, tilt 49.5 deg, heading left; eval.yaml `home.start_pose`) |
| Eraser placement | E7 deploy: `scene/randomization.yaml`; trainstart: `randomization_trainstart.yaml`; P3: `randomization_start.yaml` (seeded) |
| Stations | E7: `stations/control_sim.yaml` / `control_sim_trainstart.yaml`; P3: `control_sim.yaml` + `start_pose.yaml` merged by run.sh into `build/control_sim_start.merged.yaml` (deploy: `q_start_rad` null). No overlay |
| Workspace box in force | E7 runs: **min x 0.250, z 0.050, max z 0.700 m** (the copy's `config.yaml` values, read from the gate's error texts and `pulled back onto box: x ... -> 0.250 m` lines; E7 fix 12's sim box came after these runs). P3: min [0.20, -0.40, -0.005], max [0.80, 0.40, 0.70] m |
| Other gate / watchdog values | `v_max_m_per_s` 0.25, cone 60 + 2 deg (`config.yaml`); watchdog `timing.watchdog_limit_s` 1.0 s (`deployment/contracts/link.yaml`) |
| Policy | `hold_still` (mock kind), and `replay:pick1-v1.zarr.zip#3` to provoke the out-of-box fault |
| Seeds / episodes | `--seed 0` (default 0); 1-2 episodes per eval (see Results) |
| Timeout | 15, 20, 30 or 10 s per eval (eval.yaml `judge.timeout_s`, see Results) |
| Sim-only settings | box min (P3), finger contact pairs, width offset 0.0, appearance (sim-experiments/README.md table); none specific to G5 |

## Method

Pass criteria (PLAN §6): `hold_still` = no motion; scripted waypoints tracked within a set error; out-of-box waypoint
-> `fault`; no C3 for 1 s -> `stopped`; homing works. P3 adds: `p` reaches the start pose; `c` away from the start
pose and `p` away from home are refused.

1. **Hold still.** `hold_still` sends the anchor back as every waypoint. The driver records the episode; E7 also
   recorded C2 at 100 Hz with `$S/e7/c2log.py` (read-only C2 subscriber, into `$S/e7/g5_*_c2.jsonl`); P3 uses the
   driver's `trace.json` TCP column (20 Hz ground truth, added by P3 fix 3). Drift = max |p - p0| (mm) and max
   rotation angle of R0ᵀR (deg) over each running segment, p0/R0 its first sample:
   [data/g5/drift.py](data/g5/drift.py) (same computation as E7's `c2stats.py`), result [data/g5/drift.csv](data/g5/drift.csv).
2. **Out-of-box waypoint -> fault, and recovery.** Replay of PICK_TEST1 episode 3 from `--home trainstart`: its pinch
   carries the TCP below the old box floor (z 0.05 + 10 mm margin), the gate rejects the waypoint and the PC latches
   `fault`. The second run (0143) checks that run.sh restarts the sim and resumes the eval folder (E7 fixes 6-8).
3. **Watchdog.** `hold_still`, 2 episodes; during episode 0 `$S/e7/stall_manager.sh` waits for C2 `running`, sleeps
   3 s, then `kill -STOP`s the policy_deploy manager for 1.5 s and `kill -CONT`s it (no C3 for 1.5 s).
4. **Homing.** Every episode starts with PC `h` (and `p` for P3); the driver checks the TCP after homing.
5. **P3 refusals.** (a) `c` away from the start pose: the driver with `--home deploy` against a sim PC that has the
   start pose set (sim PC started by hand, not by run.sh, which would match it to `--home`). (b) `p` away from home:
   in the same hand-started sim PC, `p` sent again via the driver's scene link right after `p` had reached the start
   pose.
6. "Scripted waypoints tracked within a set error" was **not run as a separate test**; the closest evidence is G6's
   replay path error ([G6_replay.md](G6_replay.md)).

## Commands to reproduce

Each needs the sim ports free (5555, 5610, 5620, 5630, 5640, 5650, 5651) and must not run alongside another sim.
About 1-2 min per eval (P3 start pose: + 15.4 s for `p`).

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
# P3 (current layout): hold still from the start pose
sim_deploy/run.sh --policy hold_still --episodes 1 --seed 0 --tag p3_g5_hold_z0 --timeout 20
# E7-style holds from the other homes (E7 used --timeout 20 --tap; deploy home 35.5 cm above the table now)
sim_deploy/run.sh --policy hold_still --episodes 2 --seed 0 --tag g5_hold_deploy --timeout 20 --tap --home deploy
# out-of-box fault + run.sh restart/resume (E7 used --home trainstart, now an alias of start; with the current box
# floor z -0.005 episode 3 may no longer leave the box: check the outcome)
sim_deploy/run.sh --home start --policy "replay:/home/ruben/data/umi/datasets/PICK_TEST1/pick1-v1/pick1-v1.zarr.zip#3" \
  --episodes 2 --tag g5_fault_recovery --timeout 30
```

Watchdog (two terminals; E7 ran the eval against a hand-started sim, `$S/e7/sim_up.sh`, copied logic in
[data/g5/stall_manager.sh](data/g5/stall_manager.sh), which waits on a C2 recording made by
[data/g5/c2log.py](data/g5/c2log.py)):

```bash
# terminal 1
cd ~/Capstone-main-sim && unset PYTHONPATH
sim_deploy/run.sh --policy hold_still --episodes 2 --tag g5_watchdog --timeout 15 --home deploy
# terminal 2, once episode 0 runs: freeze the manager for 1.5 s
pid=$(pgrep -f "python[0-9.]* .*-m policy_deploy\.manager" | head -1); kill -STOP $pid; sleep 1.5; kill -CONT $pid
```

`c` refusal (P3): start the sim PC by hand with the start pose merged (as run.sh step 2 does for `--home start`,
see `sim_deploy/run.sh` lines 8-12 and 94), start the sim Pi, then run the driver with `--home deploy`:

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
~/envs/sim_deploy/bin/python -m sim_deploy.eval run --policy hold_still --episodes 1 --seed 0 \
  --tag p3_g5_refuse_c_deploy --home deploy --timeout 10       # expected outcome start_failed
```

Drift from the raw files (read-only): `~/envs/sim_deploy/bin/python -B data/g5/drift.py` (in this folder).

## Results

### Per eval

| Eval (`sim_evals/...`) | Stage | Home | Model | Episodes: outcome | Detail (raw) |
|---|---|---|---|---|---|
| `_baseline/hold_still/2026-11-20_0130_g5_hold` | E7 | deploy | `701815be` | 2: timeout, timeout (15 s) | gate 1817 / 2027 accepted, 0 shortened / pulled back / rejected; no C2 recording of this run |
| `_baseline/hold_still/2026-11-20_0134_g5_hold_deploy` | E7 | deploy | `701815be` | 2: timeout, timeout (20 s) | 0 shortened / rejected; policy 8.3 ms p50; drift below |
| `_baseline/hold_still/2026-11-20_0136_g5_hold_trainstart` | E7 | trainstart | `701815be` | 2: timeout, timeout (20 s) | 0 shortened / rejected; drift below |
| `_replay/PICK_TEST1/pick1-v1_ep003/2026-11-20_0141_g5_fault_trainstart` | E7 | trainstart | `701815be` | ep 0 fault at 1.16 s; ep 1 `not_run` | "safety gate rejected waypoint 6 of chunk 34: outside workspace box by 0.016 m on z (z = 0.034 m < min 0.050 m, margin 0.010 m)"; gate 164 accepted, 82 shortened, 3 pulled back, 2 rejected. Ep 1: "policy_deploy did not start" (problem: "no [idle] within 180 s"): the failed start was not recognised (E7 fix 6) |
| `_replay/PICK_TEST1/pick1-v1_ep003/2026-11-20_0143_g5_fault_recovery_43` | E7 | trainstart | `701815be` | 2: fault (1.16 s), fault (1.10 s) | z = 0.037 m and 0.038 m < min 0.050 m; `run.sh` "the sim PC latched a fault: restarting the sim (1) and resuming", "resuming ...: 1 of 2 episodes done" (`_logs/2026-11-20_014332/eval.log`, [data/g5/run_fault.log](data/g5/run_fault.log)) |
| `_baseline/hold_still/2026-11-20_0145_g5_watchdog` | E7 | deploy | `701815be` | ep 0 **stopped** at 4.36 s; ep 1 timeout (15 s) | ep 0 C2 modes: holding 78, running 358, stopped 106 samples, no error text; ep 1 homed and ran 15 s in the same sim PC |
| `_baseline/hold_still/2026-11-20_0939_p3_g5_hold` (**SUPERSEDED.txt**: old table) | P3 | start | `7746a439` | 1: timeout (20 s) | 0 shortened / rejected; TCP start z 0.2113 m (old table) |
| `_baseline/hold_still/2026-11-20_0943_p3_g5_hold_z0` | P3 | start | `166a0649` | 1: timeout (20 s) | `t_home_s` 2.06, `t_start_pose_s` 15.45, `max_joint_err_rad` 0.000114 (tol 0.05), width 0.0800 m; gate 3622 accepted, 0 shortened / pulled back / rejected; policy 2.6 ms p50 |
| `_baseline/hold_still/2026-11-20_0945_p3_g5_refuse_c_deploy` (NOTE.txt) | P3 | deploy vs start-pose PC | `166a0649` | 1: **start_failed** | note: "c: the PC refused start: refused: not at the start pose: press h, then p on the PC" |

### Hold-still drift (raw: [data/g5/drift.csv](data/g5/drift.csv))

| Eval | Source | Segment (episode) | Duration (s) | Max translation (mm) | Max rotation (deg) | Width (m) |
|---|---|---|---|---|---|---|
| 0134_g5_hold_deploy | `$S/e7/g5_hold_c2.jsonl` (100 Hz) | 0 | 20.1 | 0.036 | 0.089 | 0.0800 |
| 0134_g5_hold_deploy | same | 1 | 20.1 | 0.040 | 0.090 | 0.0800 |
| 0136_g5_hold_trainstart | `$S/e7/g5_hold_ts_c2.jsonl` (100 Hz) | 0 | 20.1 | 0.069 | 0.122 | 0.0800 |
| 0136_g5_hold_trainstart | same | 1 | 20.1 | 0.068 | 0.121 | 0.0800 |
| 0939_p3_g5_hold (superseded) | `trace.json` (20 Hz, 0.01 mm steps) | 0 | 20.0 | 0.070 | 0.123 | 0.0800 |
| 0943_p3_g5_hold_z0 | `trace.json` (20 Hz, 0.01 mm steps) | 0 | 20.0 | 0.066 | 0.121 | 0.0800 |

### Watchdog timeline (raw: `$S/e7/g5_wd_c2.jsonl`, C2 monotonic time; `$S/e7/stall.log`)

| C2 time (s) | Mode | Note |
|---|---|---|
| 97046.064 | running | episode 0 starts |
| 97049.643 | holding | 3.58 s later: the waypoints ran out (manager frozen; `stall.log`: STOP at 01:46:06.276, CONT at 01:46:07.780 wall clock, 1.50 s) |
| 97050.412 | stopped | 0.77 s after `holding`, no error text; the driver records `stopped` (duration 4.36 s) |
| 97060.033 | homing | episode 1, same sim PC (no restart) |
| 97062.482 | running | episode 1 runs to its 15 s timeout |

### P3 `p` refusal (raw: `$S/p34/refusal_logs/sim_pc.log`, copied as [data/g5/p34_refusal_sim_pc.log](data/g5/p34_refusal_sim_pc.log); not in sim_evals)

```
09:45:20,282 key p: start pose: 15.4 s joint move to homing.q_start_rad
09:45:35,661 start pose reached (fingers at 0.080 m): start the episode
09:45:35,767 key p refused: not at home: press h first
```

## Analysis

- **Hold still: pass.** Drift <= 0.040 mm / 0.090 deg from the deploy home and <= 0.069 mm / 0.122 deg from the
  trainstart and start poses over 20 s, the hand stays at 0.0800 m, and the gate never shortens or rejects a
  hold_still waypoint. The larger drift away from home is consistent with the impedance controller holding a
  stretched-out pose under gravity, and it is far below anything the gate or the judge reacts to.
- **Out-of-box fault: pass**, and latched as on the robot: after a fault the manager cannot start again ("FATAL
  manager: the PC reports fault ... the PC holds; Ctrl-C and restart control_deploy run on the PC"). Run 0141 shows the
  bug E7 then fixed (a failed start was waited on for 180 s and the second episode became `not_run`); run 0143_43
  shows the fix: run.sh restarts the sim PC and Pi and resumes the folder. The fault came from a replay whose pinch
  goes below the then box floor z 0.05 (the TCP sits at the table when the fingers pinch a 12 mm eraser); this is the
  observation that led to E7 fix 12 (sim box floor 0.045, then -0.005 with the table at z 0).
- **Watchdog: pass.** With the manager frozen 1.5 s, the PC went `holding` and then `stopped` 0.77 s later. With the
  1.0 s limit this places the last C3 about 0.23 s before `holding` (inferred: the C3 times themselves were not
  recorded). The next episode homed and ran in the same sim PC, so `stopped` is not latched (unlike `fault`).
- **Homing: pass.** Every episode in these evals homed (`h`; deploy and trainstart in E7; `h` 2.06 s then `p`
  15.45 s in P3) and the start pose was reached to 0.000114 rad (tolerance 0.05 rad).
- **P3 refusals: pass.** `c` away from the start pose -> `start_failed` with the PC's text; `p` away from home ->
  "key p refused: not at home: press h first". The `p` evidence is only in the scratchpad log, not in sim_evals.

### Docs vs raw data

1. PLAN §6 G5 (E7) "hold_still 0.04 mm / 0.09° (deploy), 0.07 mm / 0.12° (trainstart) over 20 s": raw 0.036-0.040 mm /
   0.089-0.090 deg and 0.068-0.069 mm / 0.121-0.122 deg (`$S/e7/g5_hold_c2.jsonl`, `g5_hold_ts_c2.jsonl`): agrees;
   but the evidence is not in sim_evals (only in the scratchpad), and the first E7 hold (`0130_g5_hold`) ran 15 s, not 20 s.
2. PLAN §6 P3 G5 "TCP drift 0.07 mm / 0.12 deg": raw 0.066 mm / 0.121 deg for `0943_p3_g5_hold_z0` (trace.json);
   0.070 / 0.123 is the superseded old-table run 0939. Rounds the same.
3. PLAN §6 P3 G5 "start pose reached within 0.0001 rad": raw `max_joint_err_rad` 0.000114
   (`0943_p3_g5_hold_z0/episodes.jsonl`).
4. sim-experiments/README.md "workspace box min ... x 0.20, y -0.40, z 0.045 (old table) ... used by every sim run",
   DESIGN "E7: integration" "Workspace box" / fix 12: the E7 G5 runs ran with the copy's box, min x 0.250 / z 0.050
   (raw gate texts in `0141_.../episodes.jsonl`, `0143_..._43/episodes.jsonl`).
5. PLAN §6 G5 (E7) "out-of-box waypoint -> fault": raw shows the fault was provoked by a replay that pinches below
   the box floor, not by a scripted out-of-box waypoint.
6. PLAN §6 G5 "scripted waypoints tracked within a set error": no run in sim_evals or the scratchpad tests it.
7. PLAN §6 G5 (E7) "stopped ~1 s after the last C3": raw shows `holding` -> `stopped` 0.77 s; the last C3 time is
   not recorded (`$S/e7/g5_wd_c2.jsonl`).
8. sim-experiments/README.md "Model versions ... `7746a439…` (E7, D8 tags)": every E7 eval.yaml here says
   `701815be08cd…`; `7746a439…` is only in the superseded P3 run 0939.

## Conclusion

Yes: the sim PC holds still (<= 0.07 mm, <= 0.12 deg over 20 s), latches a `fault` on an out-of-box waypoint (with
run.sh's restart and resume working), stops 0.77 s after going to `holding` when C3 stops, homes to every home and
refuses `c` and `p` where the real PC does. G5 passes in E7 and again in P3 with the table at O z 0. The tracking
part of the criterion is covered only indirectly, by G6.

## Limitations / sim-vs-real caveats

- The impedance controller acts on a MuJoCo arm with modelled masses; real drift will depend on the real gains,
  friction and payload estimate.
- The E7 evals ran with the copy's box (x 0.25, z 0.05), the P3 ones with the sim box; neither is a measured real box.
- E7's C2 recordings and the P3 refusal log live in the temporary scratchpad; only the small copies here survive a reboot.
- 0134, 0136, 0141, 0145 have no `sim_evals/_logs/` folder (they ran against a sim started with `$S/e7/sim_up.sh`,
  inferred from the missing logs and that script), so their sim_pc.log is gone (except the last session's
  `$S/e7/sim_logs/sim_pc.log`).

## Open questions

1. Tracking error under scripted waypoints (a square or a line at 0.1-0.25 m/s, compared with C2): a short scripted
   policy run would answer it; not needed so far because G6 shows the gate, not tracking, limits the path.
2. Is the real watchdog's holding-to-stopped time the same? A manager freeze on the robot (as in the method) answers it.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| eval folders (E7) | `sim_evals/_baseline/hold_still/2026-11-20_0130_g5_hold/`, `..._0134_g5_hold_deploy/`, `..._0136_g5_hold_trainstart/`, `..._0145_g5_watchdog/`; `sim_evals/_replay/PICK_TEST1/pick1-v1_ep003/2026-11-20_0141_g5_fault_trainstart/`, `.../2026-11-20_0143_g5_fault_recovery_43/` |
| eval folders (P3) | `sim_evals/_baseline/hold_still/2026-11-20_0939_p3_g5_hold/` (SUPERSEDED.txt), `..._0943_p3_g5_hold_z0/`, `..._0945_p3_g5_refuse_c_deploy/` (NOTE.txt) |
| logs | `sim_evals/_logs/2026-11-20_013018/` (0130), `2026-11-20_014251/` (a first `0143_g5_fault_recovery` folder, no longer in sim_evals or index.csv), `2026-11-20_014332/` (0143_43), `2026-11-20_093919/` (0939), `2026-11-20_094335/` (0943) |
| scratchpad (temporary) | `$S/e7/g5_hold_c2.jsonl`, `g5_hold_ts_c2.jsonl`, `g5_fault_c2.jsonl`, `g5_fault2_c2.jsonl`, `g5_wd_c2.jsonl` (8-12 MB each), `$S/e7/sim_up.sh`, `$S/e7/sim_logs/`, `$S/p34/refusal_logs/` |
| copied into this folder | `data/g5/drift.py`, `drift.csv` (computed), `c2log.py`, `c2stats.py`, `stall_manager.sh`, `stall.log`, `run_hold.log`, `run_fault.log` (from `$S/e7/`), `p34_g5.log` (`$S/p34/g5.log`), `p34_refusal_sim_pc.log` (`$S/p34/refusal_logs/sim_pc.log`) |
