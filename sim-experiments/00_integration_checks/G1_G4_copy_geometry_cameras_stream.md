# G1-G4: copy faithful, geometry, cameras vs reality, stream and clocks

**Status:** done: G1, G2, G4 pass; G3 pass with an explained residual (all E7; not re-run after P3/P4, see Analysis) ·
**Date:** 2026-11-20 · **Who:** agent (E2, E3, E5, E7), documented by agent · **Plan ref:** sim_deploy/PLAN.md §6
(G1-G4 rows), §2; DESIGN.md "E2: as built", "E3: as built", "E5: as built", "E7: integration" ("D8 settled")

`$S` below = `/tmp/claude-1000/-home-ruben-Capstone/20deb2cf-0c82-44ce-870d-39667f0b1130/scratchpad`, the session
scratchpad of the E1-E7 and P1-P4 agents: **temporary, not in sim_evals**. G1-G4 were checked **outside
`sim_evals/`**: none of them has an eval folder. The small evidence files are copied into
[data/g1_g4/](data/g1_g4/); the rest is linked by its scratchpad path.

## Question

Before any closed-loop run: (G1) is `deployment/` a faithful copy of `rnoro/final-deployment@ea9f7fb` with only the
listed sim changes, and does `policy_deploy check --mock` pass against the sim? (G2) Do the rig, cameras and TCP in
the model match the STEP and the 11-17 numbers? (G3) Rendered in the handheld pose, do finger and tag pixels match a
real training frame after the copy's unwarp? (G4) Does the sim Pi stream like the real Pi (30 packets/s, clocks,
pair rule) and do the policy's images look like training's?

## Why it matters for the real robot

G1 makes sim results transferable: the laptop code that runs in the sim is the code that runs on the robot, apart from
a listed set of sim-only changes. G2/G3 decide whether the policy sees the fingers and the eraser where training put
them; G3's residual (training's `T_cam_tcp` vs the image evidence) is a real-rig finding, not a sim one. G4 exercises
the copy's frames process (clock map, pair rule, staleness) with the real Pi's wire format.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (+ uncommitted `sim_deploy/`, `deployment/`); upstream `~/Capstone-final-deployment` @ `ea9f7fb` (the copy's source) |
| Model | E7 checks ran 01:20-01:37 on 2026-11-20 with the model the E7 evals of that hour record, `701815be08cd…` (old table top O z 0.05, D8 tags, appearance). G2 compared here: wave-2 model `9c75edf8d681…` (`$S/e7/backup/model_before.json`, built 2026-11-20T05:49:54Z) vs the current `166a0649571a…` (`sim_deploy/build/sim_rig/model.json`, built 14:40:51Z, table at z 0) |
| Homes | G1/G4: `--home deploy` (TCP O (0.307, 0.000, 0.354)) and `--home trainstart` (G4 images); G3: the handheld pose (no arm) |
| Stations | `sim_deploy/stations/policy_sim.yaml` (laptop, under `--mock`; for G1 a copy `$S/e7/g1/policy_sim_check.yaml`), `control_sim.yaml` (sim PC), `sim_rig.yaml` (model, sim Pi) |
| Policy | `hold_still` (G1 check, G4 images) |
| Reference data (G3) | PICK_TEST1 `pick1-v1` index 14 = `pi-2026-11-18_episode_000152`, 64 frames with both tags in training's 224 px IMX219 image |
| Sim-only settings | the sim stations (127.0.0.1 addresses, `launcher.fake_pc/fake_pi: false`), finger contact pairs, appearance (sim-experiments/README.md) |

## Method

| Check | Pass when (PLAN §6) | Tool, what it measures |
|---|---|---|
| G1 | the diff report shows only the listed changes; `policy_deploy check --mock` passes against the sim | file-by-file compare of `deployment/` with `git ls-tree ea9f7fb deployment` (byte compare per file, every changed file carries a header naming its change); `policy_deploy check --mock` in a pty with the sim running (every row PASS) |
| G2 | rig poses match the STEP; 11-17 numbers reproduced (±1 mm); the rendered views show the rig parts where the STEP puts them | E2/E3 smoke scripts (renders, ray tests, `F_T_TCP` from the kit) during the build; at E7 a compare of `model.json` with the wave-2 model (masses, cameras, TCP); here the same compare against the current model |
| G3 | handheld pose with the real fingers: finger and tag pixel positions after the unwarp match a real training frame within a few px; straight edges stay straight | `python -m sim_deploy.sim_pi.handheld_check`: renders the handheld gripper (kit `iphumi_umi_soft_v1`, real ArUco ids 0/1, training's `T_cam_tcp`) through the sim Pi chain + JPEG + the copy's `ImagePrep`, detects the tags in sim and real 224 px images, corner errors in px; variants E7 (gap = width, real stickers), E7+camera (camera shifted by the fit), D8_old (wave 2's width mapping); fits of camera pose / tag distance / tag size |
| G4 | 30 packets/s, no stale or down camera, the clock map converges, the pair rule passes, `policy_deploy.frames.probe --save` images look like training's | `policy_deploy.frames.pi_collect` (packets, gaps, pair gap from `t_mid_ns`), the copy's `policy_deploy.frames` stats lines (fps, age, clock offset / drift, refused old/new/reused), `frames.probe --save` at both homes |

## Commands to reproduce

All need the sim chain (sim PC + sim Pi) running except G3 and the G1 file compare; do not run them alongside
another sim (ports 5555, 5610, 5620, 5630, 5640, 5650, 5651). Under a minute each.

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
# G1 file compare (read-only; M = modified, D = missing, A = only in the copy)
cd ~/Capstone-final-deployment && git ls-tree -r --name-only ea9f7fb deployment | while read f; do r=${f#deployment/}; \
  [ -f ~/Capstone-main-sim/deployment/$r ] && { git show ea9f7fb:"$f" | cmp -s - ~/Capstone-main-sim/deployment/$r || echo "M $r"; } || echo "D $r"; done
# G1 check (sim running, e.g. a hold_still eval in another terminal or the E7 sim_up.sh; needs a terminal or pty)
cd ~/Capstone-main-sim/deployment/policy_deploy && ~/envs/sim_policy_deploy/bin/policy_deploy check --mock --station ../../sim_deploy/stations/policy_sim.yaml
# G3 (no sim chain; renders offscreen with MuJoCo, exports the reference once with the torch env)
cd ~/Capstone-main-sim && ~/envs/sim_deploy/bin/python -m sim_deploy.sim_pi.handheld_check --out /tmp/g3 --index 14
# G4 (sim Pi streaming): frames alone, then the probe and the packet collector (sim_deploy/sim_pi/README.md)
cd ~/Capstone-main-sim/deployment/policy_deploy
~/envs/sim_policy_deploy/bin/python -m policy_deploy.frames --mock --station ../../sim_deploy/stations/policy_sim.yaml &
~/envs/sim_policy_deploy/bin/python -m policy_deploy.frames.probe --station ../../sim_deploy/stations/policy_sim.yaml --mock --save /tmp/probe
~/envs/sim_policy_deploy/bin/python -m policy_deploy.frames.pi_collect --endpoint tcp://127.0.0.1:5555 --seconds 30
```

## Results

### G1: copy faithful

| Copy | Compared with | Files differing | Source |
|---|---|---|---|
| at E7 (snapshot taken before P1, `$S/p1/copy_before/`) | `ea9f7fb:deployment` | **18**: 17 modified + 1 new (`policy/replay_policy.py`); plus build artefacts (`*.egg-info`) | [data/g1_g4/g1_copy_e7_vs_ea9f7fb.txt](data/g1_g4/g1_copy_e7_vs_ea9f7fb.txt) |
| now (2026-11-20 evening) | `ea9f7fb:deployment` | **61**: 60 modified + 1 new (E1/E4/E6 changes + P1's port of the start-pose snapshot) | [data/g1_g4/g1_copy_now_vs_ea9f7fb.txt](data/g1_g4/g1_copy_now_vs_ea9f7fb.txt) |
| `$S/e7/g1/deployment/` | `ea9f7fb:deployment` | 0 (it is a pristine extract of ea9f7fb, the reference E7 diffed against) | - |

The 18 at E7 split as DESIGN says: E1 12 (`policy_deploy/STRUCTURE.md`, `config/launcher.yaml`, `config/policy.yaml`,
`frames/app.py`, `frames/settings.py`, `launcher/launcher.py`, `launcher/processes.py`, `launcher/use_policy.py`,
`policy/policy_factory.py`, `policy/self_check.py`, `policy/settings.py`, new `policy/replay_policy.py`), E4 5
(`franka_control/.gitignore`, `CMakeLists.txt`, `include/franka_control/command_line.h`, `src/app.cpp`,
`src/command_line.cpp`), E6 1 (`contracts/frames.yaml`). In the current copy every one of the 61 files has a header
comment naming its change in its first 12 lines (checked by a keyword grep, not read one by one).

`policy_deploy check --mock` (2026-11-20 01:32, pty, sim running; [data/g1_g4/g1_check_pty.log](data/g1_g4/g1_check_pty.log)):
21 PASS rows, 5 INFO rows, 0 WARN / FAIL, "policy_deploy check: ready (0 warning(s))", exit 0. Rows of note:
`frames Pi reachable PASS TCP connection to 127.0.0.1:5555 opened`, `manager terminal PASS keyboard on this terminal`,
`manager worst gap PASS ... 0.57 s < watchdog_limit_s 1 s`, `contracts/frames.yaml PASS F_T_TCP loaded, settled: false`,
`manager PC INFO PC on 127.0.0.1 started outside the launcher` (under `--mock` the PC is not probed).

### G2: geometry (wave-2 model `9c75edf8` vs current `166a0649`, `model.json`)

| Item | Wave 2 (`9c75edf8…`) | Current (`166a0649…`) |
|---|---|---|
| `F_T_TCP` | +45 deg about flange z, z 0.23565 m | same; `F_T_TCP_vs_kit` 0.0021 mm / 4.5e-6 deg |
| link0-7, hand, rig, finger, eraser masses (kg) | hand 0.73, rig 0.39066, fingers 0.137788 each, eraser 0.03, robot 18.088143 | identical (difference 0.0); hand subtree 1.396236 |
| `cam_imx219`, `cam_imx519` poses at home and K | IMX219 K (246.65, 246.65, 320.16, 239.54); IMX519 K (492.50, 492.50, 319.5, 239.5) | identical |
| `finger_kit.tag_separation_closed_m` | 0.040577 (CAD sticker outlines) | **0.044287** (E7 D8: real stickers 1.855 mm further out = training `min_width` 0.04429) |
| finger kit sha | `ee6328eb…` | `88fc5453…` (kit.yaml + SOURCE.md only, STLs unchanged) |
| table | top z 0.05 m, x 0.16..0.96, y ±0.5, pedestal | top z 0.0 m, x -0.40..1.00, y ±0.80, no pedestal |
| new in current | - | `appearance` (pads, holders, ArUco tag textures, wood table), `finger_contact` pairs (margin 3.5 mm, solref [-200, -28]) |

The 11-17 numbers (±1 mm) and the renders were checked in E2/E3 (DESIGN "E2: as built", "E3: as built", renders in
`$S/e2`, `$S/e3`; from DESIGN, not in sim_evals) and not re-run since; the compare above shows nothing they cover changed.

### G3: cameras vs reality (raw: `$S/e7/g3/g3_report.json`, summary [data/g1_g4/g3_summary.csv](data/g1_g4/g3_summary.csv))

64 frames of episode_000152, both tags detected in all 64 sim frames for every variant. Corner errors in the 224 px
IMX219 image:

| Variant | Tag 0 corners mean / max (px) | Tag 1 corners mean / max (px) | Tag size sim / real |
|---|---|---|---|
| E7 (gap = width, real stickers, camera = training `T_cam_tcp`) | 4.97 / 8.20 | 12.03 / 17.71 | 0.865 / 0.825 |
| E7+camera (camera moved by the fit) | 1.21 / 3.09 | 0.90 / 1.95 | 0.989 / 0.967 |
| D8_old (wave 2: width = gap + 44.3 mm) | 24.54 / 30.75 | 34.67 / 42.22 | 0.828 / 0.779 |

| Fit (both tags) | Camera shift (mm, IMX219 axes) | Rotation (deg) | Mean / max (px) |
|---|---|---|---|
| nominal (no fit) | 0, 0, 0 | 0 | 8.40 / 17.18 |
| camera translation | -4.0, +2.5, +8.8 | 0 | **1.01 / 3.30** |
| distance + camera pose | +2.0, -3.8, +10.1 | -3.1, -1.93, +1.85 | 0.75 / 2.96 |
| + tag size (16.16 mm fitted) | +2.0, -4.0, +9.4 | -2.9, -1.93, +1.85 | 0.75 / 3.00 |

Camera check: training's IMX519-from-IMX219 translation (-0.05, -26.29, 2.86) mm vs E2's CAD (0.2, -28.18, -3.29) mm:
6.44 mm apart, rotation 0.0033 deg. Images: `$S/e7/g3/g3_<variant>_frame{000,026,053}.png` (0.7-0.8 MB each, not copied).

### G4: stream and clocks (raw: [data/g1_g4/g4_pi_collect.log](data/g1_g4/g4_pi_collect.log), [data/g1_g4/g4_frames.log](data/g1_g4/g4_frames.log), 2026-11-20 01:32-01:33, deploy home)

| Measure | Value |
|---|---|
| `pi_collect` 30 s | 900 packets, 29.9-30.1 pkt/s, 0 bad, 0 packet_no gaps, 0 frame_no gaps, 0 `uw_reused`; main 640x480 ~30 kB, ultrawide 640x480 ~37 kB |
| pair gap (main - ultrawide mid-exposure) | min / median / max -16.3 / -16.3 / -16.3 ms (all 900 in one 5 ms bin); longest same-pair run 1 |
| frames: clock | "clock synced: offset 999.999380 s" at start (sim Pi clock = monotonic + 1000 s + 20 ppm); drift 0.0 -> 20.2 -> 20.1 -> 22.5 ppm in the 10 s stats lines |
| frames: cameras | main and ultrawide `ok` 30.0 fps, age 58-94 ms, drop 0, unusable 0, unpaired 0, packets lost 0, latency 59-60 ms (one 99 ms max) |
| frames: pair rule | refused old / new / reused 0 / 0 / 0 |
| policy images | the four 224 px images at the deploy home and at trainstart: [data/g1_g4/g4_policy_images_both.png](data/g1_g4/g4_policy_images_both.png) (from `$S/e7/g4/policy_images_both.png`; per-home PNGs in `$S/e7/g4/policy_images_{deploy,trainstart}/`) |

![Policy images at both homes (E7)](data/g1_g4/g4_policy_images_both.png)

## Analysis

- **G1 pass.** At E7 the copy differed from ea9f7fb in exactly the 18 listed files and the check passed with 0
  warnings against the running sim. Since then P1 ported the real deployment's start-pose snapshot into the copy
  (61 files now). That is a listed change, but the PLAN §6 G1 text describes the E7 state only. PLAN §11 notes the
  copy is now behind upstream `f632576` (gripper max 0.04 m + `policy/gripper_map.py`, settled TCP, real addresses,
  `q_start_rad` in station.yaml, manager / pose_converter / diffusion_policy changes); W0 re-syncs it, after which
  G1 must be redone.
- **G2 pass.** Geometry, masses and cameras are byte-identical between the wave-2 and the current model; only the
  tag positions (D8), appearance, contact and the table changed. The STEP-level checks themselves are E2/E3's and are
  documented only in DESIGN and the scratchpad.
- **G3 pass with an explained residual.** With training's `T_cam_tcp` the tags land 5.0 px (tag 0) and 12.0 px (tag 1)
  off on average and look 13-17 % too small; moving the camera (-4.0, +2.5, +8.8) mm brings both to about 1 px. The
  sim renders what the CAD says; the real handheld camera sits ~9 mm closer to the fingers along its optical axis than
  `rig_pi_v3`'s chain says (E5's finding, open item 4 in DESIGN E7). Training files are not changed (the user's call).
  The wave-2 width mapping (D8_old, 25-35 px) is ruled out.
- **G4 pass.** The sim Pi streams at 30.0 pkt/s with no gaps, the copy's frames process syncs its clock at once and
  tracks the 20 ppm drift, the pair rule refuses nothing, both cameras are `ok`. The pair gap is -16.3 ms
  (ultrawide exposed half a frame after the main), as the sim Pi is built to do.
- **After P3/P4 (table at z 0, model `166a0649`).** G1-G4 were not re-run as such. The P3/P4 sim runs exercised G1/G4
  implicitly (every eval starts `policy_deploy` against the sim and reaches `[idle]`), and G2's compare above covers
  the model. G3 does not depend on the table (handheld pose). Whether the stream numbers hold with the current model
  was not re-measured.

### Docs vs raw data

1. PLAN §6 G2 "one sha (`7746a439…` now) checked by sim PC, sim Pi and driver": the E7 runs of that hour record
   `701815be08cd…` (`sim_evals/_baseline/hold_still/2026-11-20_0130_g5_hold/eval.yaml` `model.sha256`,
   `$S/e7/sim_logs/sim_pc.log`, `$S/e7/run_hold.log` "sim PC up: model 701815be08cd"). `7746a439…` first appears at
   09:18 in `$S/p1/fc_*.log` (P1) and is used only by P2 dry runs and the superseded `0939_p3_g5_hold`. When it was
   built (between 01:53 and 09:18) is not recorded.
2. PLAN §6 G3 "with training's `T_cam_tcp` 5.0 / 12.0 px": these are the two tags' **means** (tag 0 4.97, tag 1 12.03);
   the maxima are 8.20 / 17.71 px, and the both-tag nominal is 8.40 mean / 17.18 max (`$S/e7/g3/g3_report.json`).
   PLAN writes "1.0 / 3.3 px (mean / max)" for the fit in the same row, so the reader takes 5.0 / 12.0 as mean / max.
3. PLAN §6 G4 "drift 20.1 ppm fitted vs 20 set": raw stats lines show 0.0, 20.2, 20.1 and, at shutdown, 22.5 ppm
   (`$S/e7/g4/frames.log`).
4. PLAN §6 G1 "the 18 listed files": true for the E7 copy (`$S/p1/copy_before/`); the current copy differs in 61
   files (P1's port; `data/g1_g4/g1_copy_now_vs_ea9f7fb.txt`).
5. sim-experiments/README.md "Upstream final-deployment still says 150 mm" (TCP) and the "workspace box min ... x
   0.25, z 0.05 (upstream)" row: upstream has F_T_TCP z 0.23565 m since `927a5ba` (`settled: true` at `f632576`) and
   box min [0.20, -0.40, -0.005] at `447910b` (read with `git show`); 150 mm and x 0.25 / z 0.05 are only in ea9f7fb.
6. DESIGN "E5: as built" G3 numbers (1.16 / 3.0, 1.07 / 2.75 px with a fitted +49.6 mm distance) are E5's earlier run;
   E7's G3 (above) supersedes them with D8 revised; both are scratchpad-only.

## Conclusion

The copy was faithful at E7 (18 listed files, check passes), the model's geometry, masses and cameras are unchanged
from wave 2 to the current model, the sim cameras match training frames to about 1 px once the real camera's ~9 mm
offset is applied (5-12 px with training's own `T_cam_tcp`), and the stream behaves like the real Pi's. G1-G4 pass;
the plan (§11) treats them as passed. G1 and the stream must be redone after W0's re-sync.

## Limitations / sim-vs-real caveats

- The evidence is in the temporary scratchpad; the copies here are the only durable record.
- G3 compares tag corners only (no edge-straightness metric was saved; the "straight edges" part rests on E5's
  visual check, from DESIGN).
- The sim Pi's timing is a model of the real Pi (fixed exposure, no auto exposure, no network jitter).
- G4's images "look like training's" is a visual judgement; P3's G7 later showed the policy rejects the IMX219 image
  (appearance), which G4 did not catch.

## Open questions

1. Is the real handheld IMX219 ~9 mm closer to the fingers than `rig_pi_v3` says? A caliper measurement or a
   re-calibration of `T_cam_tcp` answers it.
2. Do the stream numbers hold with the current model (166a0649) and after W0? One `pi_collect` + frames stats run.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`, or `$S`) |
|---|---|
| eval folders | none (G1-G4 have no eval folder) |
| G1 | `$S/e7/g1/check_pty.log`, `$S/e7/g1/policy_sim_check.yaml`, `$S/e7/g1/deployment/` (= ea9f7fb), `$S/p1/copy_before/` (copy at E7), `$S/p1/files.txt` |
| G2 | `$S/e7/backup/model_before.json` (wave 2), `sim_deploy/build/sim_rig/model.json` (current), `$S/e2/`, `$S/e3/` |
| G3 | `$S/e7/g3/g3_report.json`, `$S/e7/g3/g3_*.png`, `$S/e7/g3/reference_pick1-v1.zarr_14.npz` (16 MB), `$S/e5/g3/` |
| G4 | `$S/e7/g4/frames.log`, `pi_collect.log`, `policy_images_both.png`, `policy_images_{deploy,trainstart}/`, `probe_deploy/` |
| copied into this folder | `data/g1_g4/g1_check_pty.log` (escape codes removed), `g1_policy_sim_check.yaml`, `g1_copy_e7_vs_ea9f7fb.txt`, `g1_copy_now_vs_ea9f7fb.txt` (computed here), `g3_summary.csv` (from g3_report.json), `g4_frames.log`, `g4_pi_collect.log`, `g4_policy_images_both.png` (342 KB) |
