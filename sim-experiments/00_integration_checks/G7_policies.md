# G7: learned policies run in the sim

**Status:** done: **fail** (E7 smoke from the deploy home, superseded; P3 from the start pose) · **Date:** 2026-11-20 ·
**Who:** agent (E7, P3), documented by agent · **Plan ref:** sim_deploy/PLAN.md §6 (G7 row, P3 table), §10;
DESIGN.md "E7: integration" (open items 2, 3), "P3/P4" -> "P3 results"

`$S` below = `/tmp/claude-1000/-home-ruben-Capstone/20deb2cf-0c82-44ce-870d-39667f0b1130/scratchpad`, the session
scratchpad of the E7 and P3 agents: **temporary, not in sim_evals**. Small files from it are copied into
[data/g7/](data/g7/).

## Question

Does each trained checkpoint (plain diffusion and behavior prompting) complete 3 episodes in the sim with no fault,
with inference inside the `exec_h` step budget and the tap images saved?

## Why it matters for the real robot

G7 is the last gate before benchmarks: it shows that a checkpoint loads in the copied `policy_deploy`, runs at a
usable rate on this laptop, and produces motions the real gate accepts. A policy that faults in the sim from a
training-like start pose would most likely fault on the Franka too, unless the cause is a sim-only difference
(the rendered images), which is what P3 found and the camera study ([02_camera_study](../02_camera_study/README.md))
now tests.

## Setup

| Item | Value |
|---|---|
| Repo / branch | `~/Capstone-main-sim`, `rnoro/main-sim` @ `3633fcb` (eval.yaml `git.sim_deploy` `3633fcb3…-dirty`, `git.training` `1e4dc645…`; + uncommitted `sim_deploy/`, `deployment/`) |
| Model | E7 smoke: `701815be08cd…` (old table top O z 0.05). P3: `166a0649571a…` (table top O z 0) |
| Home / start pose | E7: `--home deploy`, TCP O (0.3069, 0.0000, 0.3542), fingers straight down. P3: `--home start`, P3 stand-in q `[-0.6419329, 0.3706554, -0.1347388, -1.8275169, 0.5918946, 1.4884701, -1.6761742]`, TCP O (0.4385, -0.1641, 0.1618), 16.2 cm above the table, tilt 49.5 deg, heading left |
| Eraser placement | E7: `scene/randomization.yaml`; P3: `scene/randomization_start.yaml` (seeded) |
| Stations / overlays | E7: `stations/control_sim.yaml`, box in force min x 0.25 / z 0.05, max z 0.70 m (gate texts); P3: `control_sim.yaml` + `start_pose.yaml`, box min [0.20, -0.40, -0.005], max [0.80, 0.40, 0.70] m. Both: v_max 0.25 m/s, cone 60 + 2 deg about straight down. No overlay |
| Policy / checkpoint | `PICK_TEST1/run_006_2026-11-18`, kind **prompting** (behavior prompting), dataset pick1-v1 (30 episodes, "first umi data, 30 eps"), 60 epochs, `checkpoints/top1_epoch=0059-val_action_mse_error_all=0.0002007.ckpt`; prompt dataset `pick1-v1.zarr.zip`, `prompt_index` 0 (eval.yaml `policy`, `station_manager`) |
| Inference settings | `--precision bf16 --diffusion-steps 8 --exec-h 12` (both); `--tap` (P3 only; `tap_images_every` 10) |
| Seeds / episodes | `--seed 0`; E7 2 episodes, P3 3 episodes (episode n: `default_rng([0, n])`) |
| Timeout | E7 45 s; P3 60 s |
| Sim-only settings | box min (P3), finger contact pairs, width offset 0.0, appearance (sim-experiments/README.md table) |

## Method

Pass (PLAN §6): each checkpoint (plain and prompting) completes 3 episodes with no fault, inference within the
`exec_h` budget, tap images saved.

1. Run the checkpoint through `run.sh` (sim PC + sim Pi + the eval driver, which starts `policy_deploy run --mock`
   with the checkpoint `use`d in a frozen station copy). A latched fault makes run.sh restart the sim and resume.
2. Per episode (raw `episodes.jsonl`, `episodes/<n>/steps.jsonl`, `trace.json`): outcome, the gate's fault text and
   time, gate counts; inference `policy_ms` p50 / p95 and `margin_ms` minimum (negative = the chunk arrived after
   its first waypoint was due) from steps.jsonl; tap files. For P3 also the TCP path from trace.json: the
   displacement after 1 s and 3 s in the start TCP frame (x, y, z; z = the fingertip direction, so +z = toward the
   eraser), max TCP z and the closest TCP-to-eraser-centre distance. Script:
   [data/g7/g7_episodes.py](data/g7/g7_episodes.py) -> [data/g7/g7_episodes.csv](data/g7/g7_episodes.csv).
3. P3 offline follow-up (no sim; torch env, CPU): the first chunk of run_006 for the same low-dim input with
   (a) the sim's first images, (b) training `episode_000152`'s first images, (c, d) one camera swapped
   (`sim_evals/_start_pose_sweep/tools/offline_first_chunk.py`, `SPLIT=1`). The full write-up of that camera swap:
   [../01_start_pose/6_camera_swap.md](../01_start_pose/6_camera_swap.md).

## Commands to reproduce

Needs the GPU (VRAM 3.3 GB of 4 GB with the renderer), the sim ports free, no other sim, the laptop on AC power and
otherwise idle (P3's inference was slowed by other processes). About 1.5 min per episode.

```bash
cd ~/Capstone-main-sim && unset PYTHONPATH
# P3 (from the start pose; P4 has since changed stations/start_pose.yaml: pass --start-pose with the P3 stand-in to
# reproduce exactly)
sim_deploy/run.sh --policy /home/ruben/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18 --episodes 3 --seed 0 \
  --tag p3_g7_start --timeout 60 --precision bf16 --diffusion-steps 8 --exec-h 12 --tap
# E7 smoke (deploy home)
sim_deploy/run.sh --home deploy --policy /home/ruben/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18 --episodes 2 \
  --seed 0 --tag smoke --timeout 45 --precision bf16 --diffusion-steps 8 --exec-h 12
# offline first chunk (torch env, no sim); SPLIT=1 for the camera swap
cd ~/Capstone-main-sim/deployment/policy_deploy && \
  ~/envs/sim_policy_deploy_torch/bin/python ../../sim_evals/_start_pose_sweep/tools/offline_first_chunk.py \
  ../../sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start
# per-episode table (read-only)
~/envs/sim_deploy/bin/python -B ~/Capstone-main-sim/sim-experiments/00_integration_checks/data/g7/g7_episodes.py
```

## Results

### Per episode (raw: [data/g7/g7_episodes.csv](data/g7/g7_episodes.csv), from each eval's episodes.jsonl / steps.jsonl / trace.json)

| Eval, episode | Outcome | Fault at (s) | Fault (gate text) | Shortened / pulled back / rejected | Steps sent / skipped | Policy p50 / p95 (ms) | Margin min (ms) | Tap files |
|---|---|---|---|---|---|---|---|---|
| E7 `0149_smoke` ep 0 | fault | 7.28 | waypoint 10 of chunk 35: outside box by 0.015 m on z (z = 0.715 m > max 0.700 m) | 50 / 1 / 2 | 34 / 0 | 176.8 / 195.0 | 26.1 | 0 (no `--tap`) |
| E7 `0149_smoke` ep 1 | fault | 5.98 | waypoint 9 of chunk 28: outside box by 0.012 m on z (z = 0.712 m > max 0.700 m) | 39 / 0 / 3 | 27 / 0 | 182.2 / 197.6 | 10.7 | 0 |
| P3 `0957_p3_g7_start` ep 0 | fault | 18.45 | waypoint 11 of chunk 68: outside box by 0.011 m on y (y = -0.411 m < min -0.400 m) | 77 / 87 / 1 | 67 / 0 | 270.1 / 277.7 | -116.7 | 67 |
| P3 ep 1 | fault | 16.89 | waypoint 9 of chunk 60: tilt 62.0 deg from approach axis > 60.0 + 2.0 deg | 58 / 82 / 2 | 59 / 0 | 270.2 / 277.4 | -112.6 | 59 |
| P3 ep 2 | fault | 11.44 | waypoint 8 of chunk 52: tilt 63.0 deg from approach axis > 60.0 + 2.0 deg | 47 / 97 / 4 | 51 / 0 | 188.0 / 212.0 | -140.4 | 51 |

Eval-level (metrics.json): E7 policy_ms_p50 179.5, policy_ms_p95_max 197.6, margin_ms_min 10.7, 0 skipped; P3
policy_ms_p50 270.1, policy_ms_p95_max 277.7, margin_ms_min -140.4, margin_ms_p50 -89.6, 0 skipped. No episode
grasped or lifted. Both evals needed run.sh's restart after each fault (`data/g7/run_g7.log`, `data/g7/p34_g7.log`).
E7 GPU memory during the smoke: up to 3306 MiB (`data/g7/g7_vram.txt`, nvidia-smi samples).

### TCP motion (P3 only; E7's trace.json has no TCP column)

| P3 episode | TCP displacement after 1 s (cm, start TCP frame x / y / z) | after 3 s | Max TCP z (m) | Closest TCP to eraser centre (cm) |
|---|---|---|---|---|
| 0 | +0.8 / -1.0 / -1.5 | +1.3 / -3.0 / -4.2 | 0.601 | 23.2 (= the start distance) |
| 1 | +0.9 / -1.8 / +0.9 | +2.3 / -3.6 / +2.3 | 0.560 | 23.2 (= the start distance) |
| 2 | +1.5 / -1.4 / -1.0 | +2.1 / -5.8 / -9.7 | 0.579 | 18.5 |

The start TCP is at z 0.162 m; every episode rises by 40-44 cm before the fault.

### Offline first chunk (raw: `sim_evals/_start_pose_sweep/p3/offline.log`, `offline_split.log`; same low-dim input, width 0.080 or 0.0868 m)

| Images (IMX519 + IMX219) | Last step of the first chunk, mean over 5 samples (TCP frame, cm) z = toward the fingertips |
|---|---|
| training episode_000152 (both) | **+5.6** (with the sim width 0.080: +6.2) |
| sim (both) | **-1.3** (with the training width 0.0868: -1.3) |
| sim IMX519 + training IMX219 | **+5.4** |
| training IMX519 + sim IMX219 | **-0.5** |

Policy images at the start pose vs episode_000152:
[g7_policy_images_vs_ep152.png](../../sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start/g7_policy_images_vs_ep152.png)
(1.3 MB, in the eval folder).

## Analysis

- **E7 (deploy home): fail.** Both episodes drove up and out of the then box top (z 0.715 / 0.712 m > 0.700) within
  6-7 s. From the deploy home the IMX219 looks down at the table near the robot base, unlike any training first
  frame, so this run says little about the policy; DESIGN E7 open item 2 led to the start pose (P1-P3).
- **P3 (start pose): fail, and the cause is the images.** From a training-like start pose (16.2 cm, 49.5 deg) all
  three episodes go up (to z 0.56-0.60 m) and sideways instead of toward the eraser, which is never closer than 18.5
  cm, and fault on the box (y) or the cone (62-63 deg > 60 + 2) after 11-18 s. Offline, the same model with the same
  low-dim input moves toward the fingertips (+5.6 cm) on the training images and backs away (-1.3 cm) on the sim's;
  swapping only the IMX219 image flips the sign (+5.4 / -0.5 cm), the width does not matter. So the policy rejects
  the sim's IMX219 render. The side-by-side image shows the geometry matches (eraser, fingers and tags where
  training has them) and the appearance does not (table texture, hard shadows, colours, the IMX219's empty table
  and grey sky vs the lab).
- **Inference.** E7: 177-182 ms p50, margin >= 10.7 ms, 0 skipped: inside the budget. P3: 270 ms p50 in episodes 0-1
  and negative margins (down to -140 ms), 0 skipped: outside the budget, because other programs (test suites) loaded
  the laptop at the time (DESIGN "P3 results"; the load itself is not in sim_evals). The PC spent many samples in
  `holding` (518 and 545 in episodes 0 and 1) waiting for late chunks. This did not cause the direction of motion
  (the offline result is timing-free), but it means P3's inference numbers are not a valid measurement.
- **Coverage.** Only run_006 (prompting) was run. No plain-diffusion checkpoint was tried, so "each checkpoint
  (plain and prompting)" is not met. Checkpoint folders that exist (`/home/ruben/Capstone/data/PICK_TEST1/policy/`:
  run_002 ... run_031, non-smoke and smoke; `/home/ruben/Capstone/data/PICK_DOUBLE/policy/` run_001 ... run_005) were
  not run in G7; which of them have a fetched checkpoint is in the eval UI's catalog.
- **Tap images.** Saved in P3 (one `chunk_<n>.npz` per step, 67 / 59 / 51); E7 ran without `--tap`.

### Docs vs raw data

1. PLAN §6 G7 (E7) "both faulted after 6-7 s (TCP up to z 0.71 > box 0.70)": raw durations 7.28 / 5.98 s; the 0.715 /
   0.712 m are rejected *waypoint* z values, not reached TCP z (`0149_smoke/episodes.jsonl`). E7's trace.json has no
   TCP column, so the reached height is not in sim_evals.
2. PLAN §6 G7 (E7) "tap images saved" is part of the criterion: the smoke ran without `--tap` (eval.yaml `command`),
   no tap files.
3. PLAN §6 G7 (E7) "VRAM 3.3 GB of 4 GB": raw maximum 3306 MiB in `$S/e7/g7_vram.txt` (not in sim_evals).
4. PLAN §6 P3 G7 "fault after ... box y -0.411 at 18.4 s, cone 62.0 deg at 16.9 s, cone 63.0 deg at 11.4 s", "policy
   270 / 270 / 188 ms p50 (p95 <= 278), margin min -140 ms": raw 18.45 / 16.89 / 11.44 s, 270.1 / 270.2 / 188.0 ms,
   p95 277.7 / 277.4 / 212.0 ms, margin min -140.4 ms: agrees.
5. PLAN §6 P3 G7 "demos +5.7 cm in 0.75 s": no raw file found in sim_evals or the scratchpad (from DESIGN, not in sim_evals).
6. The offline logs use `prompt ('PICK_TEST1', 14)` while the eval ran with `station_manager.prompt_index` 0
   (`offline.log` vs `0957_p3_g7_start/eval.yaml`): the offline numbers are for prompt 14 (not stated in PLAN/DESIGN).
7. sim-experiments/README.md model versions "`7746a439…` (E7, D8 tags)": the E7 smoke's eval.yaml says `701815be08cd…`.
8. sim-experiments/README.md / DESIGN E7 "workspace box min x 0.20 ... z 0.045 (old table)": the E7 smoke ran with
   the copy's box (max z 0.700 m shown, min x 0.250 / z 0.050 in the same session's other faults).

## Conclusion

No: run_006 (behavior prompting) faults in every sim episode, from the deploy home (E7) and from the training-like
start pose (P3), and no plain-diffusion checkpoint has been tried. The P3 diagnosis is that the policy rejects the
sim's IMX219 images (appearance, not geometry); this is the open item that blocks any benchmark, and it is what the
camera study ([02_camera_study](../02_camera_study/README.md), PLAN §10) works on. Inference fits the budget when the
laptop is otherwise idle (E7: 177-182 ms p50, margin >= 10.7 ms).

## Limitations / sim-vs-real caveats

- One checkpoint, 2 + 3 episodes, one seed.
- P3's inference ran on a loaded laptop; its timings and the `holding` time they caused are not representative.
- The offline attribution uses the first chunk of one start pose and one training episode (000152), 5 samples each.
- The appearance gap is sim-only; the real robot's images come from the real lab, but the Franka mount's view of
  the fingers differs from the handheld one by a few percent (PLAN §10), which the sim also has to settle.

## Open questions

1. Does a plain-diffusion checkpoint (e.g. run_022..run_030 of PICK_TEST1) behave the same? One 3-episode eval each
   answers it once the camera study's fixes exist.
2. Which part of the IMX219 render makes the policy back away (texture, light, background)? PLAN §10 B / C1.
3. Does the first chunk depend on the prompt index (0 in the eval, 14 offline)? An offline run with index 0 answers it.

## Raw data

| What | Path (relative to `~/Capstone-main-sim/`) |
|---|---|
| eval folders | `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0149_smoke/` (E7), `sim_evals/PICK_TEST1/run_006_2026-11-18/2026-11-20_0957_p3_g7_start/` (P3, with `g7_policy_images_vs_ep152.png` and `episodes/<n>/tap/`) |
| logs | `sim_evals/_logs/2026-11-20_014949/` (E7), `sim_evals/_logs/2026-11-20_095724/` (P3; one log folder per eval, the restarts append to it) |
| offline | `sim_evals/_start_pose_sweep/p3/offline.log`, `offline_split.log`, `ep152_ref.npz`, `g7_policy_images_vs_ep152.png`; tools `sim_evals/_start_pose_sweep/tools/offline_first_chunk.py`, `g7_images.py` |
| checkpoint | `/home/ruben/Capstone/data/PICK_TEST1/policy/run_006_2026-11-18/checkpoints/top1_epoch=0059-val_action_mse_error_all=0.0002007.ckpt` |
| scratchpad (temporary) | `$S/e7/run_g7.log`, `$S/e7/g7_vram.txt`, `$S/p34/g7.log`, `$S/p34/offline*.log` |
| copied into this folder | `data/g7/run_g7.log`, `g7_vram.txt` (from `$S/e7/`), `p34_g7.log` (`$S/p34/g7.log`), `g7_episodes.py` + `g7_episodes.csv` (computed here) |
