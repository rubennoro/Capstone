# umi-processing: TODO

Open work, in the order it unblocks things. Background: [PROBLEM.md](PROBLEM.md). Status by step: [PLAN.md](PLAN.md).
How to do each step: [DIRECTIONS.md](DIRECTIONS.md). Tick items off here as they land.

## 1. Now: bring the Jetson up and get a passing lens (you)

- [ ] **Pull on the Jetson:** `git fetch && git checkout rnoro/data-work && git pull` (DIRECTIONS "Start here", 1.1).
- [ ] **Python environment** (DIRECTIONS 1.2): `python3-venv`, venv, `pip install -e umi-processing[test]`; the install
      check must pass.
- [ ] **SLAM build on the Jetson** (1.3, in `tmux`, about an hour). First run of `native/build.sh` on aarch64 /
      JetPack: report anything that fails (it builds its own OpenCV 4.5.4). Then the slow settings check (end of 1.3).
- [ ] **Lens calibration that passes** (§2). What was missing in tries 122–127:
  - a **flat board**: taped to cardboard, or the board shown full-screen on a monitor or tablet;
  - the board **right out at the image's left and right edges, both top corners and the lower left** (almost beside
    the lens: see the table in §2.2);
  - holds at about 20, 30 and 45 cm, plus a few tilted ones.
- [ ] `umiproc lens-calibrate … --apply`, then `umiproc status` shows the lens as calibrated (1.5).
- [ ] **Pink colour cast** on the IMX219 images: check the Pi's camera tuning file / IR filter. It doesn't affect SLAM
      or the lens calibration, but it does affect training images.

## 2. Next: a map that passes (you + umiproc map)

- [ ] Re-run clip 128 with the real lens: `umiproc map eraser_pick episode_000128` (it can be re-run; a failed map
      records nothing).
- [ ] Record a new mapping clip if needed (DIRECTIONS 4.1): table tag **flat** on the
      table, calm start above it, slow continuous scan, **no fast swings toward monitors or blank areas** (clip 128 lost
      track there at 32 s, which split it into several maps and fails M1).
- [ ] Map passes M1–M5. Remove the table tag before the demos.
- [ ] **Jetson speed:** read the `pacing` lines of the map report. If the Jetson tracks slower than 30 frames/s
      (the laptop managed about 20–23), decide what to do (max power mode `sudo nvpmodel -m 0`, fewer ORB features, …).
- [ ] **Comparisons on the first passing mapping clip** (DIRECTIONS Appendix A, one data root each): IMU noise 1× vs 10×;
      patches 0001–0012 vs without 0007–0009; record the outcome and the chosen defaults in docs/DESIGN.md.

## 3. Build: step 3, `calibrate` and `run`

- [x] `umiproc calibrate TASK_FOLDER EPISODE`: the gripper opening range (the copied method + UMI's rule that both
      finger tags are seen in ≥ 80 % of frames). Resolve the 3 inherited failing tests in
      `tests/dp/calibrate/test_calibrate_task.py` (docs/DESIGN.md "Changes to the copy").
- [x] `umiproc run TASK_FOLDER START END`: localize each demo in the map (`--max_lost_frames 30`), UMI's lost-frame
      rules at 30 fps (> 5 lost → drop, < 30 tracked → drop, split at gaps, pieces < 12 frames dropped), relocalized
      within 0.5 s, the copied pose checks, finger widths, `processed/episode.zarr`; "waiting for gripper calibration"
      when `calibrate` hasn't run (exit 3).
- [x] Wire the **effective rig** (with the lens file) into the copied stages that still load the rig file directly:
      `umiproc/dp/process.py` (`load_folder_context`), `umiproc/dp/calibrate/task.py`, `umiproc/dp/run/range.py`.
      Otherwise tags, widths and training images would use the nominal lens.
- [x] DIRECTIONS: prove the demos, gripper calibration and finishing; reordered 2026-11-17 into verify the map with test demos, gripper calibration, the rest of the demos (now §5, §6, §7).
- [x] Reviewer pass, fixes, commit.

## 4. Then: smoke test and the first real session end to end

- [ ] Step 4: run the whole session (map → calibrate → run) on real recordings, fix what breaks, final review.
- [ ] Check that demos relocalize in the map **without** the table tag in the scene (UMI removes it after mapping).
- [ ] The patch comparison is most telling on the demos (patches 0007–0009 only act after a later relocalization):
      localize the session's demos against each variant's map.

## Decisions still open

- [ ] **Provisional lens mode** (`lens-calibrate --apply --provisional`): install a fit that failed some checks,
      clearly labelled, everything made with it marked out of date once a real calibration passes. Only if the lens keeps
      blocking progress.
- [ ] **`lens-board --screen`**: a full-screen board image for a monitor or tablet (a perfectly flat board for free).
- [ ] **Pooling several board recordings** in one `lens-calibrate` (valid if focus and camera mode didn't change).
- [ ] **`build.sh` A/B variant**: today `PATCHES_UPTO=0006` also drops the crash fixes 0010–0011, so that variant can
      crash when a second map starts. Add a way to skip only 0007–0009 (e.g. `PATCHES_SKIP=0007,0008,0009`).
- [ ] **Split pieces at finger-width gaps like UMI** (`06:669-684` interpolates the widths and warns below 0.9) instead
      of keeping the longest usable stretch and dropping below 80 % (what `run` does now, with the trimmed part
      reported in `piece.json` / `reason.json`).
- [ ] **RECENTLY_LOST frames** (IMU-only predictions) count as lost in umiproc but as tracked in UMI: this keeps the
      > 5-lost rule and the splitting stricter than UMI's. Keep, or match UMI exactly?

## Later

- [x] Packaging episodes for training: `umiproc package DATASET FOLDER... [--episodes N]` (the copied
      `umiproc/dp/package`, wired to a command).
- [ ] Camera–IMU calibration and time offset with a board (OpenICC / Kalibr) instead of CAD and 0.
- [ ] IMU random walks from a long still recording (Allan variance), replacing the assumed values.
- [ ] Deployment reads the same rig and lens files, so the robot sees images undistorted the same way as training.

## Small cleanups

- [x] `umiproc/status.py`: an f-string without placeholders (pyflakes).
- [x] `tests/test_umiproc_map_command.py`: unused `numpy` import.
- [ ] Delete the experiment root `~/data/umi_ab/lens126/` (on the laptop) once a real lens is installed: its lens file
      is a failed calibration labelled "EXPERIMENT ONLY".
- [x] Repo-level: update `AGENTS.md` ("Current state" / "Next time") for umi-processing, and decide what to do with
      `HANDOFF_SLAM_MASK.md` (the parked dataproc mask work).
