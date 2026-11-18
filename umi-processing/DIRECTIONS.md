# umiproc: directions

How to turn the Pi rig's recordings into training episodes. Read top to bottom, copy the commands as you go.
Commands run on the Jetson unless a step says otherwise. Every command ends with a `Next:` line naming the next step. Why it works this way: [PROBLEM.md](PROBLEM.md). Real recordings that worked or not, and why: [COLLECTION.md](COLLECTION.md). Design
details: [docs/DESIGN.md](docs/DESIGN.md).

Words used below:

- **SLAM**: software (here ORB-SLAM3) that works out, from the video and the IMU (motion sensor), where the camera was
  in every frame and a 3-D **map** of the scene.
- **Tracking**: SLAM knows where the camera is in that frame. **Lost**: it does not.
- **Localize**: find where a new video is inside a saved map.
- **Lens model**: the numbers that say how the fisheye lens bends the image. SLAM needs them to be right.

## 0. Start here

| Step | When | Time | Section |
|---|---|---|---|
| Setup: code, Python, SLAM build, check | once per Jetson | ~1.5 h (mostly the build) | [1](#1-setup-once-per-jetson) |
| Lens calibration | once per lens; again only if the lens moves | 1 min recording + 2 min processing | [2](#2-lens-calibration) |
| **Per session:** prepare, then record in this order | every session | | [3](#3-a-session-the-order) |
| 1. Map: mapping clip, `umiproc map` | start of the session | 1–2 min clip, ~3–5x that to process | [4](#4-the-map) |
| 2. Look at the map: video and 3-D page | right after the map | a few minutes | [4.6](#46-look-at-the-map) |
| 3. Test demos: 2–3 demos, `umiproc run` | before going on | ~1.5x the demos' length | [5](#5-test-demos-verify-the-map) |
| 4. Gripper calibration: clip, `umiproc calibrate` | once the map is verified | 20 s clip, ~1 min | [6](#6-gripper-calibration) |
| 5. The rest of the demos: `umiproc run` | the rest of the session | ~1.5x the demos' length | [7](#7-the-rest-of-the-demos) |
| After the session: what you have, copy it to the laptop | end of the session | | [8](#8-after-the-session) |
| A/B comparison of SLAM settings | first real session only | | [Appendix A](#appendix-a-the-ab-comparison) |

**Do the lens calibration before any map.** With the rig file's guessed lens, SLAM loses every map it starts. With a
measured lens, the same clip gives a good map.

## 1. Setup (once per Jetson)

Long commands (the build, a map) belong in `tmux`, so they keep running if ssh drops: `tmux new -s work`, run the
command, detach with `Ctrl-b` then `d`, come back with `tmux attach -t work`.

### 1.1 Get the code

```bash
cd ~/Capstone && git fetch && git checkout rnoro/data-work-auto && git pull
git log --oneline -1
```

`~/Capstone` is the Jetson's clone of the repo. The last line shows the newest commit.

### 1.2 Python environment (about 5 min)

```bash
sudo apt install python3-venv python3-pip
python3 -m venv ~/envs/env_umiproc
source ~/envs/env_umiproc/bin/activate
unset PYTHONPATH
pip install --upgrade pip
pip install -e ~/Capstone/umi-processing[test]
umiproc --help
```

Python must be 3.10 or newer (`python3 --version`). `umiproc --help` lists `lens-board`, `lens-calibrate`, `map`,
`calibrate`, `run`, `status`, `plot-run`, `slam-video`, `slam-3d`. Keep this environment separate from dataproc's.

**In every new terminal**, before any `umiproc` command:

```bash
source ~/envs/env_umiproc/bin/activate
unset PYTHONPATH
```

(`unset PYTHONPATH` stops ROS, if it is in your shell, from breaking the imports.)

### 1.3 Build SLAM (about 1 hour, in tmux)

```bash
sudo apt install build-essential cmake git libeigen3-dev libgl1-mesa-dev libopengl-dev libglew-dev \
    libboost-serialization-dev libssl-dev
~/Capstone/umi-processing/native/build.sh
```

- Builds UMI's ORB-SLAM3 version with our fixes (patches 0001–0012) into `~/ORB_SLAM3_umi`. The first run takes about
  an hour, mostly its own copy of OpenCV. Do the lens calibration (section 2) while it runs.
- **Worked when** it ends with `== done`. `~/ORB_SLAM3_umi/umiproc_build/BUILD_INFO` lists the patches.
- On failure it prints the end of the failing log (all logs: `~/ORB_SLAM3_umi/umiproc_build/logs/`). If the compiler
  ran out of memory, run it again with `JOBS=1` in front.
- Build it on the machine that runs it; do not copy it between machines.
- A rebuild after a patch change takes about 20–30 min. **After any rebuild, existing maps are "out of date"** and
  must be mapped again ([4.8](#48-out-of-date-and-interrupted-maps)).

Check that SLAM starts and reads umiproc's settings (about 30 s):

```bash
cd ~/Capstone/umi-processing
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_umiproc_slam_inputs.py -m slow
```

It must say `1 passed`. `skipped` means the SLAM program was not found: check the build.

### 1.4 The data root

Nothing to do. Know where things go:

- umiproc writes everything under **`~/data/umi/`** (created by the first command that needs it). Leave
  `UMIPROC_ROOT` unset.
- The Pi's recordings in **`/data/incoming`** are only read, never changed. `/data` and `~/data` themselves are
  refused as a data root (they belong to the Pi and to dataproc).
- `~/data/umi/rigs/` holds the three files the processing reads: `rig_pi_v3.yaml` (cameras, IMU, geometry),
  `rig_pi_v3.slam.yaml` (gripper mask, table tag, SLAM settings), `rig_pi_v3.lens.yaml` (from section 2). The first
  two are copied from the repo the first time. To change a setting, edit the copy there, not the repo.
- Every map and episode remembers the settings it was made with. Changing one makes them out of date.

### 1.5 Check with `umiproc status`

```bash
umiproc status
```

Shows the data root, the three rig files and the SLAM program. **Ready for a session when**:

- `lens file:` says `calibrated` with a date (after section 2);
- `umiproc_slam:` and `vocabulary:` say `found`.

`NOT FOUND`: redo the build (1.3). `refused`: `UMIPROC_ROOT` points at `/data` or `~/data`; unset it. A `PROBLEM:`
line names the file and what to change.

### 1.6 Quick tests (1–2 min)

```bash
cd ~/Capstone/umi-processing
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_umiproc_*.py tests/test_lens_*.py tests/test_slam_*.py -m "not slow"
```

Must end with `passed` and no `failed`.

> **On the laptop instead of the Jetson.** Same setup (the laptop clone is `~/Capstone-data-work`). The recordings
> are copied to `~/Downloads/data/incoming`, so add `--source ~/Downloads/data/incoming` to `lens-calibrate`, `map`,
> `calibrate` and `run`. Lens results also go to an analysis folder (section 2.3); on the laptop set it with
> `export UMIPROC_ANALYSIS=~/Downloads/data/analysis`.

## 2. Lens calibration

Measures the real lens of the ultrawide camera (IMX219, 160°). Do it once, before the first map. It needs only the
Python environment (1.2), not the SLAM build.

**You need:** a printer, something rigid and flat (foam board, clipboard, flat wood), glue or tape, a ruler, a small
piece of tape for the lens, good light.

### 2.1 Print the board

```bash
umiproc lens-board                 # ./umiproc_lens_board.pdf, US Letter
umiproc lens-board --paper a4      # A4 instead (--out FILE to name the file)
```

1. Print at **100 % / "Actual size"**, not "Fit to page". Matte paper if you have it.
2. Measure: **one square = 30.0 mm**, the scale bar = 100 mm. If not, print again. (Printed larger and evenly, e.g.
   A3 at 140 %, is also fine.)
3. **Glue it flat onto the rigid backing**, all four edges, no bubbles or curled corners. **Never use bare paper**:
   it bends, and that alone ruins the fit.

### 2.2 Record (on the Pi, about 60 s)

Before you start:

- **Focus, then lock it:** turn the lens until sharp, then tape across the lens and its holder so it cannot turn.
- **Light:** bright and even, no lamp or window reflected in the board.

Record one ordinary Pi episode. The fingers show at the bottom of the image; that is fine (that area is ignored).
Keep them still.

1. **Prop the rig on the table so it cannot move**, looking straight ahead. Start recording.
2. Hold the board by its backing, **always facing the lens**.
3. **Move about 1 s, hold still about 1 s.** Only the still moments count.
4. Visit every part of the image. The lens sees almost sideways, so the image edges are **beside** the lens, not in
   front of it. If a spot feels "too far to the side to be seen", it is usually right.

| Board in the image | Where you hold it (at about 30 cm) |
|---|---|
| centre | straight ahead |
| left / right edge | about 28 cm to the side, only about 10 cm forward |
| top edge | about 25 cm above the lens, about 15 cm forward |
| **top-left / top-right corner** | **up and to the side at 45°, barely in front (about 5 cm forward); hold 2 s** |
| lower left | low and left, at finger height, left of the left fingertip |

5. Do this circuit **three times: at about 30 cm, 20 cm and 45 cm**. Then a few holds with the board tilted 30–45°
   (left, right, up, down). About 30 holds in all. Stop recording.

The checks split the image into a 3 x 3 grid. The gripper covers the bottom middle and bottom right, so 7 cells
count, and each needs board corners:

```
+-----------+-----------+-----------+
| TOP-LEFT  |    TOP    | TOP-RIGHT |
+-----------+-----------+-----------+
|   LEFT    |  CENTRE   |   RIGHT   |
+-----------+-----------+-----------+
| LOW-LEFT  |  gripper  |  gripper  |
+-----------+-----------+-----------+
```

**Common mistakes:** pointing the rig at the table tag instead of the board; the board only ever in
the middle; bare paper that bent; too few still holds. The two things that matter most: **a rigid flat board**, and
**the board right out at the edges, both top corners and the lower left.**

### 2.3 Process it

```bash
ls -t /data/incoming | head -3                 # newest first: your board recording
umiproc lens-calibrate <LENS_EPISODE>
```

- Takes 1–2 minutes. It finds the board in every frame, keeps the still ones, picks up to 60 views (every 5th is
  held back to test the fit), fits the lens and runs checks (a)–(h).
- Results: `~/data/umi/lens/pi-<date>_<LENS_EPISODE>/` (`<date>` is when the Pi finished the recording), with
  `report.json` (every check), `coverage.png` (every corner used, the 3 x 3 grid with counts) and `lens.yaml` (the
  fit). A copy goes to `~/data/analysis/lens_calibration/<LENS_EPISODE>/` (`$UMIPROC_ANALYSIS` changes that folder).
- **Every check only warns.** Any fit is written. It fails (`NO LENS`, exit 1) only when the board was seen too
  rarely to fit anything.
- **Look at `coverage.png`:** green dots should reach all 7 cells and, in the top corners, past the magenta circle.

### 2.4 The checks

| Check | What it means | If it warns |
|---|---|---|
| (a) ≥ 25 views | enough still, different views went into the fit | record longer, pause at each pose; the message says if the board was too rarely seen (closer, face the lens, more light) or the frames were moving (slower, longer pauses) |
| (b) fit error ≤ 0.5 px | the corners agree with one lens model | flatter board (glued to something rigid), slower moves, focus locked, no glare |
| (c) test error ≤ 0.6 px | the fit also predicts the held-back views | more varied poses: whole image, tilts, near and far |
| (d) reaches the edge | at least 10 corners reached 85 % of the way to the image's edge | put the board in the top-left and top-right corners (see the table in 2.2) |
| (e) all 7 cells | each cell has ≥ 20 corners | the message names the empty cells: hold the board there |
| (f) focal length within 10 % of the rig file's guess | the size of the lens result is plausible | the guess is known to be ~18 % off, so this one often warns; fine if the others pass |
| (g) lens centre within 40 px of the image centre | the centre is plausible | usually too little coverage, see (d) and (e); if it stays, check the lens is screwed in straight |
| (h) field of view 140–180° | the model gives about 160° | more corners near the top corners, see (d) |

A good recording later simply replaces a weak one. The map checks and the test demos show whether the lens is good
enough.

### 2.5 Install it with --apply

```bash
umiproc lens-calibrate <LENS_EPISODE> --apply
```

Repeats the calibration (same result) and installs the fit as `~/data/umi/rigs/rig_pi_v3.lens.yaml`, the file the
processing reads. An older one is kept as `rig_pi_v3.lens.yaml.bak-<time>`. **Worked when** it prints `applied:`.
Then:

```bash
umiproc status
```

`lens file:` must say `calibrated` with the date and recording. Install a lens **between** sessions, never in the
middle of one: maps made with the old lens become out of date.

### 2.6 When to redo it

- The lens was refocused or turned (the tape moved).
- The camera was taken off, remounted, or the lens swapped.
- The camera's resolution or mode changed.

## 3. A session: the order

One session = one place, one table setup, one sitting. It starts by proving the map works for this task (a mapping
clip, then a few real demos). Only then do you record the gripper calibration and the rest. A bad map spoils every
demo of the session, so these few minutes up front are worth it.

Each clip is one ordinary Pi episode (start, record, stop). **Write down which number is which:**

```
<MAP_EPISODE>                             mapping clip, table tag on the table  -> umiproc map        (section 4)
                                          then remove the table tag
<FIRST_TEST_DEMO> … <LAST_TEST_DEMO>      2-3 test demos                        -> umiproc run        (section 5)
<GRIPPER_EPISODE>                         gripper calibration clip              -> umiproc calibrate  (section 6)
<FIRST_DEMO> … <LAST_DEMO>                the rest of the demos                 -> umiproc run        (section 7)
```

The test demos are real demos. They wait until the gripper calibration is in, then become episodes like the rest.

**Prepare before the mapping clip:**

- **Table tag:** ArUco dictionary `DICT_4X4_50`, id 13, 160 mm (UMI's `table_marker.pdf`). Glue it flat onto
  something rigid and tape it **flat on the table** where the demos happen. Not on a wall, not bowed. It must not
  move while you record the map.
- **Scene:** set up everything the demos need (objects, lighting) first. Don't move anything that isn't part of the
  task until the session ends. Even light, and some texture in view (objects, edges, posters), not only a blank wall
  or bare table.
- **Rig:** lens calibrated and taped (section 2), fingers move freely, both finger tags clean and visible.

In the commands below, replace the `<...>` placeholders: each `<..._EPISODE>` / `<..._DEMO>` is a recording's folder
name in `/data/incoming` (`episode_` + the Pi's number; `ls -t /data/incoming | head` lists the newest). `<TASK>` is
the task's name (e.g. `block_raise`). `<FOLDER>` is the session folder `umiproc map` creates and prints: `<TASK>`, or
`<TASK>_2`, `<TASK>_3`, … when the name is taken.

## 4. The map

Every demo of the session is found inside this one map, so all demos come out in the same coordinates: the table
tag's.

### 4.1 Record the mapping clip (1–2 min)

A demo can only be found where the map has seen the scene from a similar place and angle.

1. Start with the gripper **30–50 cm above the table tag, looking at it**. Hold still about 3 s.
2. **Small, slow motions over the tag** for about 5 s: up and down, side to side, a little tilt. This gets SLAM and
   the IMU started.
3. **Scan the workspace slowly.** Go everywhere the demos will go, at the heights and angles they use (approach,
   grasp, lift, place), **including where the demos start**. Add a few wider views. Mix gentle turns with real
   sideways and up/down movement; don't just spin in place.
4. **At least 30 s of smooth, continuous motion.** No fast swings, no hand over the lens, no pointing at a blank wall
   or straight down at the bare table.
5. Keep the fingers still. Stop recording.

**Leave the tag and the scene alone until the map has passed.** If it fails, you record another clip of the same
setup.

### 4.2 Run the map

```bash
ls -t /data/incoming | head -3                 # newest first: the mapping clip
umiproc map <TASK> <MAP_EPISODE>
```

- Takes about **3 times the clip's length** (SLAM runs twice in real time: once to build the map, once to find the
  clip again in the saved map). About **5 times** if the clip gets cut (4.5). Run it in tmux.
- Needs free disk for a temporary copy: about the recording's size plus ~400 KB per frame (~1.4 GB for 2 min).
  Removed at the end.
- **Check the first lines** say `lens calibrated` (not `nominal, uncalibrated`).
- **Worked when** it prints `map: PASSED -> tasks/<FOLDER>`. Write down `FOLDER`: `run` and `calibrate` take it.

### 4.3 Read the result

The output has one line per check M1–M5: `PASS`, `WARN`, `FAIL`, or `n/a` (could not be judged). Each warning comes
with what to change. Then a few numbers that are reported but not judged, the report's path, and a `Next:` line with the commands
to look at the map.

- **Only M1 stops the map.** M2–M5 only warn: the folder is still created. The test demos (section 5) show whether
  the warnings matter.
- **Exception:** if M5 saw the table tag in no keyframe at all, the map has no table position, and `umiproc run`
  refuses that folder. Map again with the tag in view.
- **pacing** line: `fed 24.6 of 30.0 frames/s` with a `WARNING` means the Jetson could not keep up, and the map can
  be worse than the clip allows. Close other programs and put the Jetson in its maximum power mode
  (`sudo nvpmodel -q` shows the current mode).
- Exit code: 0 map attached (maybe with warnings), 1 M1 failed, 2 could not run (the message says why: recording
  missing or still syncing, already used, wrong camera settings, SLAM build missing, another umiproc command running).

**When it passed: remove the table tag** carefully, without moving anything else. The robot will not have it, so the
demos must not see it either.

### 4.4 The map checks

| Check | Passes when | If it warns or fails |
|---|---|---|
| **M1** one map (**blocks**) | SLAM ends with exactly one map: it never lost its place and started over | "never initialised": tag fully in view and sharp at the start, more light, more texture. "lost every map" or several maps: calm start above the tag, slow moves, no blank views. "crashed" or "did not finish": not the recording; send the log. A hint naming the **camera model**: not the recording either; redo the lens calibration |
| **M2** IMU ready | the IMU was fully set up (two stages, after about 5 s and 15 s of steady tracking) | record longer (≥ 30 s), keep tracking unbroken, move up/down, sideways and forward/back, not only turning. If the hint names the camera model: lens calibration |
| **M3** finds itself again | the same clip, played against the saved map, is found within 0.5 s and ≥ 95 % of its frames are tracked | start still above the tag; no fast moves or blank views (the message gives the time of the longest loss); nothing moved during the clip |
| **M4** gravity, scale, up, speed | the path agrees with the IMU: gravity 9.81 ± 0.5 m/s², size within 10 % of real, "up" stays within 3° through the clip (needs ≥ 10 s tracked), never faster than 3 m/s | scale off or "could not be measured": more slow 20–40 cm strokes up/down and sideways, especially right after the still start. "Up" off on a smooth clip: not the recording; lens model or camera mounting in the rig file. "not enough tracked span": keep it tracked (M3). Speed: move slower |
| **M5** table tag | ≥ 20 keyframes (snapshots SLAM keeps) see tag 13 at 0.3–2 m near the image centre; its position varies ≤ 1 cm; it lies within 5° of level | tag in the middle of the view at the start and again during the scan, 30 cm–2 m away, flat and never moved. A spread or tilt warning with a flat, still tag means the map itself is off (see M3, M4). **No sighting at all: `run` refuses the folder** |

### 4.5 When SLAM ends with several maps

When SLAM loses tracking partway, it starts a new map, so the clip ends with several map pieces. `umiproc map` then
**handles it by itself**: it picks the longest piece, cuts the clip to it, and maps and checks only that part again.
The output says so (`longest map (automatic): … checked only the longest map piece: N of M frames, a-b s`). The whole
clip's run is kept in `mapping_full/`.

The demos must then stay inside the part of the workspace that piece saw. To keep the **first** piece instead (from
the start, where the table tag is, up to the first loss):

```bash
umiproc map <TASK> <MAP_EPISODE> --first-map
```

M1 fails only when even the cut part does not give one map.

### 4.6 Look at the map

Two commands draw the map's SLAM runs. Both take the session folder and write into
`~/data/umi/tasks/FOLDER/analysis/`:

```bash
umiproc slam-video <FOLDER>
```

Writes `slam_video_mapping.mp4` and `slam_run_mapping.png`. About a minute per minute of clip. Play it with VLC (a
browser may not play it). It shows each frame SLAM was fed, with a coloured border: **green** = tracked in the saved
map, **amber** = tracked in a later map (after a loss), **red** = lost. Beside it, the camera path from above; below,
a timeline. `--run relocalize` shows the second run (check M3), `--run mapping_full` the whole clip before a cut,
`--run all` every run.

```bash
umiproc slam-3d <FOLDER>
```

Writes `slam_3d.html` (about 2 MB). Open it in a browser (it needs internet the first time). It shows the map's 3-D
points in their real colours (taken from the keyframe photos), the keyframes as small cameras, and the camera path
coloured by time, with a slider to move along it. Beside the plot is the photo of the last keyframe at or before the
slider's frame (that keyframe is drawn in orange). Hover over a keyframe to see its photo, click it to move the slider
there. The axes are real metres only when M4's scale passed. `--run mapping_full` shows the whole clip's run.

Points far from the work area are hidden by default; the bar at the top changes this and shows how many points each
rule hides. The defaults are `--min-obs 3` (hide points seen by fewer than 3 keyframes) and `--near 1` (hide points
farther from the camera path than 1 x the path's size, the largest side of its bounding box). `--near 0.5m` sets a
distance instead, and `--near 0` turns the rule off. `--box 0.5` also hides points outside the path's box grown by
0.5 x its size. `--show-outliers` keeps the far outliers, and `--no-images` skips the photos.

**What to look for:** the border mostly green; at each red stretch, what the camera saw (blank wall, fast swing,
hand); the points form the table and objects; the path covers where the demos will go.

`umiproc plot-run` draws one SLAM run from its files (`--csv trajectory.csv --log umiproc_slam.log`, optionally
`--video` with the ultrawide MP4) for one-off figures (example in 5.3).

### 4.7 When the map fails

Nothing is created and the clip is not used up, so it can be run again. The report and SLAM logs are kept in
`~/data/umi/maps_failed/<pi_name>/<time>/` (the last lines print the folder; `umiproc status` lists recent ones).
`<pi_name>` is umiproc's name for the recording, e.g. `pi-<date>_<MAP_EPISODE>`.

1. Read the failed check's hints (4.4).
2. Look at the failed run: pass the run folder instead of a session folder. Output goes to `…/<time>/analysis/`.

   ```bash
   umiproc slam-video ~/data/umi/maps_failed/<pi_name>/<time>/mapping
   umiproc slam-3d ~/data/umi/maps_failed/<pi_name>/<time>/mapping
   ```

3. With the tag still in place, record a new mapping clip following the hints, and run `umiproc map` on it.

### 4.8 Out of date and interrupted maps

**Out of date.** A map remembers the settings it was made with: the three rig files, the gripper mask, the SLAM
settings and the SLAM build. If any changed (an edit in `~/data/umi/rigs/`, a new lens, a rebuild), the map is out of
date and `umiproc run` refuses it. See it with:

```bash
umiproc status                  # "map passed, OUT OF DATE" per folder
umiproc status <FOLDER>      # what differs
```

Put back what changed, or record a new mapping clip (a new session folder) and process the demos against that.

**Interrupted map** (killed, power lost, ssh dropped outside tmux):

- Usually nothing was used up: run the same clip again. If the next command stops on a leftover
  `~/data/umi/incoming/<pi_name>` (status says "incoming/ is not empty"), delete it: it is only a converted copy,
  the Pi's recording is untouched.

  ```bash
  rm -rf ~/data/umi/incoming/<pi_name>
  ```

- Rare: killed in the second it attaches a passed map. `umiproc status FOLDER` then shows `INCOMPLETE`, and the clip
  counts as used. Record a new mapping clip (it makes a new folder). If status also says the map is still in
  `~/data/umi/.work/map/<pi_name>`, you can move it in by hand instead, **before any other `umiproc map`**:

  ```bash
  W=~/data/umi/.work/map/<pi_name>
  S=~/data/umi/tasks/<FOLDER>/map/<pi_name>           # the INCOMPLETE slot status names
  grep '"folder"' $W/report.json                       # must be <FOLDER>
  for f in map_atlas.osa orbslam3_settings.yaml slam_mask.png slam_fingerprint.json mapping relocalize \
           mapping_full tag_detections.json tx_slam_tag.json; do [ -e $W/$f ] && [ ! -e $S/$f ] && mv $W/$f $S/; done
  mv $W/report.json $S/report.json                     # last: this marks the map complete
  umiproc status <FOLDER>                              # map passed
  ```

## 5. Test demos: verify the map

Two or three real demos show that demos of this task, without the table tag, find themselves in the map.

### 5.1 Record 2–3 test demos

Table tag removed. Record them exactly as you will record the rest. If demos start from different places, start each
test demo from a different one.

1. Start with the gripper **already looking at the mapped workspace** and **hold still about 1 s**. The demo must be
   found in the map within 0.5 s.
2. Do the task once, at a natural pace. No fast swings, never cover the lens. Losing track for more than 5 frames
   (0.17 s) drops or splits the demo.
3. Stay inside the area the map covered. Keep both finger tags in view.
4. Stop, reset the scene, next demo.

### 5.2 Run them

```bash
umiproc run <FOLDER> <FIRST_TEST_DEMO> <LAST_TEST_DEMO>
```

- The range is first to last recording, inclusive. The mapping clip and recordings
  already done are skipped by themselves.
- Takes about **1.5 times the demos' length**.
- `--dry-run` shows what each recording would get and changes nothing.
- The map is checked first. If it is out of date, it stops (`OUT OF DATE`, exit 2) and changes nothing (4.8).

There is no gripper calibration yet, so each demo is found in the map, checked, and then **waits**. **Expected: exit
code 3**, with lines like:

```
<demo 1>  ->  waiting for the gripper calibration (1 piece(s) passed every pose check)
<demo 2>  ->  waiting for the gripper calibration (1 piece(s) passed every pose check)
<demo 3>  ->  failed (slam: late_relocalization)
```

Exit codes: 3 some demos wait (expected here); 0 every recording got an outcome (here: all failed); 2 could not run,
nothing changed (no such folder, map unusable or out of date, SLAM missing, bad range, another umiproc command
running); 1 unexpected error (what was done keeps its outcome; run it again).

### 5.3 What happens to each demo

1. It is found in the map, with the gripper masked out. SLAM stops after 30 lost frames (1 s).
2. **UMI's rules** (at our 30 frames/s): dropped if SLAM stopped, if more than 5 frames (0.17 s) were lost in total,
   if fewer than 30 frames (1 s) were tracked, or if it was not found within 0.5 s. Otherwise it is **split at every
   lost frame** into **pieces**; pieces shorter than 12 frames (0.4 s) are dropped.
3. **Pose checks** on the whole demo: gravity, scale, and for demos of 10 s or more the "up" check of M4. Per piece:
   speed ≤ 3 m/s.
4. Without the gripper calibration it stops here and waits.
5. Once the calibration is in (section 6): the finger opening from the finger tags, checks on each piece, then the
   training file. Each passing piece becomes its own `episode_N`.

A failed demo goes to `~/data/umi/tasks/FOLDER/failed/<pi_name>/` with `reason.json`. The reasons:

| Reason | Means | Change in the recording |
|---|---|---|
| `stopped_lost` | SLAM could not follow the demo for 1 s and stopped | start looking at the mapped workspace, stay in the mapped area, no fast swings, never cover the lens |
| `too_many_lost` | more than 5 frames lost in total, **including the first frames before it was found** | start still (1 s) looking at what the map saw; smooth motion |
| `late_relocalization` | not found in the map within 0.5 s | as `too_many_lost`: the start must show what the map knows |
| `too_few_tracked` | tracked for less than 1 s | longer demos |
| `no_piece` | no tracked stretch of 0.4 s between the losses | fewer losses |
| `new_map` | SLAM lost the map and started a new one | stay in the mapped area, move more slowly |
| `slam_error` | SLAM crashed or timed out | not the recording: read `failed/<pi_name>/processed/localize/umiproc_slam.log` |
| `gravity` | the path disagrees with the IMU's gravity (whole demo) | smooth motion, no bumps; a very short brisk demo can fail on its own; if it repeats on smooth demos: map M4, lens |
| `scale_mismatch` | the size of the motion is off by more than 10 % (whole demo) | the map's scale (M4) or a SLAM jump after a loss |
| `orientation` | "up" drifts by more than 3° (demos ≥ 10 s, whole demo) | not the recording if smooth: lens model, camera mounting |
| `speed`, `speed_extreme` | faster than 3 m/s (over 7 m/s: a SLAM jump) | move slower |
| `tag_coverage` | finger tags seen in fewer than 80 % of the piece's frames | keep both finger tags in view |
| `kept_fraction`, `piece_too_short` | finger opening or main-camera frames missing for over 20 % of the piece | as `tag_coverage` |
| `ultrawide_gap`, `imu_gap` | frames or IMU samples missing in the recording | not the hand: check the Pi's recording |

A piece that fails is dropped and the others kept. `gravity`, `scale_mismatch` and `orientation` drop the whole
demo. "scale not observable" (demo too short or still to measure) is only reported, not a failure.

To see a demo's SLAM run as a figure (tracking over time, the path):

```bash
D=~/data/umi/tasks/<FOLDER>/failed/<pi_name>
umiproc plot-run --csv $D/processed/localize/trajectory.csv --log $D/processed/localize/umiproc_slam.log \
    --video $D/raw/cam1.mp4 --out slam_run_demo.png
```

`cam1.mp4` is the ultrawide camera in umiproc's copy of a recording (`raw/meta.json` names each camera's role).

### 5.4 Is the map verified?

| Test demo outcome | Means | Do |
|---|---|---|
| **all waiting** | demos find themselves in the map and pass the checks | **verified**: go to section 6 |
| one failed for a recording reason (`speed`, `tag_coverage`, a fast swing), the others waiting | the map is fine, that demo wasn't | record one more test demo and run it; verified once it waits |
| `late_relocalization`, `too_many_lost`, `stopped_lost` or `new_map` on calm demos | the map does not cover where the demos start or go | put the tag back, record a new mapping clip covering the demos' starts and paths, `umiproc map` (a new folder), new test demos |
| `gravity`, `scale_mismatch` or `orientation` on smooth demos | the map's scale or "up", or the lens | stop and look into it: the map's M4 line, `umiproc status FOLDER` |
| `slam_error` | SLAM crashed | read the log named in the table above |

## 6. Gripper calibration

Measures the finger opening range, so every frame's opening can be read from the two finger tags. Once per session,
after the map is verified.

### 6.1 Record (about 15–20 s)

1. Start recording with the gripper pointed at the scene (anything works).
2. **Close the fingers fully, then open them fully, slowly.** At least 5 open/close cycles. "Fully open" is the
   widest the demos will use (the robot's limit is 80 mm).
3. **Both finger tags visible the whole time:** keep your hand and objects off them. Stop recording.

### 6.2 Run it (about 1 min)

```bash
umiproc calibrate <FOLDER> <GRIPPER_EPISODE>
```

**Worked when** it prints `calibrate: PASSED -> tasks/<FOLDER>/gripper_range.json`, with lines like:

```
  measured opening 0.080 m (expected 0.080 m +- 0.010 m); closed width 0.0440 m (expected 0.044 m), open width 0.1240 m
  UMI's finger-tag rule PASS: tag 0 seen in 97 % and tag 1 in 95 % of 452 frames (both at once: 94 %); UMI needs each tag in >= 80 %
```

Exit codes: 0 passed; 1 failed (nothing recorded: fix it and record again); 2 could not run (no such folder, map not
passed, recording missing or already used, another umiproc command running).

| Line | Passes when | If not |
|---|---|---|
| finger-tag rule (**fails**) | each finger tag seen in ≥ 80 % of the frames | keep both tags in view (the message names the one missed most); even light; open and close slowly |
| the fit (**fails**) | both tags found at the expected distance in enough frames | the message names the cause |
| `warning: measured opening` | opening 0.080 m ± 0.010 m | open and close fully, tags visible at both ends; record again if far off |
| `warning: closed width` | closed width 0.044 m ± 0.010 m | close fully; a tag hidden when closed reads as a wider gripper |
| `warning: only N open/close cycles` | at least 5 cycles | more cycles |

**Redoing it:** a new calibration replaces the old one only while the folder has no episodes yet (the waiting test
demos don't count). Once episodes exist, `calibrate` refuses: record a new session instead. For the same reason `run`
refuses if the rig file's tag/gripper/camera settings or the lens changed after episodes were made.

## 7. The rest of the demos

### 7.1 Record

As the test demos (5.1): start still for 1 s looking at the mapped workspace, natural pace, stay in the mapped area,
no fast swings, finger tags visible. Reset between demos. Keep writing down the numbers.

Process in batches as you go (7.2 on what you have so far): a problem shows after a few demos, not at the end.

### 7.2 Run

```bash
umiproc run <FOLDER> <FIRST_DEMO> <LAST_DEMO>
```

- Now each demo goes all the way: found in the map, split, checked, finger opening, training file. Each passing
  piece becomes `episode_N`; a failed demo goes to `failed/<pi_name>/`.
- **The waiting test demos are finished by this same run** (any range does it), in about a minute each, without
  SLAM again.
- **Give `run` only demo numbers.** Run `calibrate` before this, so the calibration clip is already used and skipped.
- Exit code 0: every recording got its outcome. Other codes as in 5.2.

Output lines look like:

```
<demo 1>  ->  episode_1
<demo 2>  ->  episode_2, episode_3 (2 of 3 pieces; piece 2 dropped: speed); piece 1 trimmed 0.27 s (no finger width)
<demo 3>  ->  failed (slam: too_many_lost)
```

"trimmed" means a short part of the piece without a finger opening or main-camera frame was cut off.

### 7.3 Check the folder

```bash
umiproc status <FOLDER>
```

Shows the map, the gripper calibration, the episodes (pieces of one recording on one line), what is still waiting
and why, and the failed recordings with their reasons.

### 7.4 Run again, interrupted runs, deleting

- **Failed recordings are not retried by themselves.** After fixing a cause outside the recording (e.g. a rebuilt
  SLAM for `slam_error`), retry them from scratch:

  ```bash
  umiproc run <FOLDER> <FIRST_TEST_DEMO> <LAST_DEMO> --retry-failed
  ```

- **An interrupted run** (Ctrl-C, ssh, power) leaves its recording in `.staging/`. The next `run` of the folder (any
  range) picks it up and finishes it.
- **Never delete the first piece of a split recording alone.** The recording (`raw/`) is stored once, in the first
  piece's `episode_N`; the other pieces link to it. To drop a whole recording, delete all its pieces. To drop one
  later piece, delete only its folder.

## 8. After the session

### 8.1 What you have

```
~/data/umi/tasks/FOLDER/
  map/<pi_name>/          the map, its checks (report.json) and the mapping clip
  calibration/            the gripper calibration clip
  gripper_range.json      the finger opening range
  episode_1/ episode_2/ … one per passing piece
    raw/                  the recording (a link to the first piece's for later pieces)
    processed/episode.zarr   the training data
  failed/<pi_name>/       failed demos, each with reason.json
  analysis/               videos and figures from slam-video / slam-3d
```

Poses in every `episode.zarr` are relative to the table tag: tag 13 at the origin, z up. All episodes of the session
share these coordinates.

### 8.2 Copy it to the laptop

On the laptop (`<jetson>` is your ssh login for the Jetson, e.g. `user@192.168.1.50`):

```bash
mkdir -p ~/Capstone/data/<FOLDER>
rsync -av --exclude .staging <jetson>:data/umi/tasks/<FOLDER>/ ~/Capstone/data/<FOLDER>/
```

Copies the whole folder, links included, without the work-in-progress `.staging/`. Running it again copies only
what is new.

### 8.3 Training: the missing step

The training scripts are on branch `rnoro/training` (`train/README.md` there:
`git -C ~/Capstone-data-work show rnoro/training:train/README.md`). Today they expect dataproc's layout
(`~/data/tasks/<folder>` on the Jetson) and a dataset zip made by `dataproc package`.

**Packaging umiproc episodes into a training dataset is not built yet** ([TODO.md](TODO.md) "Later"). The missing
`umiproc package` step will gather a folder's `episode_N/processed/episode.zarr` files into one `<dataset>.zarr.zip`
plus `package.json` in the iPhUMI format, which the training scripts then upload and train on.

## Appendix A: the A/B comparison

First real session only. The first passing mapping clip is mapped three times, each in its own data root, to choose
the SLAM settings:

| Variant | SLAM build | IMU noise |
|---|---|---|
| `all_infl1` | all patches 0001–0012 (the default) | as in the datasheet |
| `p0006_infl1` | patches 0001–0006 only | as in the datasheet |
| `all_infl10` | all patches 0001–0012 | ×10 |

The 0001–0006 build also lacks the crash fixes 0010–0011, so it can crash when a second map starts; run it again if
it does. `/data/incoming` is only read and `~/data/umi` is not touched, so this can run after the session.

Build the 0001–0006 variant once (20–30 min on the Jetson; goes to `~/ORB_SLAM3_umi/umiproc_build_0006/`):

```bash
PATCHES_UPTO=0006 ~/Capstone/umi-processing/native/build.sh
```

Paste this into the terminal. It makes one root per variant with a copy of the current rig files and lens, sets the
IMU noise factor, and prints `STOP: …` on anything unexpected:

```bash
AB=~/data/umi_ab
REPO=~/Capstone/umi-processing
RIG=rig_pi_v3

setup_variant() {                    # setup_variant NAME BINARY INFLATION
  local name=$1 bin=$2 infl=$3 root=$AB/$1 f src
  [ $# -eq 3 ] || { echo "usage: setup_variant NAME BINARY INFLATION"; return 1; }
  [ -e "$root" ] && { echo "STOP: $root already exists (rm -rf $root to set it up again)"; return 1; }
  [ -x "$bin" ] && [ -f "$(dirname "$bin")/BUILD_INFO" ] || { echo "STOP: no SLAM build at $bin"; return 1; }
  mkdir -p "$root/rigs" || return 1
  for f in $RIG.yaml $RIG.slam.yaml; do
    src=~/data/umi/rigs/$f
    [ -f "$src" ] || { [ $f = $RIG.yaml ] && src=$REPO/configs/rigs/$RIG.yaml || src=$REPO/configs/slam/$RIG.yaml; }
    cp "$src" "$root/rigs/$f" || return 1
    echo "  $f from $src"
  done
  if [ -f ~/data/umi/rigs/$RIG.lens.yaml ]; then
    cp ~/data/umi/rigs/$RIG.lens.yaml "$root/rigs/" && echo "  lens file from ~/data/umi/rigs/$RIG.lens.yaml"
  else
    echo "  lens file: none (nominal, uncalibrated)"
  fi
  sed -i -E "s/^( *)vio_inflation: \{[^}]*\}/\1vio_inflation: {noise_density: $infl, random_walk: $infl}/" \
      "$root/rigs/$RIG.yaml"
  grep -qF "vio_inflation: {noise_density: $infl, random_walk: $infl}" "$root/rigs/$RIG.yaml" ||
    { echo "STOP: vio_inflation $infl is not set in $root/rigs/$RIG.yaml (edit it by hand)"; return 1; }
  echo "$name: SLAM binary $bin ($(grep -c '^patch ' "$(dirname "$bin")/BUILD_INFO") patches), vio_inflation $infl"
}

setup_variant all_infl1   ~/ORB_SLAM3_umi/umiproc_build/umiproc_slam      1.0
setup_variant p0006_infl1 ~/ORB_SLAM3_umi/umiproc_build_0006/umiproc_slam 1.0
setup_variant all_infl10  ~/ORB_SLAM3_umi/umiproc_build/umiproc_slam      10.0
```

Each must end with `NAME: SLAM binary … (12 patches)` (6 for `p0006_infl1`) and no `STOP`. Then map the clip in
each (in tmux):

```bash
env -u UMIPROC_SLAM_BINARY UMIPROC_ROOT=$AB/all_infl1 \
    umiproc map <TASK> <MAP_EPISODE> 2>&1 | tee $AB/all_infl1.txt
UMIPROC_ROOT=$AB/p0006_infl1 UMIPROC_SLAM_BINARY=$HOME/ORB_SLAM3_umi/umiproc_build_0006/umiproc_slam \
    umiproc map <TASK> <MAP_EPISODE> 2>&1 | tee $AB/p0006_infl1.txt
env -u UMIPROC_SLAM_BINARY UMIPROC_ROOT=$AB/all_infl10 \
    umiproc map <TASK> <MAP_EPISODE> 2>&1 | tee $AB/all_infl10.txt
```

- **Check the first lines** of each: `SLAM build …/umiproc_build/umiproc_slam (patches 0001-0012 (12 patches))` for
  the `all_` variants, `…/umiproc_build_0006/umiproc_slam (patches 0001-0006 (6 patches))` for `p0006_infl1`;
  `noise_density 10, random_walk 10` only for `all_infl10`; the same `lens …` in all three.
- **Compare** the M2, M3 and M4 lines (`$AB/*.txt`): which finishes the IMU setup, is found sooner and tracks more
  frames, and has the scale closest to 1. SLAM varies a little between runs, so small differences mean nothing; run a
  close one twice (a passed variant says "already taken": `rm -rf $AB/<variant>`, then its `setup_variant` line
  again).
- Patches 0007–0009 only matter after SLAM finds the map again later in a video, so the mapping clip may not show
  them. The demos will: run `umiproc run` with the same `UMIPROC_ROOT` / `UMIPROC_SLAM_BINARY` as that variant's
  `map`. Without a gripper calibration in that root they stop at "waiting", which is enough to compare tracking.
- Report which variant won (TODO.md, "Comparisons"); later sessions then use that setting in `~/data/umi`. Clean
  up when done: `rm -rf ~/data/umi_ab`.
