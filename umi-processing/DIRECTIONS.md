# umiproc: directions

Step-by-step procedures for the Pi 5 rig, in the order you do them. Each section is added once its commands exist
(docs/DESIGN.md "Build order"). Commands run on the Jetson unless a section says otherwise.

0. Python environment (once per machine)
1. Lens calibration (optional, about 1 minute of recording)
2. Jetson setup
- A collection session at a glance: what to record, in order, and what to run
3. Prove the environment: the map
4. Prove the demos
5. Gripper calibration and finishing the episodes

## Start here: bringing the Jetson up, step by step

Do these in order; each step has a check that tells you it passed before you go on.

| # | Do | Where | Passed when |
|---|---|---|---|
| 1 | Get the code: `cd <repo> && git fetch && git checkout rnoro/data-work && git pull` | Jetson | `git log --oneline -1` shows the newest umi-processing commit |
| 2 | Python environment (section 0, about 5 min) | Jetson | `umiproc --help` lists `lens-board`, `lens-calibrate`, …, and the check tests pass |
| 3 | Start the SLAM build (section 2.1, about an hour) **inside `tmux`**, and let it run while you do step 4 | Jetson | it ends with `== done` |
| 4 | **Lens calibration** (section 1): print the board, tape it to cardboard, record about 60 s on the Pi, run `umiproc lens-calibrate`, repeat until every check passes, then `--apply` | Pi + Jetson | `lens-calibrate` prints no `FAIL`, and `--apply` installs `~/data/umi/rigs/rig_pi_v3.lens.yaml` |
| 5 | `umiproc status` (section 2.3) | Jetson | lens file: calibrated; umiproc_slam and vocabulary: found |
| 6 | A collection session: map, then gripper calibration, then demos | Pi + Jetson | sections 3–5 (`umiproc map` is being finished; it comes with the next update of this branch) |

**Do step 4 before any mapping.** The lens model matters more than anything else for SLAM on this rig: on the first
real mapping clip (episode 128) the rig file's nominal lens, about 18 % off in focal length, made SLAM lose every map
it started (34 resets, no usable map), while the same clip with a measured lens built a map with the IMU fully
initialised. Lens calibration needs only step 2, not the SLAM build.

`tmux` in one line: `tmux new -s build`, run the command, detach with `Ctrl-b` then `d`, come back with
`tmux attach -t build`. A long command keeps running even if your ssh connection drops.

## 0. Python environment (once per machine, about 5 minutes)

Every `umiproc` command, including the lens calibration in section 1, runs in this environment. It is all section 1
needs; the SLAM build (section 2.1) is only needed from the map on.

umiproc has its own environment and never imports dataproc, so don't install it into `~/envs/env_dataproc`. On the
Jetson, first make sure Python can make environments:

```bash
sudo apt install python3-venv python3-pip
```

Then:

```bash
python3 --version                                   # 3.10 or newer
python3 -m venv ~/envs/env_umiproc
source ~/envs/env_umiproc/bin/activate              # in every new terminal
unset PYTHONPATH                                    # also in every new terminal, if ROS is set up in your shell
pip install --upgrade pip
pip install -e <repo>/umi-processing[test]
umiproc --help                                      # lens-board, lens-calibrate, map, calibrate, run, status
```

Check the install (a minute or two; it must end with `passed`, no `failed`):

```bash
cd <repo>/umi-processing
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_umiproc_*.py tests/test_lens_*.py tests/test_slam_*.py -m "not slow"
```

**In every new terminal**, before any `umiproc` command:

```bash
source ~/envs/env_umiproc/bin/activate
unset PYTHONPATH
```

## 1. Lens calibration

**Why:** it measures how the ultrawide (IMX219, 160°) lens bends light, so SLAM uses the real lens instead of the
rig file's nominal guess. `map` and `run` never wait for it (without it they use the nominal lens, labelled
"nominal, uncalibrated"), **but do it first**: the nominal lens is about 18 % off in focal length, which is enough
to make SLAM fail (see "Start here"). About 1 minute of recording, then 1–2 minutes of processing.

**What you need:** a printer, something rigid and flat to glue the board to (foam board, a clipboard, a flat piece
of wood or acrylic), glue (spray adhesive or a glue stick), a ruler, a small piece of tape, and a well-lit spot.
The commands need the Python environment from section 0 (about 5 minutes, once); "Start here" gives the order.

### Print and check the board

1. Make the PDF (on any computer with `umiproc`):

   ```bash
   umiproc lens-board                 # writes ./umiproc_lens_board.pdf (US Letter, landscape)
   umiproc lens-board --paper a4      # A4 instead; --out FILE to choose the file name
   ```

2. Print it at **100 % / "Actual size"** (not "Fit to page"), on matte paper if you have it.
3. Measure with the ruler: **one square must be 30.0 mm**, and the scale bar under the board 100 mm. If not, the
   printer scaled it: print again at 100 %.
   Printing it larger is fine if it is scaled the same in both directions (e.g. Tabloid/A3 at 140 %): the lens model
   does not depend on the board's absolute size; the 30 mm check then reads ~42 mm.
4. **Glue or tape it onto the rigid backing** (cardboard, foam board, a clipboard, a hardcover book), all four
   edges, smoothly: no bubbles, waves or curled corners. **Never record with bare paper**, held or lying on a table:
   paper bends and curls by a few millimetres, and that alone fails checks (b) and (c) (see "Lessons from the first
   tries" below). A flat board matters more than anything else here.

The board uses `DICT_5X5_100` markers on purpose: the gripper's finger tags and the table tag are `DICT_4X4_50` and
are often in view; they are never mistaken for the board.

### Before recording

- **Lock the lens focus:** the IMX219's lens focuses by turning in its holder. Once it is sharp, put a small piece of
  tape across the lens and its holder so it cannot turn. If the lens turns later, the calibration no longer
  applies (see "When to redo it").
- **Light:** bright and even, with no lamp or window reflected in the board. More light means shorter exposures and
  less motion blur.

### Record (on the Pi)

Record **one ordinary episode of about 60 seconds** on the Pi, started and stopped like a demo. The fingers are in
view at the bottom of the image; that is expected (that area is masked and ignored). Keep the fingers still the whole
time.

**The easy way: the rig stays still, you move the board.**

1. **Prop the rig on the table so it cannot move**, looking straight ahead (horizontal), fingers in their usual
   place. Start recording.
2. Hold the board **by its backing** (both hands if you can), **always turned to face the lens**.
3. Rhythm: **move for about 1 s, hold still for about 1 s** (count "one-one-thousand"). Only the holds count.
4. Put the board in each of these places. **The lens sees almost sideways** (about 75° off-axis at the image's left
   and right edges, about 80° at its top corners), so the edges and corners of the image are **beside** the lens,
   not in front of it: if a spot feels "too far to the side to be seen", it is usually exactly right.

| In the image | Where the board physically is (at about 30 cm from the lens) |
|---|---|
| centre | straight ahead |
| left edge / right edge | far out to the side: about 28 cm left or right and only about 10 cm forward |
| top edge | high up: about 25 cm above the lens and about 15 cm forward |
| **top-left / top-right corner** | **up and to the side at about 45° diagonally, barely in front of the lens (about 5 cm forward)**; hold these **2 s** |
| lower left | low and to the left, at the fingers' height, left of the left fingertip |

5. Do that circuit **three times: at about 30 cm, about 20 cm and about 45 cm**, then a few holds with the board
   tilted 30–45° (turned left, right, up, down). About 30 holds in all.

The image, as the 3 x 3 grid the checks use (the gripper covers the bottom middle and bottom right; the other 7
cells each need at least 20 board corners):

```
+-----------+-----------+-----------+
| TOP-LEFT  |    TOP    | TOP-RIGHT |
+-----------+-----------+-----------+
|   LEFT    |  CENTRE   |   RIGHT   |
+-----------+-----------+-----------+
| LOW-LEFT  |  gripper  |  gripper  |   only the lower left is usable
+-----------+-----------+-----------+
```

(The other way also works: tape the board flat on a wall and move the rig, turning it so the board lands in each
part of the image: turn the rig right and the board slides left in the image, tip it down and the board slides up.)

**What a good recording looks like when you play it back:** the board sharp in every hold and visibly flat; over the
clip it visits all 7 cells, including right up into both top corners and the lower left; some holds big (close),
some small (far), some at a slant. After processing, `coverage.png` shows green dots in all 7 cells and beyond the
magenta circle in the top corners.

#### Lessons from the first tries (episodes 122–127)

| Episode | What happened | Checks it failed |
|---|---|---|
| 122 | the table tag (one big 4×4 marker) instead of the board | all: the board was never seen |
| 124 | board on the table, rig pointed at it: the board only ever in the middle; few holds | a, b, d, e, h |
| 125 | bare paper held by one edge: it bent; never close, never lower left | a, b, c, d, e, g |
| 126 | bare paper held taut with two hands: much better (0.26–0.5 px when taut, over 1 px when it bent); not far enough into the corners or the lower left | a, b, c, d, e, h |
| 127 | loose sheets on the table, rig moved over them: curled edges; the board stayed in the middle | a, b, c, d, e |

Pooling the views of 125–127 does not help: the bent views pull the whole fit off (1.5 px). The two things that
were always missing: **a rigid board**, and **the board right out at the image's edges, top corners and lower
left**. The focal length came out at 464–472 px every time (the nominal is 572.9), so the lens is clearly not the
nominal one.

### Process it (on the Jetson)

This needs the Python environment from section 0 (once per Jetson; the SLAM build is not needed). In every new
terminal, turn it on first:

```bash
source ~/envs/env_umiproc/bin/activate     # the environment made in section 0
unset PYTHONPATH                          # if ROS is set up in this shell, its Python packages must not leak in
umiproc --help                            # check: lists lens-board, lens-calibrate, ...
```

Then find the recording's directory name (the newest one in `/data/incoming`) and run the calibration on it:

```bash
ls -t /data/incoming | head -3                 # newest first, e.g. episode_000123
umiproc lens-calibrate episode_000123
```

(On another computer, add `--source DIR` for where the recordings are, e.g. `--source ~/Downloads/data/incoming`.)

It checks the recording (complete, every file matches its manifest hash, the IMX219 at 1280x960). It then finds
the board in every frame, keeps the still frames, picks up to 60 varied views (every 5th one is held back to test the
result), fits the lens and runs the checks. Results go to `~/data/umi/lens/pi-YYYY-MM-DD_episode_000123/` (the Pi's
episode number with the date the Pi finished the recording, e.g. `pi-2026-11-16_episode_000123`, so a later recording
that reuses the number never overwrites this one; the last lines printed give the exact folder):

- `report.json`: every check with its value and limit, and the frames and views used.
- `coverage.png`: a frame with every corner used (green: fit, cyan: held back), the gripper mask (red), the 3 x 3
  grid with the corner count of each cell, and the circle that check (d) needs corners beyond (magenta).
- `lens.yaml`: the measured lens. Only written when every check passed.

It ends with exit code 0 when every check passed, 1 otherwise. A failure names the failed check and says what to
record differently. Every check is a hard limit except (f), which only warns (see the table).

### Reading the checks

| Check | What it means | If it fails |
|---|---|---|
| (a) fit views ≥ 25 | enough distinct, still views of the board went into the fit | record longer and pause at each pose; the message says whether the board was too rarely seen (closer, face the lens, more light, focus) or the frames were moving (slower, longer pauses) |
| (b) fit RMS ≤ 0.5 px | the corners agree with one lens model | flatten the board (glue it to something rigid), move more slowly, lock the focus, avoid glare |
| (c) held-out RMS ≤ 0.6 px | the model also predicts the views it was not fitted to | more varied poses: the whole image, tilts, near and far; same care as (b) |
| (d) radial coverage ≥ 85 % | at least 10 corners reached the image's outer edge, ≥ 85 % of the way from the image centre to its farthest visible pixel | put the board in the top-left and top-right corners of the image (turn the rig, see above) |
| (e) angular coverage | each of the 7 cells of a 3 x 3 grid that are mostly outside the gripper holds ≥ 20 corners | the message names the cells: move the board through those parts of the image and pause there |
| (f) focal length within ±10 % of nominal (**warning only**) | the result is plausible for this lens. The nominal value is the rig file's, never measured, so this check only warns (`WARN`, and in `report.json`'s `warnings`): it does not block `lens.yaml` or `--apply` | check that it is the 160° IMX219 lens recording at 1280x960; if it is, and every other check passed, the nominal is the likelier one to be off |
| (g) principal point within 40 px of the centre | the lens centre is plausible | usually too little coverage (see (d), (e)); if it persists, check that the lens is screwed in straight |
| (h) field of view 140–180° | the lens model gives the ~160° of this lens at the top corners | record more corners near the top corners (see (d)) |

### Use it: `--apply`

When every check passed, the last lines show the command that installs the result, which is the same command
plus `--apply`:

```bash
umiproc lens-calibrate episode_000123 --apply
```

It repeats the calibration (same result) and copies `lens.yaml` to `~/data/umi/rigs/rig_pi_v3.lens.yaml`, the lens
file the processing reads. A lens file already there is kept as `rig_pi_v3.lens.yaml.bak-<UTC time>`. `--apply`
does nothing when a check failed. Apply it before you record a session's map, not in the middle of a session: maps
and episodes record the lens they were made with.

### When to redo it

- The lens was refocused or turned (the tape moved).
- The camera was taken off or remounted, or the lens was swapped.
- The camera mode changed (resolution or sensor mode of the IMX219).

## 2. Jetson setup

Once per Jetson. `<repo>` below is the Jetson's clone of the Capstone repo, on branch `rnoro/data-work` (pull it
first). Nothing here touches dataproc's setup: its `~/ORB_SLAM3`, Pangolin in `/usr/local`, `~/envs/env_dataproc`
and `~/data` stay as they are.

### 2.1 Build the SLAM program (about an hour the first time)

umiproc uses UMI's own ORB-SLAM3 fork with our fixes (docs/DESIGN.md "SLAM"), built in `~/ORB_SLAM3_umi`:

```bash
sudo apt install build-essential cmake git libeigen3-dev libgl1-mesa-dev libopengl-dev libglew-dev \
    libboost-serialization-dev libssl-dev python3-venv python3-pip
<repo>/umi-processing/native/build.sh
```

- The first run takes about an hour: it builds its own OpenCV 4.5.4 + contrib, because the Jetson's OpenCV 4.8 is
  too new for the fork's tag code. Re-runs only rebuild what changed.
- If the compiler runs out of memory, run it again with `JOBS=1 <repo>/umi-processing/native/build.sh`.
- **Check:** it ends with `== done`, and `~/ORB_SLAM3_umi/umiproc_build/BUILD_INFO` lists the fork commit, every
  patch and the OpenCV it used.
- On a failure it prints the end of the failing step's log; all logs are in `~/ORB_SLAM3_umi/umiproc_build/logs/`.
- Build it on the machine that runs it (it is compiled for that machine's processor).

**Check** that the SLAM program runs and reads umiproc's settings file (about 30 s):

```bash
cd <repo>/umi-processing     # with the environment from section 0 turned on
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_umiproc_slam_inputs.py -m slow
```

It must say `passed`, not `skipped` (skipped means the SLAM program or vocabulary was not found: see above).

### 2.2 The data root

- umiproc writes everything under **`~/data/umi/`**; the first command that takes a recording creates it. Leave
  `UMIPROC_ROOT` unset.
- `/data` is refused (its `incoming/` is the Pi's), and so is dataproc's root (`~/data` itself, `$DATAPROC_ROOT`), so
  dataproc's data is never touched. `/data/incoming` is only read.
- `~/data/umi/rigs/` holds the files the processing reads:
  - `rig_pi_v3.yaml`: copy of `configs/rigs/rig_pi_v3.yaml` (cameras, IMU, CAD geometry);
  - `rig_pi_v3.slam.yaml`: copy of `configs/slam/rig_pi_v3.yaml` (gripper mask, table tag, ORB settings);
  - `rig_pi_v3.lens.yaml`: optional, from section 1.
- The first `map` copies the first two. To change something (for example a re-traced mask), edit the copy there, not
  the repo; `umiproc status` shows where a copy differs from the repo.
- Every map and episode records the values it was made with, so changing them makes the maps and episodes out of
  date.

### 2.3 Check: `umiproc status`

```bash
umiproc status
```

On a fresh Jetson after sections 0 and 2.1 it shows:

```
data root: /home/<you>/data/umi (from the default)
  it does not exist yet: the first umiproc command that takes a recording creates it

rig rig_pi_v3: the files the processing reads (in /home/<you>/data/umi/rigs)
  rig file:      rig_pi_v3.yaml: not there yet (copied from <repo>/umi-processing/configs/rigs/rig_pi_v3.yaml by the first command that needs it)
  SLAM config:   rig_pi_v3.slam.yaml: not there yet (copied from <repo>/umi-processing/configs/slam/rig_pi_v3.yaml by the first command that needs it)
  lens file:     none, so SLAM uses the rig file's nominal lens model ("nominal, uncalibrated"); optional: DIRECTIONS.md section 1

SLAM program (built by umi-processing/native/build.sh):
  umiproc_slam:  /home/<you>/ORB_SLAM3_umi/umiproc_build/umiproc_slam: found
  vocabulary:    /home/<you>/ORB_SLAM3_umi/Vocabulary/ORBvoc.txt: found
```

- `NOT FOUND` next to umiproc_slam or vocabulary: run the native build (2.1).
- `refused`: `UMIPROC_ROOT` points at `/data` or `~/data`; unset it.
- `PROBLEM:`: the line names the file and what to change.
- Later, in a session, it also shows how many recordings were taken (map / calibration / demonstration) and one line
  per task folder; `umiproc status block_raise` shows one folder: its map, gripper calibration, episodes, the
  recordings waiting in `.staging/` and why, and the failed ones with their reasons.

## A collection session at a glance

One session = one place, one table setup, one sitting: **one mapping clip, one gripper calibration clip, then the
demos**, recorded in that order on the Pi (UMI's order). Each clip is one ordinary Pi episode (start, record, stop),
so they come out as consecutive `episode_NNNNNN` folders. Write down which number is which as you go:

```
episode_000130   map
episode_000131   gripper calibration
episode_000132 … episode_000140   demos
```

The lens calibration (section 1) is not part of a session: do it once, and again only when the lens changes.

### Before the session

- **Table tag:** print `table_marker.pdf` (ArUco `DICT_4X4_50` id 13, 160 mm), glue it flat onto something rigid,
  and tape it **flat on the table** inside the workspace, where the demos happen. Never on a wall or a monitor, never
  bowed. It must not move while you record the mapping clip. It is only needed for the mapping clip: SLAM can start
  from it, and `map` measures the table's position and level from it (check M5). **After the mapping clip, remove it**
  (UMI's way), carefully, without moving anything else, so it is not in the demos' images: the robot will not have
  it when it runs the policy. The gripper calibration and the demos are recorded without it.
- **Scene:** set up everything the demos need (objects, clutter, lighting) before the map, and don't move anything
  that isn't part of the task until the session ends. Good, even light; something textured in view (objects, edges,
  posters), not just a blank wall or a bare tabletop.
- **Rig:** lens focus taped (section 1), gripper fingers moving freely, both finger tags clean and fully visible.

### 1. The mapping clip (1–2 minutes)

The map is what every demo is located in. A demo can only be tracked where the map has seen the scene from a similar
place and angle.

1. Start recording with the gripper held **30–50 cm above the table tag, looking at it**. Hold still about 3 s.
2. Make **small, slow motions over the tag** for about 5 s: up and down, side to side, a little tilt. (This starts
   SLAM and the IMU off cleanly.)
3. **Scan the workspace slowly**: go everywhere the demos will go, from the heights and angles the demos will use
   (approach, grasp, lift, place), and add a few wider views of the surroundings. Mix gentle rotations with real
   sideways and up/down movement; don't just spin in place.
4. Keep it **continuous for at least 30 s**: no fast swings, no hand over the lens, no pointing at a blank wall or
   right down at the bare table. (ORB-SLAM3's last IMU refinement needs about 15 s of unbroken tracking.)
5. Keep the gripper fingers still during the map. Stop recording.

### 2. The gripper calibration clip (about 15–20 s)

It measures the finger opening from the two finger tags.

1. Start recording with the gripper pointed at the scene (anything; no SLAM is involved).
2. **Close the fingers fully, then open them fully, slowly**, 3–4 times. "Fully open" is the most the gripper will
   open in the demos (the robot's limit is 80 mm).
3. **Both finger tags must be visible the whole time** (UMI's rule: both seen in at least 80 % of the frames): keep
   your hand and objects off them. Stop recording.

### 3. The demos (one recording each)

1. Start each recording with the gripper **already looking at the mapped workspace**, and **hold still about 1 s**
   before moving: the demo must find itself in the map within the first 0.5 s.
2. Do the task once, at a natural pace. Avoid fast swings and covering the lens: if SLAM loses track for more than
   about 0.17 s (5 frames), the recording is split at the gap or dropped (UMI's rule, docs/DESIGN.md "Checks").
3. Stay inside the area the map covered, and keep both finger tags visible as much as you can.
4. Stop recording; reset the scene; next demo. Start with 1–3 demos for the first session.

### Process the session

On the Jetson (recordings in `/data/incoming`). On the laptop, add `--source ~/Downloads/data/incoming` to each
command. `TASK` is the task's name (e.g. `block_raise`).

| Order | Command | Result | Status |
|---|---|---|---|
| 1 | `umiproc map TASK episode_000130` | builds and checks the map (section 3); on success creates the session's folder `~/data/umi/tasks/TASK` (or `TASK_2`, … if taken) and prints its name | being built (step 2) |
| 2 | `umiproc calibrate TASK_FOLDER episode_000131` | the gripper opening range for the folder (section 5) | coming with step 3 |
| 3 | `umiproc run TASK_FOLDER episode_000132 episode_000140` | every demo in the range localized in the map and checked: `episode_N` folders, or `failed/<name>/` with the reason (section 4) | coming with step 3 |
| any time | `umiproc status TASK_FOLDER` | what the folder has: map, gripper calibration, episodes, waiting, failed | available |

- `TASK_FOLDER` is the folder name `map` printed (`TASK`, or `TASK_2` …).
- `run` skips the map and calibration recordings if the range includes them.
- Without step 2 (gripper calibration), `run` still localizes and checks every demo, then leaves them waiting for
  the gripper calibration; `calibrate` + `run` again finishes them.
- If `map` fails, nothing is created: read its message (section 3), record a new mapping clip, and run `map` on it.
  Demos recorded after a failed map can still be used once a later map of the same setup passes, as long as nothing
  in the scene moved.
- You can record sessions before the commands exist: the recordings simply wait in `/data/incoming`.

## 3. Prove the environment: the map

**Why:** every session starts with one mapping clip. SLAM builds a 3-D map of the workspace from it, and every demo of
the session is later located inside that map (UMI's recipe). `umiproc map` builds the map, checks it (M1–M5 below)
and, only when every check passes, creates the session's task folder. A bad map would make every demo of the session
bad, so a failed map creates nothing: you record (or re-run) until it passes.

**How to record the mapping clip** (table tag, scene, the still start above the tag, the scan): see "A collection
session at a glance" above, step 1. What matters most for the checks below: the still start 30–50 cm above tag 13,
then small slow motions; at least 30 s of continuous tracking (VIBA 2, check M2, needs about 15 s of it); real
translation, not only rotation; no fast moves or blank views; nothing in the scene moved; fingers still. If you do
the lens calibration (section 1), do it before the map: a map records the lens it was made with.

### Run it (on the Jetson)

```bash
ls -t /data/incoming | head -3                 # newest first: the mapping clip, e.g. episode_000123
umiproc map block_raise episode_000123
```

- `block_raise` is the session's task label. A pass creates `~/data/umi/tasks/block_raise/` (or `block_raise_2`,
  `block_raise_3`, … when that name is taken: one folder per session) and prints it: `map: PASSED -> tasks/<folder>`.
  That folder name is what `calibrate` and `run` take.
- On the laptop, add `--source ~/Downloads/data/incoming`. `--root DIR` (or `UMIPROC_ROOT`) chooses another data
  root (see the A/B comparison below); `--rig` another rig (default `rig_pi_v3`).
- It takes about **3 times the clip's length** (the frames are converted, then SLAM runs twice in real time: the
  mapping run, then the same clip located again against the saved map, as UMI does). Run it inside `tmux` on the
  Jetson, so a dropped ssh connection does not kill it: `tmux new -s map`, run the command, detach with `Ctrl-b d`,
  come back with `tmux attach -t map`.
- Disk: while it runs it needs room for the converted copy of the recording (`~/data/umi/incoming/<pi_name>`, about
  the recording's size) and the frames SLAM reads (`~/data/umi/.work/map/<pi_name>/euroc`, about 400 KB per frame:
  ~1.4 GB for a 2-minute clip). Both are removed at the end.
- The first lines say which SLAM build (and its patches), which IMU setting (`vio_inflation`) and which lens
  (`calibrated` or `nominal, uncalibrated`) it uses, and the settings' fingerprint.
- It ends with one line per check (`PASS`, `FAIL`, or `n/a` when an earlier failure made it impossible to judge),
  the diagnostics, and where the report is.

Exit code 0: every check passed. 1: a check failed. 2: it could not run on this recording (the message says why:
not there or still syncing, already taken, not this rig's camera/IMU settings, rig files or SLAM build missing,
another umiproc command running).

**When it passes:** `~/data/umi/tasks/block_raise/map/<pi_name>/` holds the recording (`raw/`), the map
(`map_atlas.osa`), the table tag's pose in the map (`tx_slam_tag.json`), both SLAM runs (`mapping/`,
`relocalize/`: trajectory, log), the settings, mask and fingerprint used, the tag detections, and `report.json`
(every check's values and limits, the diagnostics). Next: the gripper calibration (section 5).

**When it fails:** nothing is created and the recording is not marked as taken, so it can be run again. The report
and the SLAM logs are kept in `~/data/umi/maps_failed/<pi_name>/<UTC time>/` (the last lines printed give the
folder; `umiproc status` lists the recent ones). Each failed check prints **what to change**: usually record a new
mapping clip following the hints, then run `umiproc map` on the new one. If every check passed but intake refused the
recording (exit 2, "intake did not take"), the map, its report and the logs are kept in `maps_failed/` too.

### An interrupted map (killed, power lost, ssh dropped without tmux)

- **Killed while it ran** (the usual case: during the SLAM runs): nothing was taken, so the same clip can be run
  again. The next command may stop on a leftover `~/data/umi/incoming/<pi_name>` (`umiproc status` shows "incoming/
  is not empty"): it is only the converted copy (the Pi's recording in `/data/incoming` is untouched), safe to
  delete: `rm -rf ~/data/umi/incoming/<pi_name>`. A leftover `~/data/umi/.work/map/` is cleared by the next `map`.
- **Killed in the second it takes to attach a passed map**: `umiproc status <folder>` shows
  `map/<pi_name>: INCOMPLETE`. The clip is now recorded as taken, so `umiproc map` on it again stops with "already
  taken": **record a new mapping clip** (it creates a new session folder) and use that folder. If status also says
  "its map is still in `~/data/umi/.work/map/<pi_name>`", the finished map can be moved in by hand instead, **before
  any other `umiproc map`** (the next one clears `.work/map/`):

  ```bash
  W=~/data/umi/.work/map/<pi_name>
  S=~/data/umi/tasks/<folder>/map/<pi_name>          # the INCOMPLETE slot status names
  grep '"folder"' $W/report.json                      # must be <folder>
  for f in map_atlas.osa orbslam3_settings.yaml slam_mask.png slam_fingerprint.json mapping relocalize \
           tag_detections.json tx_slam_tag.json; do [ -e $W/$f ] && [ ! -e $S/$f ] && mv $W/$f $S/; done
  mv $W/report.json $S/report.json                    # last: report.json marks the map complete
  umiproc status <folder>                             # map passed
  ```

  Ctrl-C cleans up after itself (nothing is left in `.work/`), so after a Ctrl-C at that moment only the new clip
  helps.

### Reading the checks

| Check | Passes when | If it fails, change this in the recording |
|---|---|---|
| M1 one map | SLAM ends with exactly one map that has keyframes (it never lost the map and started over) | "never initialised": the tag fully in view and sharp at the start, more light, more texture. "lost every map" or "2 maps": the start still above the tag, then slow motions; no fast moves, no blank views; keep it tracked from start to end. "crashed (SIGSEGV, exit -11)" or "did not finish": not the recording; send the mapping log. A hint naming the **camera model** (the map was reset ≥ 5 times, most new maps lost within 0.5 s of starting): not the recording either; calibrate the lens (section 1) and check the rig file's camera–IMU geometry (`T_imu_cam`) |
| M2 IMU initialised | the IMU is initialised and both IMU refinements finished (VIBA 1 after ~5 s, VIBA 2 after ~15 s of tracking) | record longer (≥ 30 s), with continuous tracking and real translation in every direction (up/down, side to side, forward/back), not only rotation. If the hint names the **camera model** (as in M1), a longer clip will not help: lens calibration (section 1), camera–IMU geometry |
| M3 re-localizes | the same clip, located against the saved map: found within the first 0.5 s, and ≥ 95 % of all its frames tracked | the start of the clip on the tag area (still); no fast moves or blank views (the message gives the time of the longest loss); nothing moved during the clip |
| M4 gravity, orientation, scale, speed | the copied gravity check (the measured gravity is 9.81 ± 0.5 m/s²) and metric-scale check (scale within 10 %) pass on that trajectory; the SLAM orientation agrees with gravity throughout: gravity as the IMU sees it, turned into the map by SLAM's orientation and averaged over each 2.5 s of the tracked clip, stays within 3° of its average over the whole clip (at least 4 such stretches, i.e. ≥ 10 s tracked); the camera never moves faster than 3 m/s | scale "could not be measured" or "off": more translation with changes of direction (slow up/down and side-to-side strokes of 20–40 cm), especially right after the still start. Orientation off by more than 3° on a smooth clip: not the recording; the lens model (section 1) or the camera–IMU rotation (`T_imu_cam`), or a map that bends after lost tracking (M3). "not enough tracked span": keep the clip tracked (M3). Speed: move slower |
| M5 table tag | ≥ 20 keyframes see tag 13 at 0.3–2 m and near the image centre (UMI's filters); its position in the map varies ≤ 1 cm along each axis (standard deviation, as UMI prints it); the median of its views lies within 5° of level | the tag in the middle of the view for the first seconds and again during the scan; between 30 cm and 2 m away; taped flat and never moved. A spread or tilt failure with a flat, still tag means the map itself is off (see M3/M4) |

The diagnostics are reported, not checked:

- **pacing**: whether SLAM kept up with the camera. `fed 24.6 of 30.0 frames/s` and a `WARNING: … slower than real
  time` mean the computer was too slow (or busy): ORB-SLAM3 then skips keyframes and the map can be worse than the
  clip allows. Run with nothing else heavy running (the Jetson in its maximum power mode).
- **camera–IMU time offset estimate**: umiproc assumes 0 ms; a large estimate here is something to look into, not a
  failure.
- **keyframes in the map**, **tag frames** (in how many frames tag 13 was seen at all), **lens**, **IMU
  vio_inflation**, **SLAM build**.

### "Out of date"

Every map stores a fingerprint of what it was made with: the rig file, SLAM config and lens file in
`~/data/umi/rigs/`, the gripper mask, the ORB-SLAM3 settings, and the SLAM program (its file, its patches, its
library). `umiproc status` recomputes it from what is there **now**:

```bash
umiproc status                      # one line per task folder: "map passed" or "map passed, OUT OF DATE"
umiproc status block_raise          # the map's checks, date, fingerprint, and what differs when out of date
```

**OUT OF DATE** means something changed since the map was made (you edited a file in `rigs/`, applied a new lens
calibration, rebuilt the SLAM program with other patches, or are running with another `UMIPROC_SLAM_BINARY`). The map
and every demo localized against it would no longer match the current settings: record a new mapping clip (a new
session folder), then process the demos again. It also shows when the environment differs from the map's run: an A/B
variant root checked without its `UMIPROC_SLAM_BINARY` says OUT OF DATE, which is the point.

### The A/B comparison on the first real session

The first mapping clip is run in three variants (docs/DESIGN.md "SLAM"): all fixes (patches 0001–0009) with the
datasheet IMU noise; the 0001–0006 build; and the IMU noise ×10. Each variant gets **its own data root**, so a
variant never mixes with another (a root holds one variant's maps and nothing else) and the same clip can be taken in
each. `/data/incoming` is only read, so all three read the same clip.

Once, build the 0001–0006 variant (section 2.1; 20–30 min on the Jetson):

```bash
PATCHES_UPTO=0006 <repo>/umi-processing/native/build.sh
```

Then set up the three roots with this shell function (paste it into the terminal; `<repo>` as in section 2). It
makes a new root `$AB/NAME` whose `rigs/` holds the rig file and SLAM config the processing reads in `~/data/umi/rigs/`
(the repo's when `~/data/umi` has none yet) and the lens file when `~/data/umi/rigs/` has one, sets the IMU noise factor,
checks it, and stops loudly (`STOP: …`) on anything unexpected:

```bash
AB=~/data/umi_ab
REPO=<repo>/umi-processing
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

setup_variant p0009_infl1  ~/ORB_SLAM3_umi/umiproc_build/umiproc_slam      1.0
setup_variant p0006_infl1  ~/ORB_SLAM3_umi/umiproc_build_0006/umiproc_slam 1.0
setup_variant p0009_infl10 ~/ORB_SLAM3_umi/umiproc_build/umiproc_slam      10.0
```

Each must end with its `NAME: SLAM binary … (9 patches)` (6 for `p0006_infl1`) line and no `STOP`. Then run the
three, with the clip's name (here `episode_000123`). Commands 1 and 3 remove `UMIPROC_SLAM_BINARY`, so a value left
over in the terminal cannot silently run them with the 0006 build:

```bash
# 1. all patches, datasheet IMU noise (the default build)
env -u UMIPROC_SLAM_BINARY UMIPROC_ROOT=$AB/p0009_infl1 \
    umiproc map block_raise episode_000123 2>&1 | tee $AB/p0009_infl1.txt
# 2. patches 0001-0006
UMIPROC_ROOT=$AB/p0006_infl1 UMIPROC_SLAM_BINARY=$HOME/ORB_SLAM3_umi/umiproc_build_0006/umiproc_slam \
    umiproc map block_raise episode_000123 2>&1 | tee $AB/p0006_infl1.txt
# 3. all patches, IMU noise x10
env -u UMIPROC_SLAM_BINARY UMIPROC_ROOT=$AB/p0009_infl10 \
    umiproc map block_raise episode_000123 2>&1 | tee $AB/p0009_infl10.txt
```

- Check the first lines of each output. The one starting `map: <pi_name>: SLAM build` reads
  `SLAM build /home/<you>/ORB_SLAM3_umi/umiproc_build/umiproc_slam (patches 0001-0009 (9 patches)); IMU vio_inflation
  noise_density 1, random_walk 1; lens nominal, uncalibrated; fingerprint …` for 1;
  `…/umiproc_build_0006/umiproc_slam (patches 0001-0006 (6 patches))` for 2; `noise_density 10, random_walk 10`
  only for 3; the same `lens …` in all three (`lens calibrated` when a lens file was copied). The diagnostics at the
  end repeat them (`SLAM build: …`, `IMU vio_inflation: noise_density 10.0, random_walk 10.0`), and every
  `report.json` records the same (`diagnostics.build`, `diagnostics.vio_inflation`, and the full fingerprint in
  `slam_fingerprint.json`).
- Patches 0007–0009 act only after a later relocalization (docs/DESIGN.md "SLAM"). The mapping clip is relocalized
  at its first frame, so it may not tell 0006 from 0009 apart; localizing the demos against each variant's map (step
  3, `umiproc run`) will.
- A 4th possible variant is the lens: a root whose `rigs/` has a lens file (`rig_pi_v3.lens.yaml`) while the others
  have none (or another one). Set it up as above, then copy the lens in, e.g. from a lens calibration's result:
  `cp ~/data/umi/lens/<pi_name>/lens.yaml $AB/<variant>/rigs/rig_pi_v3.lens.yaml`; its output then says
  `lens calibrated`. (One was already run this way: `~/data/umi_ab/lens126`, the lens from clip 126.)
- Compare the M2, M3 and M4 lines of the three outputs (`$AB/*.txt`): which variant finishes VIBA 2, re-localizes
  sooner and tracks more frames, and has the scale closest to 1. SLAM differs a little from run to run, so a small
  difference means nothing; run a variant twice if two are close (a failed variant can be run again as it is; a
  passed one says "already taken": delete its root, `rm -rf $AB/<variant>`, and run its `setup_variant` line again).
- The variant roots are only for this comparison. Afterwards, tell me (or the next session) which variant won; the
  real session is then processed in `~/data/umi` with that setting.
- Remove them when done: `rm -rf ~/data/umi_ab`.

## 4. Prove the demos

(Coming with step 3.)

## 5. Gripper calibration and finishing the episodes

(Coming with step 3.)
