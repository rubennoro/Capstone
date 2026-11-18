# umiproc: directions

Step-by-step procedures for the Pi 5 rig, in the order you do them. Commands run on the Jetson unless a section says
otherwise.

| Section | When | What |
|---|---|---|
| [0. Jetson setup](#0-jetson-setup) | once per Jetson | the code, the Python environment, the SLAM build, the data root, a check |
| [1. Lens calibration](#1-lens-calibration) | once per lens (again only when the lens changes) | measure the ultrawide lens; SLAM fails without it |
| [2. A collection session](#2-a-collection-session-the-order) | every session | the order of the recordings, what to prepare |
| [3. Verify the map with task demos](#3-verify-the-map-with-task-demos) | start of every session | mapping clip → `umiproc map` → 2–3 task demos → `umiproc run`: proves the map works for this task before you record the rest |
| [4. Gripper calibration](#4-gripper-calibration) | every session, once section 3 is verified | the finger opening range |
| [5. The rest of the demos](#5-the-rest-of-the-demos-and-finishing) | every session | record them, `umiproc run`, check the result |
| [Appendix A](#appendix-a-the-ab-comparison-first-real-session-only) | first real session only | compare SLAM builds and IMU settings on the first mapping clip |

## Start here: bringing the Jetson up, step by step

Do these in order; each step has a check that tells you it passed before you go on.

| # | Do | Where | Passed when |
|---|---|---|---|
| 1 | Get the code (0.1) | Jetson | `git log --oneline -1` shows the newest umi-processing commit |
| 2 | Python environment (0.2, about 5 min) | Jetson | `umiproc --help` lists `lens-board`, `lens-calibrate`, …, and the check tests pass |
| 3 | Start the SLAM build (0.3, about an hour) **inside `tmux`**, and let it run while you do step 4 | Jetson | it ends with `== done`, and the slow check passes |
| 4 | **Lens calibration** (section 1): print the board, glue it to something rigid, record about 60 s on the Pi, run `umiproc lens-calibrate ... --apply` (there is no pass/fail: any fit is installed, with warnings) | Pi + Jetson | `--apply` prints `applied` and installs `~/data/umi/rigs/rig_pi_v3.lens.yaml` |
| 5 | `umiproc status` (0.5) | Jetson | lens file: calibrated; umiproc_slam and vocabulary: found |
| 6 | A collection session (sections 2–5): verify the map with a few task demos, then gripper calibration, then the rest of the demos | Pi + Jetson | `umiproc status <folder>` lists the episodes |

**Do step 4 before any mapping.** The lens model matters more than anything else for SLAM on this rig: on the first
real mapping clip (episode 128) the rig file's nominal lens, about 18 % off in focal length, made SLAM lose every map
it started (34 resets, no usable map), while the same clip with a measured lens built a map with the IMU fully
initialised. Lens calibration needs only step 2, not the SLAM build.

`tmux` in one line: `tmux new -s build`, run the command, detach with `Ctrl-b` then `d`, come back with
`tmux attach -t build`. A long command keeps running even if your ssh connection drops.

## 0. Jetson setup

Once per Jetson. `<repo>` below is the Jetson's clone of the Capstone repo. Nothing here touches dataproc's setup: its
`~/ORB_SLAM3`, Pangolin in `/usr/local`, `~/envs/env_dataproc` and `~/data` stay as they are.

### 0.1 Get the code

```bash
cd <repo> && git fetch && git checkout rnoro/data-work && git pull
git log --oneline -1                                # the newest umi-processing commit
```

### 0.2 Python environment (about 5 minutes)

Every `umiproc` command, including the lens calibration in section 1, runs in this environment. It is all section 1
needs; the SLAM build (0.3) is only needed from the map on.

umiproc has its own environment and never imports dataproc, so don't install it into `~/envs/env_dataproc`. First
make sure Python can make environments:

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

### 0.3 Build the SLAM program (about an hour the first time)

umiproc uses UMI's own ORB-SLAM3 fork with our fixes (docs/DESIGN.md "SLAM"), built in `~/ORB_SLAM3_umi`. Run it
inside `tmux` (see "Start here") and do the lens calibration (section 1) while it builds:

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
cd <repo>/umi-processing     # with the environment from 0.2 turned on
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_umiproc_slam_inputs.py -m slow
```

It must say `passed`, not `skipped` (skipped means the SLAM program or vocabulary was not found: see above).

### 0.4 The data root

Nothing to do here; this is where things go.

- umiproc writes everything under **`~/data/umi/`**; the first command that takes a recording creates it. Leave
  `UMIPROC_ROOT` unset.
- `/data` is refused (its `incoming/` is the Pi's), and so is dataproc's root (`~/data` itself, `$DATAPROC_ROOT`), so
  dataproc's data is never touched. `/data/incoming` is only read.
- `~/data/umi/rigs/` holds the files the processing reads:
  - `rig_pi_v3.yaml`: copy of `configs/rigs/rig_pi_v3.yaml` (cameras, IMU, CAD geometry);
  - `rig_pi_v3.slam.yaml`: copy of `configs/slam/rig_pi_v3.yaml` (gripper mask, table tag, ORB settings);
  - `rig_pi_v3.lens.yaml`: from section 1.
- The first `map` copies the first two. To change something (for example a re-traced mask), edit the copy there, not
  the repo; `umiproc status` shows where a copy differs from the repo.
- Every map and episode records the values it was made with, so changing them makes the maps and episodes out of
  date.

### 0.5 Check: `umiproc status`

```bash
umiproc status
```

On a fresh Jetson after 0.2 and 0.3 it shows:

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

After the lens calibration's `--apply` (section 1), `lens file:` says `calibrated` with its date and recording. That
is the state to be in before the first session.

- `NOT FOUND` next to umiproc_slam or vocabulary: run the native build (0.3).
- `refused`: `UMIPROC_ROOT` points at `/data` or `~/data`; unset it.
- `PROBLEM:`: the line names the file and what to change.
- Later, in a session, it also shows how many recordings were taken (map / calibration / demonstration) and one line
  per task folder; `umiproc status block_raise` shows one folder: its map, gripper calibration, episodes, the
  recordings waiting in `.staging/` and why, and the failed ones with their reasons.

## 1. Lens calibration

**Why:** it measures how the ultrawide (IMX219, 160°) lens bends light, so SLAM uses the real lens instead of the
rig file's nominal guess. `map` and `run` never wait for it (without it they use the nominal lens, labelled
"nominal, uncalibrated"), **but do it first**: the nominal lens is about 18 % off in focal length, which is enough
to make SLAM fail (see "Start here"). About 1 minute of recording, then 1–2 minutes of processing.

**What you need:** a printer, something rigid and flat to glue the board to (foam board, a clipboard, a flat piece
of wood or acrylic), glue (spray adhesive or a glue stick), a ruler, a small piece of tape, and a well-lit spot.
The commands need the Python environment from 0.2.

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

With the environment from 0.2 turned on (the SLAM build is not needed), find the recording's directory name (the
newest one in `/data/incoming`) and run the calibration on it:

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
- `lens.yaml`: the measured lens. Written whenever there is a fit.

**There is no pass/fail** (as in UMI, which has no lens step at all): every check below only **warns**. A check
outside its limit prints `WARN` with what to record differently, and is listed in `report.json`'s `warnings` and in
`lens.yaml` (`metrics.checks_outside_limits`; `umiproc status` shows them). The lens is still written and `--apply`
still installs it. The map's checks and the test demos (section 3) then show whether it is good enough; a better
recording later simply replaces it. Exit code 0 when a lens was written, 1 only when there is **no fit at all**
(the board was too rarely seen: `NO LENS`).

### Reading the checks

| Check (all warnings) | What it means | If it warns, to improve it |
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

Whenever there is a fit (warnings or not), the last lines show the command that installs it, which is the same
command plus `--apply`. Or run with `--apply` straight away:

```bash
umiproc lens-calibrate episode_000123 --apply
```

It repeats the calibration (same result) and copies `lens.yaml` to `~/data/umi/rigs/rig_pi_v3.lens.yaml`, the lens
file the processing reads. A lens file already there is kept as `rig_pi_v3.lens.yaml.bak-<UTC time>`. `--apply`
always installs the fit, whatever the warnings; it does nothing only when there is no fit. Apply it before you record a session's map, not in the middle of a session: maps
and episodes record the lens they were made with. Then `umiproc status` (0.5) shows `lens file: … calibrated`.

### When to redo it

- The lens was refocused or turned (the tape moved).
- The camera was taken off or remounted, or the lens was swapped.
- The camera mode changed (resolution or sensor mode of the IMX219).

## 2. A collection session: the order

One session = one place, one table setup, one sitting. It starts by **proving that the map works for this task**:
a mapping clip, then a few real task demos localized in it. Only when those come out right do you record the gripper
calibration and the rest of the demos. A bad map makes every demo of the session bad, and the only real test of a
map is whether demos find themselves in it, so a few minutes of checking up front saves a whole session.

Each clip is one ordinary Pi episode (start, record, stop), so they come out as consecutive `episode_NNNNNN` folders.
**Write down which number is which as you go**:

```
episode_000130                     mapping clip (table tag on the table)
                                   -> umiproc map: must pass; then remove the table tag        (section 3)
episode_000131 … episode_000133    2-3 test demos (real demos of the task)
                                   -> umiproc run: must come out "waiting for the gripper calibration"  (section 3)
episode_000134                     gripper calibration clip
                                   -> umiproc calibrate                                          (section 4)
episode_000135 … episode_000150    the rest of the demos
                                   -> umiproc run: also finishes the test demos                  (section 5)
```

The test demos are ordinary demos: nothing is thrown away. They wait, already localized and checked, until the
gripper calibration is in, and then become episodes like the others.

The lens calibration (section 1) is not part of a session: do it once, and again only when the lens changes.

### The commands, in order

On the Jetson (recordings in `/data/incoming`). On the laptop, add `--source ~/Downloads/data/incoming` to each
command. `TASK` is the task's name (e.g. `block_raise`).

| Order | Command | Result |
|---|---|---|
| 1 | `umiproc map TASK episode_000130` | builds and checks the map (section 3); on success creates the session's folder `~/data/umi/tasks/TASK` (or `TASK_2`, … if taken) and prints its name |
| 2 | `umiproc run TASK_FOLDER episode_000131 episode_000133` | the test demos localized in the map and checked; each one **waits** for the gripper calibration (exit 3) or goes to `failed/` with the reason (section 3) |
| 3 | `umiproc calibrate TASK_FOLDER episode_000134` | the gripper opening range for the folder (section 4) |
| 4 | `umiproc run TASK_FOLDER episode_000135 episode_000150` | the rest of the demos: `episode_N` folders, or `failed/<name>/` with the reason; the waiting test demos are finished too (section 5) |
| any time | `umiproc status TASK_FOLDER` | what the folder has: map, gripper calibration, episodes, waiting, failed |

- `TASK_FOLDER` is the folder name `map` printed (`TASK`, or `TASK_2` …).
- **Give `run` only demo numbers.** It skips the mapping and calibration clips once `map` and `calibrate` have taken
  them, but a clip nobody has taken yet looks like a demo to it. Running `calibrate` (3) before the next `run` (4), as
  above, keeps that from happening.
- If `map` fails, nothing is created: read its message (section 3), record a new mapping clip, and run `map` on it.

### Before the session

- **Table tag:** print `table_marker.pdf` (ArUco `DICT_4X4_50` id 13, 160 mm), glue it flat onto something rigid,
  and tape it **flat on the table** inside the workspace, where the demos happen. Never on a wall or a monitor, never
  bowed. It must not move while you record the mapping clip. It is only needed for the mapping clip: SLAM can start
  from it, and `map` measures the table's position and level from it (check M5). **Once the map passed, remove it**
  (UMI's way), carefully, without moving anything else, so it is not in the demos' images: the robot will not have
  it when it runs the policy. The test demos, the gripper calibration and the rest of the demos are recorded without
  it.
- **Scene:** set up everything the demos need (objects, clutter, lighting) before the map, and don't move anything
  that isn't part of the task until the session ends. Good, even light; something textured in view (objects, edges,
  posters), not just a blank wall or a bare tabletop.
- **Rig:** lens calibrated (section 1) and its focus taped, gripper fingers moving freely, both finger tags clean and
  fully visible.

## 3. Verify the map with task demos

**Why:** every demo of the session is located inside the session's map (UMI's recipe), so all demos come out in the
same coordinates: the table's, fixed by tag 13 in the mapping clip. `umiproc map` builds the map and checks it on its
own clip (M1–M5); the 2–3 test demos then show that demos of *this task*, recorded without the table tag, find
themselves in it and pass the pose checks. Both have to come out right before you record the rest of the session.

The steps: **3.1** record the mapping clip, **3.2** run `umiproc map` (leave the scene and the tag alone until it
passes), **3.3** remove the tag and record 2–3 test demos, **3.4** run `umiproc run` on them, **3.5** decide whether
the map is verified. The rest of this section is reference for reading the results.

### 3.1 Record the mapping clip (1–2 minutes)

A demo can only be tracked where the map has seen the scene from a similar place and angle.

1. Start recording with the gripper held **30–50 cm above the table tag, looking at it**. Hold still about 3 s.
2. Make **small, slow motions over the tag** for about 5 s: up and down, side to side, a little tilt. (This starts
   SLAM and the IMU off cleanly.)
3. **Scan the workspace slowly**: go everywhere the demos will go, from the heights and angles the demos will use
   (approach, grasp, lift, place), **including where the demos start**, and add a few wider views of the
   surroundings. Mix gentle rotations with real sideways and up/down movement; don't just spin in place.
4. Keep it **continuous for at least 30 s**: no fast swings, no hand over the lens, no pointing at a blank wall or
   right down at the bare table. (ORB-SLAM3's last IMU refinement needs about 15 s of unbroken tracking, check M2.)
5. Keep the gripper fingers still during the map. Stop recording.

**Leave the table tag and the scene exactly as they are** until the map has passed (3.2): if it fails, you record
another mapping clip of the same setup.

### 3.2 Run the map (on the Jetson)

```bash
ls -t /data/incoming | head -3                 # newest first: the mapping clip, e.g. episode_000130
umiproc map block_raise episode_000130
```

- `block_raise` is the session's task label. A pass creates `~/data/umi/tasks/block_raise/` (or `block_raise_2`,
  `block_raise_3`, … when that name is taken: one folder per session) and prints it: `map: PASSED -> tasks/<folder>`.
  That folder name is what `run` and `calibrate` take.
- On the laptop, add `--source ~/Downloads/data/incoming`. `--root DIR` (or `UMIPROC_ROOT`) chooses another data
  root (see Appendix A); `--rig` another rig (default `rig_pi_v3`).
- It takes about **3 times the clip's length** (the frames are converted, then SLAM runs twice in real time: the
  mapping run, then the same clip located again against the saved map, as UMI does). Run it inside `tmux` on the
  Jetson, so a dropped ssh connection does not kill it: `tmux new -s map`, run the command, detach with `Ctrl-b d`,
  come back with `tmux attach -t map`.
- Disk: while it runs it needs room for the converted copy of the recording (`~/data/umi/incoming/<pi_name>`, about
  the recording's size) and the frames SLAM reads (`~/data/umi/.work/map/<pi_name>/euroc`, about 400 KB per frame:
  ~1.4 GB for a 2-minute clip). Both are removed at the end.
- The first lines say which SLAM build (and its patches), which IMU setting (`vio_inflation`) and which lens
  (`calibrated` or `nominal, uncalibrated`) it uses, and the settings' fingerprint. **Check that it says
  `lens calibrated`.**
- It ends with one line per check (`PASS`, `WARN`, `FAIL`, or `n/a` when an earlier failure made it impossible to
  judge), the diagnostics, and where the report is.
- **No quality check stops the map.** M2–M5 only **warn**: each is printed with how to improve it and listed in
  `report.json`'s `warnings`, and the folder is created anyway, as UMI does (it judges no map). The test demos
  (3.3–3.5) are the real check.
- **Several maps are handled by themselves.** If SLAM lost tracking and started a new map, `map` picks the longest
  map piece, cuts the clip to it and maps and checks that part again (see below). No flag needed.
- It only stops when there is nothing to attach: no saved map at all, or several maps even after that cut (M1).

Exit code 0: the map was attached (possibly with warnings). 1: M1 failed (no single map). 2: it could not run on this recording (the message says why:
not there or still syncing, already taken, not this rig's camera/IMU settings, rig files or SLAM build missing,
another umiproc command running).

**When it passes:** `~/data/umi/tasks/block_raise/map/<pi_name>/` holds the recording (`raw/`), the map
(`map_atlas.osa`), the table tag's pose in the map (`tx_slam_tag.json`; missing when M5 saw the tag in no keyframe,
and `umiproc run` then refuses the folder: map again with the tag in view), both SLAM runs (`mapping/` with the
map's points, `relocalize/`: trajectory, log; `mapping_full/` when the clip was cut), the settings, mask and fingerprint used, the tag detections, and `report.json`
(every check's values and limits, the diagnostics). **Next: remove the table tag (carefully, nothing else moves) and
go on to 3.3.**

**When it fails** (M1): nothing is created and the recording is not marked as taken, so it can be run again. The report
and the SLAM logs are kept in `~/data/umi/maps_failed/<pi_name>/<UTC time>/` (the last lines printed give the
folder; `umiproc status` lists the recent ones). The failed check prints **what to change** (see 3.6): usually
record a new mapping clip following the hints (the tag is still in place), then run `umiproc map` on the new one. If
the map passed but intake refused the recording (exit 2, "intake did not take"), the map, its report and the logs
are kept in `maps_failed/` too.

**Several maps: the longest one is picked by itself.** When SLAM loses tracking partway, it starts a new map, so the
whole clip ends with several maps. Plain `umiproc map block_raise episode_000130` then picks the **longest map
piece** (the one that tracked the most frames), cuts the clip to it (from where SLAM started that map to its last
tracked frame, found from the run, not a fixed time), and runs the mapping, the re-localization and every check again
on that part. The cut is printed (`longest map (automatic): ... checked only the longest map piece: N of M frames,
a-b s`) and kept in `report.json` (`longest_map`); the whole clip's run is kept in `mapping_full/`. It then takes
about 5 times the clip's length. The demos must stay in the part of the workspace the cut clip saw. (`--longest-map`
names this default; it is never needed.)

To keep the **first** map piece instead (the clip from its start, with the table tag, up to where tracking was first
lost), add `--first-map`:

```bash
umiproc map block_raise episode_000130 --first-map
```

**Looking at a run.** `map` only runs SLAM and the checks; it draws nothing. Its mapping run(s) store the map's
points (`mapping/map_points.csv` + `_keyframes.csv` + `_meta.json`; a SLAM build from before patch 0012 stores none,
`map` says so). To look at a folder's map afterwards, by the task folder's name (as `run` and `status` take it):

```bash
umiproc slam-video <folder>              # --run mapping (default) | relocalize | mapping_full | all
umiproc slam-3d <folder>                 # --run mapping (default) | mapping_full
```

Both write into `~/data/umi/tasks/<folder>/analysis/` and print the path. `slam-video` writes `slam_video_<run>.mp4`
(each fed frame with its tracking state, the path from above, the timeline; ~1 minute per minute of clip on the
laptop; play it with VLC) and `slam_run_<run>.png` (the tracking timeline, the path, the numbers); it reads the frames
from the map's `raw/` copy (`--recording DIR` for another). `slam-3d` writes `slam_3d.html` (`slam_3d_mapping_full.html`
for that run): open it in a browser (it needs internet once, for plotly). It shows the map's points (coloured by height
or by observations), its keyframes as small camera frustums and the camera path coloured by time, with a slider that
moves a camera along it; without stored points, the path only. Up is gravity once the IMU initialised. The axes are
metres only when M4's scale check passed.

A failed map has no task folder: give its run folder instead, and the output goes next to it, into
`maps_failed/<pi_name>/<UTC time>/analysis/`:

```bash
umiproc slam-video ~/data/umi/maps_failed/<pi_name>/<UTC time>/mapping
umiproc slam-3d ~/data/umi/maps_failed/<pi_name>/<UTC time>/mapping
```

(The recording then comes from `report.json`'s `source`; pass `--recording DIR` if it moved.)

**The analysis folder.** `lens-calibrate` copies its results to `~/data/analysis/lens_calibration/<episode>/` (another
folder: `export UMIPROC_ANALYSIS=~/Downloads/data/analysis`, the laptop). A map's figures, video and 3-D page are in
its task folder's `analysis/` (above); `map` no longer copies anything to `~/data/analysis`.

### 3.3 Record 2–3 test demos

With the table tag removed, record **2–3 real demos of the task**, exactly as you will record the rest: same start,
same task, same pace. If the demos start from different places, start each test demo from a different one.

1. Start each recording with the gripper **already looking at the mapped workspace**, and **hold still about 1 s**
   before moving: the demo must find itself in the map within the first 0.5 s.
2. Do the task once, at a natural pace. Avoid fast swings and covering the lens: if SLAM loses track for more than
   about 0.17 s (5 frames), the recording is split at the gap or dropped (UMI's rule, docs/DESIGN.md "Checks").
3. Stay inside the area the map covered, and keep both finger tags visible as much as you can.
4. Stop recording; reset the scene; next demo.

### 3.4 Run the test demos (on the Jetson)

```bash
ls -t /data/incoming | head                    # newest first: note the first and last test demo
umiproc run block_raise episode_000131 episode_000133
```

- `block_raise` is the session's folder (the name `umiproc map` printed: `block_raise`, `block_raise_2`, …).
- The range is every recording from the first to the last name, inclusive. The mapping clip is skipped by itself
  (`skipped (is the map recording …)`), and so is a recording another folder took, one still syncing, and one that is
  already an episode of this folder.
- On the laptop, add `--source ~/Downloads/data/incoming`. `--root DIR` (or `UMIPROC_ROOT`) chooses another data
  root.
- `--dry-run` checks the folder and its map and prints what each recording would get (`take`, `resume`, `skip` and
  why), changing nothing.
- It takes about **1.5 times the demos' length** (the frames are converted, then SLAM runs in real time). Use `tmux`
  on the Jetson for a long range.
- Disk: one demo's converted frames at a time (`~/data/umi/.work/run/`, about 400 KB per frame), removed after each.

The command first checks the folder's map: complete, passed, and **made with exactly the current settings** (the
rig file, SLAM config and lens file in `~/data/umi/rigs/`, the mask, the ORB-SLAM3 settings and the SLAM build: the
same fingerprint `umiproc status` compares). If anything changed since the map, it stops with `the map … is OUT OF
DATE` and the keys that differ (for example `sensors.imu` after editing `vio_inflation`, `sensors.ultrawide` after a
new lens calibration, `binary` / `patches` after a rebuild, or a different `UMIPROC_SLAM_BINARY`), and changes
nothing: put back what changed, or record a new session with the new settings (3.8).

Since there is no gripper calibration yet, each test demo is localized, split by UMI's lost-frame rules and
pose-checked, then **waits** in `.staging/` (steps 1–5 of 3.7). **Expected: exit code 3**, with one line per demo:

```
episode_000131  ->  waiting for the gripper calibration (1 piece(s) passed every pose check)
episode_000132  ->  waiting for the gripper calibration (1 piece(s) passed every pose check)
episode_000133  ->  failed (slam: late_relocalization)
```

Exit codes: 3: some wait for the gripper calibration (the expected result here). 0: every recording got its outcome
(here: all of them failed). 2: it could not run (no such folder, no usable map, map out of date, SLAM build missing,
bad range, `incoming/` in the data root not empty, another umiproc command running); nothing was changed. 1: an
unexpected error stopped it (the recordings done so far keep their outcome; run it again).

### 3.5 Is the map verified?

| Outcome of the test demos | What it means | Do |
|---|---|---|
| **every test demo waiting** for the gripper calibration | the demos find themselves in the map and pass the pose checks | **verified**: go on to section 4 |
| one failed for a reason in the recording (`speed`, `tag_coverage`, a fast swing that caused `too_many_lost`), the others waiting | the map is fine; that demo was not | record one more test demo with the hint in mind and run it; verified once the new one waits |
| `late_relocalization`, `too_many_lost`, `stopped_lost` or `new_map` on calm demos that started looking at the workspace | the map does not cover where the demos start or go | record a new mapping clip that covers the demos' start poses and paths (put the tag back), run `map` on it (a new session folder), then new test demos for the new folder (3.3–3.4) |
| `gravity`, `scale_mismatch` or `orientation` on smooth demos | not the recording: the map's scale or gravity, the lens model or the camera–IMU rotation (map check M4) | stop and look into it before recording more (read the map's M4 line, `umiproc status <folder>`) |
| `slam_error` | the SLAM run crashed or timed out: not the recording | read `~/data/umi/tasks/<folder>/failed/<pi_name>/processed/localize/umiproc_slam.log` |

Every reason is explained in 3.7. To look at one SLAM run of a failed demo as a figure (tracking over time, the path,
frames), `umiproc plot-run --csv <dir>/trajectory.csv --log <dir>/umiproc_slam.log` with
`<dir>` = `~/data/umi/tasks/<folder>/failed/<pi_name>/processed/localize`.

On the first real session, the A/B comparison (Appendix A) also runs on this mapping clip. It uses separate data
roots and only reads `/data/incoming`, so it can run after the session; it does not block it.

### 3.6 Reading the map checks

| Check | Passes when (M1 and M5's sightings block; the rest only warn) | If it fails or warns, change this in the recording |
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

### 3.7 What `run` does to each demo, and reading its outcome

1. **Taken** into `~/data/umi/tasks/<folder>/.staging/<pi_name>/` (and recorded in `pi_sources.csv`); the copied
   contract and camera checks run on it.
2. **Localized** in the map: ORB-SLAM3 with the gripper mask, the frames fed in real time, stopped after 30 frames it
   could not track (UMI's `--max_lost_frames`, 1 s).
3. **UMI's lost-frame rules** (converted to the same times at our 30 frames/s, docs/DESIGN.md "Checks → run"):
   dropped when the run stopped, when more than 5 frames (0.17 s) were lost, when fewer than 30 frames (1 s) were
   tracked, or when it did not find itself in the map within 0.5 s. Otherwise the demo is **split at every lost
   frame**, and each tracked stretch of at least 12 frames (0.4 s) is a **piece**; shorter ones are dropped.
   "Tracked" means seen by the camera (ORB-SLAM3's state OK), never an IMU-only guess.
4. **Pose checks**, with the poses in the table frame (tag 13 of the map at the origin, z up). **Once on the whole
   demo** (all its tracked frames): gravity (|g| = 9.81 ± 0.5 m/s²), metric scale (within 10 %; a demo too short or
   too still to measure it is only reported: "scale not observable"), and, for a demo of at least 10 s, the
   orientation check of map check M4 (SLAM's orientation agrees with gravity within 3°); a failure drops the whole
   demo. **Per piece**: speed (≤ 3 m/s). So a short piece (0.4 s or more) is kept as UMI keeps it.
5. Without the gripper calibration it stops here: **waiting** in `.staging/` (exit 3).
6. **Widths and export per piece** (once the gripper calibration is in, section 4): the finger widths from the tags
   with the folder's `gripper_range.json`, the copied checks on the piece (finger tags seen in ≥ 80 % of its frames,
   widths and main-camera frames for ≥ 80 % of it, no camera or IMU gaps, speed again), then the training images and
   `processed/episode.zarr`. Only the piece's longest stretch with a pose, a finger width and a main-camera frame is
   exported; what is cut off is **reported** (`trimmed 0.27 s (no finger width)` on the outcome line, the details in
   `piece.json` / `reason.json`).
7. A demo with k pieces that pass becomes **k consecutive `episode_N` folders**. The first holds the recording
   (`raw/`); the others point to it (`raw -> ../episode_<first>/raw`), each with its own `processed/episode.zarr`
   (named `<pi_name>_piece<k>` when the demo was split). A demo with no piece left goes to `failed/<pi_name>/` with
   `reason.json`.

At the end it prints one line per recording, what each reason means, and `umiproc status <folder>`; the same is in
`tasks/<folder>/report.json`.

```
episode_000132  ->  episode_1
episode_000133  ->  episode_2, episode_3 (2 of 3 pieces; piece 2 dropped: speed); piece 1 trimmed 0.27 s (no finger width)
episode_000134  ->  failed (slam: too_many_lost)
episode_000135  ->  waiting for the gripper calibration (1 piece(s) passed every pose check)
```

| Reason | Means | Change this in the recording |
|---|---|---|
| `stopped_lost` | SLAM could not follow the demo in the map for 30 frames (1 s) and stopped | start looking at the mapped workspace, stay inside the mapped area, no fast swings, never cover the lens |
| `too_many_lost` | more than 5 frames (0.17 s) not tracked, counted over the whole demo, **including the first frames before it found itself in the map** | start still (about 1 s) looking at the mapped workspace from where the mapping clip went; smooth motion |
| `too_few_tracked` | fewer than 30 tracked frames (1 s) | a demo must be tracked for at least a second |
| `late_relocalization` | not found in the map within 0.5 s | as `too_many_lost`: the start must show what the map knows |
| `no_piece` | no tracked stretch of 12 frames (0.4 s) between the losses | fewer, shorter losses |
| `new_map` | SLAM lost the map and started a new one (its poses would not be in the map's coordinates) | stay inside the mapped area, move more slowly |
| `slam_error` | the SLAM run crashed, timed out or left no trajectory | not the recording: read `processed/localize/umiproc_slam.log` (in `failed/<pi_name>/`) |
| `gravity` | the demo's trajectory disagrees with the IMU's gravity (checked once on the whole demo) | smooth motion, no bumps; a demo of only about 1 s with brisk motion can fail it through its own acceleration (record longer demos); if it repeats on smooth demos, the map, lens or camera-IMU rotation (map check M4) |
| `scale_mismatch` | the demo's scale (checked once on the whole demo) is off by more than 10 % | the map's scale (re-check map check M4) or a SLAM jump after a loss |
| `speed`, `speed_extreme` | faster than 3 m/s (7 m/s: a SLAM jump, not the hand) | move slower |
| `orientation` | SLAM's orientation disagrees with gravity by more than 3° (demos of 10 s or more; drops the whole demo) | not the recording if the demo was smooth: lens model, camera-IMU rotation |
| `tag_coverage` | the finger tags were seen in fewer than 80 % of the piece's frames | keep both finger tags in view; no hand or object over them |
| `kept_fraction`, `piece_too_short` | widths or main-camera frames were missing for more than 20 % of the piece, or what is left covers fewer than 12 frames | as `tag_coverage` |
| `ultrawide_gap`, `imu_gap` | frames or IMU samples missing in the recording | not the hand: check the Pi's recording (dropped frames) |

A piece that fails is dropped and the others are kept (the line says which piece and why); `gravity`,
`scale_mismatch` and `orientation` are judged on the whole demo and drop all of it. A scale that could not be
measured ("scale not observable") is not a reason: it is printed with the demo's pose checks and kept in
`localize.json` (`demo_checks`). Every reason is also in `failed/<pi_name>/reason.json` (`stage`, `reasons`,
`message`, the rules' counts and each piece's outcome, with what was trimmed) or, for a kept recording, in
`episode_N/processed/localize.json`.

### 3.8 "Out of date"

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

### 3.9 An interrupted map (killed, power lost, ssh dropped without tmux)

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

## 4. Gripper calibration

**Why:** the policy also learns the gripper's opening. It is read from the two finger tags in every frame, which
needs the tag-to-tag distance of the fully closed and fully open gripper: the **gripper calibration**, measured once
per session from a short clip. Do it once the map is verified (3.5).

### 4.1 Record the gripper calibration clip (about 15–20 s)

1. Start recording with the gripper pointed at the scene (anything; no SLAM is involved).
2. **Close the fingers fully, then open them fully, slowly**, 3–4 times (at least 5 open/close cycles in all is
   better). "Fully open" is the most the gripper will open in the demos (the robot's limit is 80 mm).
3. **Both finger tags must be visible the whole time** (UMI's rule: each seen in at least 80 % of the frames): keep
   your hand and objects off them. Stop recording.

### 4.2 Run it (on the Jetson)

```bash
umiproc calibrate block_raise episode_000134
```

- `block_raise` is the session's folder; the folder's map must have passed (section 3).
- On the laptop, add `--source ~/Downloads/data/incoming`; `--root` as above.
- It takes about a minute (every frame is read and the tags found in it).

It prints the measured opening, the tag statistics and **UMI's rule**: each finger tag must be seen in at least
80 % of the frames.

```
calibrate: PASSED -> tasks/block_raise/gripper_range.json (from calibration/pi-2026-11-20_episode_000134)
  measured opening 0.080 m (expected 0.080 m +- 0.010 m); closed width 0.0440 m (expected 0.044 m), open width 0.1240 m
  ultrawide frames 452; tag 0 seen 440 (97%), tag 1 seen 431 (95%), both 425 (94%; the fit needs 20 %, UMI's rule 80 %)
  both tags at the expected depth (z 0.018-0.178 m): 425; median z 0.097 m (aruco.nominal_z_m 0.098 m)
  open/close cycles: 5; closed 0.044 m, open 0.124 m (tag-to-tag x; 3-D distance 0.044 m / 0.124 m)
  UMI's finger-tag rule PASS: tag 0 seen in 97 % and tag 1 in 95 % of 452 frames (both at once: 94 %); UMI needs each tag in >= 80 %
```

Exit code 0: passed; 1: failed (nothing was recorded and the clip was not taken: record a new one, or fix the cause,
and run it again); 2: it could not run (no such folder, map not passed, recording missing / still syncing / already
taken, another umiproc command running).

| Line | Passes when | If it fails, change this |
|---|---|---|
| UMI's finger-tag rule | each finger tag seen in ≥ 80 % of the frames | keep both tags in view the whole time (the message names the one missed most): no hand or object over them, even light, open and close slowly, the lens in focus |
| the fit (`gripper_range_fit`) | both tags at the expected depth in at least 3 frames, each tag in ≥ 20 % of the frames | the message names the cause (a tag never seen, tags at the wrong depth: `aruco.nominal_z_m`) |
| `warning: measured opening …` | (a warning, not a failure) the opening is the rig's 0.080 m ± 0.010 m | open fully and close fully; both tags visible at both ends. Record again if it is far off: every width is measured against it |
| `warning: closed width …` | (a warning) the closed width is the rig's 0.044 m ± 0.010 m | close the fingers fully; a tag hidden when closed reads as a wider closed gripper |
| `warning: only N open/close cycles` | (a warning) at least 5 cycles | more cycles |

The tags are found through the lens model the SLAM runs use (the lens file when there is one, section 1).

One calibration per session: a second one replaces the first only while the folder has no episodes yet (the test
demos waiting in `.staging/` are not episodes yet, so a calibration with warnings can be redone right away); once
episodes were made with it, `calibrate` refuses (record a new session to calibrate again). For the same reason `run`
refuses (exit 2, nothing changed) when the settings the calibration was fitted with changed after episodes were made
with it (the rig file's `aruco`, `gripper` or ultrawide camera section in `~/data/umi/rigs/`, the lens file): it would
otherwise fit it again, and the session's episodes would have two different gripper ranges. Put the settings back, or
record a new session. Without episodes it is simply fitted again.

## 5. The rest of the demos and finishing

### 5.1 Record the demos

As the test demos (3.3): start looking at the mapped workspace and still for about 1 s, do the task once at a natural
pace, stay inside the mapped area, no fast swings, never cover the lens, keep both finger tags visible. Reset the
scene between demos. Keep writing down the numbers.

You can process them in batches as you go (5.2 on the demos recorded so far): a problem shows up after a few demos
instead of at the end of the session.

### 5.2 Run them (on the Jetson)

```bash
umiproc run block_raise episode_000135 episode_000150
```

The same command as in 3.4, now with the gripper calibration in place, so each demo goes all the way: localized,
split, pose-checked, then the widths, the copied checks and the export (steps 1–7 of 3.7). Each demo becomes its
`episode_N` folders or goes to `failed/<pi_name>/` with the reason.

The **test demos waiting in `.staging/` are finished by this same run** (any range of the folder does it): they are
not localized again, only the widths, the copied checks and the export run (about a minute per demo).

Exit code 0: every recording got its outcome (episodes or failed: both are listed). The other exit codes are as in
3.4; exit 2 also covers the gripper calibration's settings having changed after episodes were made with it
(section 4).

### 5.3 Check the result

```bash
umiproc status block_raise
```

It shows the map, the gripper calibration `run` uses (opening, the finger-tag rule, its warnings), the episodes with
the pieces of one recording on one line (`episode_2, episode_3 … (pieces 1, 3 of 3)`), pieces whose recording is gone
(below), what waits in `.staging/` and why, and the failed recordings with their reasons. Each `episode_N/processed/`
holds `episode.zarr` (the training data, poses in the table frame: tag 13 at the origin, no other offset),
`status.json`, `piece.json` (which piece of which recording, what was trimmed) and, for the first piece of a
recording, `localize.json` and the SLAM run (`localize/`).

**Never delete the first piece of a split recording alone.** The recording (`raw/`) is stored once, in the first
piece's `episode_N`; the other pieces' `raw` is a link to it (`raw -> ../episode_N/raw`). Deleting that folder leaves
them without their recording: `umiproc status` lists them under "pieces without their recording", and packaging
skips them with a warning that names them. To drop a whole recording, delete all its pieces; to drop one piece that
is not the first, delete only its folder.

### 5.4 Run it again

- **Failed recordings** are not run again by themselves (`skipped (failed earlier at …)`). After fixing a cause
  outside the recording (for example a rebuilt SLAM program for `slam_error`), `--retry-failed` runs them again
  **from scratch** (a new localization; SLAM differs a little from run to run).
- A run that was interrupted (Ctrl-C, ssh dropped, power) leaves its recording in `.staging/`; the next run of the
  folder (any range) picks it up. One interrupted while moving a finished demo into its `episode_N` folders is
  finished by the next run without localizing or exporting it again (`reconciled: …` / `finishing the move`).

## Appendix A: The A/B comparison (first real session only)

The first passing mapping clip is run in three variants (docs/DESIGN.md "SLAM"): all fixes (patches 0001–0012) with
the datasheet IMU noise; the 0001–0006 build (note: it also lacks the crash fixes 0010–0011, so it can crash when a
second map starts; rerun it if it does); and the IMU noise ×10. Each variant gets **its own data root**, so a variant
never mixes with another (a root holds one variant's maps and nothing else) and the same clip can be taken in each.
`/data/incoming` is only read, so all three read the same clip, and none of this touches the session in `~/data/umi`:
it can run after the session.

Once, build the 0001–0006 variant (0.3; 20–30 min on the Jetson):

```bash
PATCHES_UPTO=0006 <repo>/umi-processing/native/build.sh
```

Then set up the three roots with this shell function (paste it into the terminal; `<repo>` as in section 0). It
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

Each must end with its `NAME: SLAM binary … (12 patches)` (6 for `p0006_infl1`) line and no `STOP`. Then run the
three, with the clip's name (here `episode_000130`). Commands 1 and 3 remove `UMIPROC_SLAM_BINARY`, so a value left
over in the terminal cannot silently run them with the 0006 build:

```bash
# 1. all patches, datasheet IMU noise (the default build)
env -u UMIPROC_SLAM_BINARY UMIPROC_ROOT=$AB/p0009_infl1 \
    umiproc map block_raise episode_000130 2>&1 | tee $AB/p0009_infl1.txt
# 2. patches 0001-0006
UMIPROC_ROOT=$AB/p0006_infl1 UMIPROC_SLAM_BINARY=$HOME/ORB_SLAM3_umi/umiproc_build_0006/umiproc_slam \
    umiproc map block_raise episode_000130 2>&1 | tee $AB/p0006_infl1.txt
# 3. all patches, IMU noise x10
env -u UMIPROC_SLAM_BINARY UMIPROC_ROOT=$AB/p0009_infl10 \
    umiproc map block_raise episode_000130 2>&1 | tee $AB/p0009_infl10.txt
```

- Check the first lines of each output. The one starting `map: <pi_name>: SLAM build` reads
  `SLAM build /home/<you>/ORB_SLAM3_umi/umiproc_build/umiproc_slam (patches 0001-0012 (12 patches)); IMU vio_inflation
  noise_density 1, random_walk 1; lens calibrated; fingerprint …` for 1;
  `…/umiproc_build_0006/umiproc_slam (patches 0001-0006 (6 patches))` for 2; `noise_density 10, random_walk 10`
  only for 3; the same `lens …` in all three. The diagnostics at the end repeat them (`SLAM build: …`,
  `IMU vio_inflation: noise_density 10.0, random_walk 10.0`), and every `report.json` records the same
  (`diagnostics.build`, `diagnostics.vio_inflation`, and the full fingerprint in `slam_fingerprint.json`).
- Patches 0007–0009 act only after a later relocalization (docs/DESIGN.md "SLAM"). The mapping clip is relocalized
  at its first frame, so it may not tell 0006 from 0009 apart; localizing the session's demos against each variant's
  map will: `umiproc run` with the same `UMIPROC_ROOT` / `UMIPROC_SLAM_BINARY` as that variant's `map` command (each
  root takes the demos on its own; without a gripper calibration in that root they stop at "waiting", which is
  enough to compare tracking).
- A 4th possible variant is the lens: a root whose `rigs/` has a lens file (`rig_pi_v3.lens.yaml`) while the others
  have none (or another one). Set it up as above, then copy the lens in, e.g. from a lens calibration's result:
  `cp ~/data/umi/lens/<pi_name>/lens.yaml $AB/<variant>/rigs/rig_pi_v3.lens.yaml`; its output then says
  `lens calibrated`. (One was already run this way: `~/data/umi_ab/lens126`, the lens from clip 126.)
- Compare the M2, M3 and M4 lines of the three outputs (`$AB/*.txt`): which variant finishes VIBA 2, re-localizes
  sooner and tracks more frames, and has the scale closest to 1. SLAM differs a little from run to run, so a small
  difference means nothing; run a variant twice if two are close (a failed variant can be run again as it is; a
  passed one says "already taken": delete its root, `rm -rf $AB/<variant>`, and run its `setup_variant` line again).
- The variant roots are only for this comparison. Afterwards, tell me (or the next session) which variant won; later
  sessions are then processed in `~/data/umi` with that setting.
- Remove them when done: `rm -rf ~/data/umi_ab`.
