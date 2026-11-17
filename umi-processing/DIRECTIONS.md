# umiproc: directions

Step-by-step procedures for the Pi 5 rig, in the order you do them. Each section is added once its commands exist
(docs/DESIGN.md "Build order"). Commands run on the Jetson unless a section says otherwise.

1. Lens calibration (optional, about 1 minute of recording)
2. Jetson setup
3. Prove the environment: the map
4. Prove the demos
5. Gripper calibration and finishing the episodes

## Start here: bringing the Jetson up, step by step

Do these in order; each step has a check that tells you it passed before you go on.

| # | Do | Where | Passed when |
|---|---|---|---|
| 1 | Get the code: `cd <repo> && git fetch && git checkout rnoro/data-work && git pull` | Jetson | `git log --oneline -1` shows the newest umi-processing commit |
| 2 | Python environment (section 2.2, about 5 min) | Jetson | `umiproc --help` lists `lens-board`, `lens-calibrate`, …, and the check tests pass |
| 3 | Start the SLAM build (section 2.1, about an hour) **inside `tmux`**, and let it run while you do step 4 | Jetson | it ends with `== done` |
| 4 | **Lens calibration** (section 1): print the board, tape it to cardboard, record about 60 s on the Pi, run `umiproc lens-calibrate`, repeat until every check passes, then `--apply` | Pi + Jetson | `lens-calibrate` prints no `FAIL`, and `--apply` installs `~/data/umi/rigs/rig_pi_v3.lens.yaml` |
| 5 | `umiproc status` (section 2.4) | Jetson | lens file: calibrated; umiproc_slam and vocabulary: found |
| 6 | A collection session: map, then gripper calibration, then demos | Pi + Jetson | sections 3–5 (`umiproc map` is being finished; it comes with the next update of this branch) |

**Do step 4 before any mapping.** The lens model matters more than anything else for SLAM on this rig: on the first
real mapping clip (episode 128) the rig file's nominal lens, about 18 % off in focal length, made SLAM lose every map
it started (34 resets, no usable map), while the same clip with a measured lens built a map with the IMU fully
initialised. Lens calibration needs only step 2, not the SLAM build.

`tmux` in one line: `tmux new -s build`, run the command, detach with `Ctrl-b` then `d`, come back with
`tmux attach -t build`. A long command keeps running even if your ssh connection drops.

## 1. Lens calibration

**Why:** it measures how the ultrawide (IMX219, 160°) lens bends light, so SLAM uses the real lens instead of the
rig file's nominal guess. `map` and `run` never wait for it (without it they use the nominal lens, labelled
"nominal, uncalibrated"), **but do it first**: the nominal lens is about 18 % off in focal length, which is enough
to make SLAM fail (see "Start here"). About 1 minute of recording, then 1–2 minutes of processing.

**What you need:** a printer, something rigid and flat to glue the board to (foam board, a clipboard, a flat piece
of wood or acrylic), glue (spray adhesive or a glue stick), a ruler, a small piece of tape, and a well-lit spot.
The commands need `umiproc` installed (section 2).

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

Find the recording's directory name (the newest one in `/data/incoming`) and run the calibration on it:

```bash
ls -t /data/incoming | head -3                 # newest first, e.g. episode_000123
umiproc lens-calibrate episode_000123
```

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

### 2.2 Python environment

umiproc has its own environment and never imports dataproc, so don't install it into `~/envs/env_dataproc`:

```bash
python3 --version                                   # 3.10 or newer
python3 -m venv ~/envs/env_umiproc
source ~/envs/env_umiproc/bin/activate              # in every new terminal
pip install --upgrade pip
pip install -e <repo>/umi-processing[test]
umiproc --help                                      # lens-board, lens-calibrate, map, calibrate, run, status
```

Check the install (a minute or two):

```bash
cd <repo>/umi-processing
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_umiproc_*.py tests/test_lens_*.py tests/test_slam_*.py -m "not slow"
```

Then check that the SLAM program from 2.1 runs and reads umiproc's settings file (about 30 s):

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_umiproc_slam_inputs.py -m slow
```

It must say `passed`, not `skipped` (skipped means the SLAM program or vocabulary was not found: see 2.1).

### 2.3 The data root

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

### 2.4 Check: `umiproc status`

```bash
umiproc status
```

On a fresh Jetson after 2.1–2.2 it shows:

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

## 3. Prove the environment: the map

(Coming with step 2.)

## 4. Prove the demos

(Coming with step 3.)

## 5. Gripper calibration and finishing the episodes

(Coming with step 3.)
