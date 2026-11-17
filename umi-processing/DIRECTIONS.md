# umiproc: directions

Step-by-step procedures for the Pi 5 rig, in the order you do them. Each section is added once its commands exist
(docs/DESIGN.md "Build order"). Commands run on the Jetson unless a section says otherwise.

1. Lens calibration (optional, about 1 minute of recording)
2. Jetson setup
3. Prove the environment: the map
4. Prove the demos
5. Gripper calibration and finishing the episodes

## 1. Lens calibration

**Why:** it measures how the ultrawide (IMX219, 160°) lens bends light, so SLAM uses the real lens instead of the
rig file's nominal guess. Optional: `map` and `run` never wait for it (without it they use the nominal lens, labelled
"nominal, uncalibrated"). About 1 minute of recording, then 1–2 minutes of processing.

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
4. Glue it onto the rigid backing, smoothly: no bubbles, waves or curled corners. A flat board matters more than
   anything else here.
5. Fix the board **flat and still**: tape it to a wall, or lay it on a table. During the recording the board stays
   put and you move the rig.

The board uses `DICT_5X5_100` markers on purpose: the gripper's finger tags and the table tag are `DICT_4X4_50` and
are often in view; they are never mistaken for the board.

### Before recording

- **Lock the lens focus:** the IMX219's lens focuses by turning in its holder. Once it is sharp, put a small piece of
  tape across the lens and its holder so it cannot turn. If the lens turns later, the calibration no longer
  applies (see "When to redo it").
- **Light:** bright and even, with no lamp or window reflected in the board. More light means shorter exposures and
  less motion blur.

### Record (on the Pi)

Record **one ordinary episode of about 60 seconds** on the Pi, started and stopped like a demo. Hold the rig as
usual; the fingers and whatever they hold are in view at the bottom of the image, and that is expected (that area is
masked and ignored).

- **Move slowly, and hold still about 1 second at each pose.** Only the still moments are used.
- **Distance:** 20–45 cm from the board. Use near, middle and far.
- **Tilt:** besides looking at the board straight on, also look at it from the side, up to about 45°: left, right,
  from above, from below.
- **Where the board appears in the image:** make it visit the **centre**, the **left** and **right** edges, the
  **top** edge, **both top corners**, and the **lower left, beside the gripper**. The bottom middle and lower right
  are covered by the gripper, so the board cannot show up there. That is expected.
- **For the edges and corners, turn the rig rather than sliding it sideways.** Face the board, then rotate the rig
  until the board sits in the corner of the image. The board then still faces the lens, and its corners are still
  detected even at the very edge of the fisheye image. Check (d) needs these corner views.
- Keep the board's markers sharp and unobstructed. A partly visible board is fine.

#### The shot list (about 30 holds, about 60 s)

Tape the board flat on a wall at chest height, so you can stand in front of it and move freely. Keep the gripper
still in one position the whole time (don't squeeze it). Rhythm: **move for about 1 s, hold still for about 1 s**
(count "one-one-thousand"), move to the next. Holds are what count; the moves in between are thrown away.

**How to put the board in a given part of the image: turn the rig the opposite way.** Turn the rig to the right and
the board slides to the left of the image; tip the rig down and the board slides up. Keep the rig at the same
distance and just rotate it, as if your wrist were a tripod head.

The image, as the 3 x 3 grid the checks use (the gripper covers the bottom middle and bottom right):

```
+-----------+-----------+-----------+
| TOP-LEFT  |    TOP    | TOP-RIGHT |   board goes here by tipping the rig DOWN
|           |           |           |   (and turning it right / left for the corners)
+-----------+-----------+-----------+
|   LEFT    |  CENTRE   |   RIGHT   |
|           |           |           |
+-----------+-----------+-----------+
| LOW-LEFT  |  gripper  |  gripper  |   only the lower left is usable
|           | (masked)  | (masked)  |
+-----------+-----------+-----------+
```

| # | Distance (lens to board) | Holds (where the board appears in the image) | How |
|---|---|---|---|
| 1 | ~40 cm (a forearm) | centre; centre again with the rig rolled ~45° | face the board squarely |
| 2 | ~40 cm | left, right, top | turn right / turn left / tip down |
| 3 | ~40 cm | top-left, top-right (hold each **2 s**, these are the hardest) | tip down **and** turn right / left until the board touches the image's corner |
| 4 | ~40 cm | low-left, beside the fingers | turn right and tip up a little |
| 5 | ~20–25 cm (a hand span) | centre, left, right, top-left, top-right, low-left | same moves; the board fills much of the image and is partly cut off at the edges, which is fine |
| 6 | about 45 cm | centre, left, right, top, top-left, top-right, low-left | same moves |
| 7 | ~35 cm, board in the centre | tilted ~45°: from the left, from the right, from above, from below | step sideways (or crouch / reach up) and aim back at the board |
| 8 | ~35 cm | two tilted holds with the board in the top-left and top-right | combine 7 and 3 |

That is about 30 holds. If the recording is cut short, rows 3 and 5 matter most (checks (d), (e) and (h)).

**What a good recording looks like when you play it back:** the board sharp (markers crisp, no smear) in every
hold; over the whole clip the board has visited every cell of the grid except the two gripper cells, including
right up into both top corners; some holds with the board big and some small; some with it seen at a slant. After
processing, `coverage.png` shows the same thing: green dots in all 7 cells and beyond the magenta circle in the top
corners.

**What makes it fail:** the board only ever near the centre (the most common mistake: the edges are what the lens
calibration needs); moving through holds without stopping (blur); the board curling or flexing; glare across the
markers; the lens focus changing.

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
