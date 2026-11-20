# Collection log: which recordings worked, and why

Real recordings from the Pi rig (Pi episode numbers), what came out, and what made them good or bad. How to record
and run everything: [DIRECTIONS.md](DIRECTIONS.md). Add a row whenever a recording teaches something.

Where the results are (laptop): `~/Downloads/data/analysis/` (`lens_calibration/<episode>/`,
`mapping/<episode>/<attempt>/`). On the Jetson: `~/data/analysis/lens_calibration/` and each task folder's
`analysis/`.

## Lens calibration

What to aim for (DIRECTIONS.md 2.4): fit error (b) ≤ 0.5 px, test error (c) ≤ 0.6 px, ≥ 25 views, corners out to the
image edges and in all 7 cells. No recording so far reaches all of these.

| Episode | How it was recorded | Fit error (b) / test error (c) | Views | Focal length | Verdict |
|---|---|---|---|---|---|
| 122 | rig pointed at the table tag (one big marker), not the board | no fit | 0 | — | unusable: the board was never seen |
| 124 | board lying on the table, rig pointed at it; board only in the middle, few still holds | — | few | — | bad: no coverage of the edges |
| 125 | bare paper held by one edge; it bent; never close, never lower left | 1.67 / 2.02 px | 14 | 524 px | bad: bent paper |
| **126** | bare paper **held taut with two hands**; still not far enough into the corners or the lower left | **0.69 / 0.90 px** | 19 | 464 px | **best of the 2026-11-16 tries; used for the maps below** |
| 127 | loose sheets on the table, rig moved over them; curled edges, board stayed in the middle | 1.83 / 4.27 px | 19 | 471 px | bad: curled sheets, no edge coverage |
| **175** | (2026-11-18, rigid board; how it was held was not written down: add it) | **0.48 / 0.64 px** | 14 | 449 px | **lowest fit error so far; the lens of the `PICK_TEST1` and `PICK_TEST2` maps below** (both measure the scale correctly). Few views; edges, lower left and top corners still missing |

What made the difference:

- **A flat board is everything.** Every bent or curled board gave over 1.5 px; the taut sheet (126) and 175 got
  under 1 px.
- **Edges and corners are always missing.** Every try warned on (d) and (e): the board never reached the top corners
  or the lower left. That is the next thing to fix: a rigid board, held out at the top corners, far left and right
  and lower left, pausing about 1 s at each.
- **The rig file's lens guess is about 20 % off.** Every fit gave 449–472 px against the guessed 573 px, so the real
  lens is clearly not the guessed one. With the guess, SLAM lost every map it started on clip 128 (below).
- 126 and 175 differ by about 3 % in focal length. Check whether the lens was refocused or turned between
  2026-11-16 and 2026-11-18 before mixing results.

## Mapping clips

What to aim for (DIRECTIONS.md 4.4): one map (M1), the IMU ready (M2), the clip found again in its own map (M3), a
measurable scale (M4), the table tag seen often and in a steady place (M5).

### Episode 128 (2026-11-16, about 76 s)

| Attempt | Lens | Result |
|---|---|---|
| whole clip | the rig file's guess | **failed M1**: 34 map resets, only 76 of 2268 frames tracked in the kept map |
| whole clip | measured (126) | **failed M1**: tracked well for the first ~32 s (IMU fully ready), then lost; 3 maps at the end |
| automatic longest piece (0–31.95 s) | measured (126) | **passed**: one map (89 keyframes, 2534 points), M2 and M3 pass (99 % of frames found again). M4 warns (scale could not be measured), M5 warns (60 tag sightings, but the tag's position spreads ~11 cm and it looks tilted 6°) |

What was good:

- A **calm start above the table tag**, then slow movement: the first 32 s tracked without a break and the IMU set
  itself up fully.
- Textured scene (keyboard, cables, objects on the desk).

What went wrong:

- **Tracking was lost when the camera rose quickly** (from close above the tag to well above it within about 1.5 s,
  around 32 s into the clip): the view changed too much at once. Rise and fall slowly.
- **Too little up/down and side-to-side movement** in the first 32 s, so the map's scale could not be measured (M4).
  Add slow 20–40 cm strokes up/down and sideways early in the clip.
- The tag's spread and tilt (M5) suggest it was not perfectly flat or the map is slightly bent; tape the tag flat on a
  rigid backing.
- On the laptop SLAM only kept up with about 9 of 30 frames/s, which makes maps worse. Map on the Jetson in its
  maximum power mode, with nothing else running.

Results: `~/Downloads/data/analysis/mapping/episode_000128/` (attempts `1_…` to `4_…`; attempt 4 has the 3-D page
`analysis/slam_3d.html` and the videos).

### Episode 137 (2026-11-18, about 44 s): task folder `PICK_TEST1`

Lens 175, whole clip, no cut needed. **Passed, the best map so far**:

- M1: one map from the start (58 keyframes, 2600 points), no resets.
- M2 and M3 pass (every frame found again).
- **M4 passes: the scale is metric (s = 0.996)**.
- M5 warns only because the tag was seen 19 times (limit 20). Its position is steady (0.2 cm spread) and level (3.7°).

Then 30 demos passed and 1 failed (episode 165: 9 frames lost, limit 5). Gripper calibration from episode 179
(opening 4.4–13.4 cm).

What was good: lens 175, slow strokes up/down and sideways early in the clip (that is what makes M4 measurable), a
flat tag, no fast rises.

Results: `~/Downloads/data/analysis/mapping/episode_000137/PICK_TEST1/analysis/`. That folder holds the slam-video and
slam-3d outputs, plus the experimental 3-D scene views: `dense_3d.html` (a point cloud), `mesh_3d.html` (a textured
surface) and `splat_3d.html` (Gaussian splatting, the most photo-like).

### Episode 181 (2026-11-19, about 52 s): task folder `PICK_TEST2`

Lens 175. **Passed**:

- M1: one map (72 keyframes, 3107 points). Two early restarts were dropped automatically: tracking was lost right
  after starting from the tag.
- M2 and M3 pass.
- **M4 passes (s = 0.955)**.
- M5 warns: 55 sightings, but the tag's position spreads about 1.1 cm (limit 1 cm).

Then 32 demos passed and 1 failed (episode 204: 10 frames lost, limit 5). Gripper calibration from episode 217.

Both folders were packaged together in `~/data/umi/datasets/multi/`: `pick1-double-v2` holds all 62 demos, and
`pick1-double-ep{15,20,25}-v1` hold the first 15/20/25 demos of each folder (a data-scaling comparison).
