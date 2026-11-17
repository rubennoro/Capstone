# The problem, and how umi-processing addresses it

A short explainer for someone new to this. How to run things: [DIRECTIONS.md](DIRECTIONS.md). The decisions in
detail: [docs/DESIGN.md](docs/DESIGN.md). Where the work stands: [PLAN.md](PLAN.md).

## 1. What we are trying to do

We teach a robot arm a task (for example, picking up an eraser) by showing it. A person holds a hand-held gripper,
the same shape as the robot's, and does the task a few dozen times. Each attempt is a **demo**. A camera rig on the
gripper records every demo, and a policy (a neural network) is trained on the recordings.

For each video frame, training needs two things:

1. **The image:** what the gripper saw.
2. **The pose:** where the gripper was in 3-D and which way it pointed, in the same coordinates for every demo. The
   robot learns "move like this", so the motion has to be measured, not guessed.

The images are easy; the camera records them. **The poses are the hard part.** Nothing on the rig measures position
directly, so it has to be worked out from the video and the IMU (motion sensor) afterwards.

## 2. Background in five terms

| Term | What it means here |
|---|---|
| **SLAM** (simultaneous localization and mapping) | Software that watches a video, picks out distinctive points (corners, texture), and from how they move between frames works out both where the camera went and a 3-D map of those points. We use **ORB-SLAM3**. |
| **Visual-inertial** | SLAM that also uses the IMU (accelerometer and gyroscope). A camera alone can't tell how big the world is; the IMU feels gravity and real accelerations, which fixes the scale in metres and which way is up. Before it can use them, the IMU has to be **initialised**: ORB-SLAM3 needs a stretch of steady tracking with real motion to do that. |
| **Map** | The 3-D points and **keyframes** (snapshots SLAM keeps) built from a video. A map can be saved and reused: a later video can **localize** in it ("I've seen this place; I'm here"). |
| **Tracking lost** | SLAM can no longer match what it sees to what it knows. Frames without a pose can't be used for training. |
| **Lens model** | The numbers that say how a pixel corresponds to a direction in the world (focal length, image centre, distortion). SLAM uses them for every point it triangulates. See section 5. |

## 3. Where the problem came from

Our pipeline started from **iPhUMI**, a version of UMI built around an **iPhone**. On an iPhone, the poses come for
free: Apple's ARKit runs its own visual-inertial tracking, with a camera and IMU calibrated at the factory, and hands
back a pose for every frame. iPhUMI just records them.

We dropped the iPhone for our own rig: a **Raspberry Pi 5** with an IMX219 ultrawide camera (160° fisheye), an IMX519
main camera and an LSM6DSOX IMU. There's no ARKit on a Pi, so the first pipeline (`data-processing/`, "dataproc")
replaced it the obvious way: **run ORB-SLAM3 on each demo by itself**, from scratch.

**Every demo failed.** All 10 demos of two tasks (`eraser_pick_1`, `eraser_pick_2`) failed at the SLAM step. The
reasons, found by digging through them:

- **The gripper is always in view.** The fingers, tags and housing take up about 30 % of the image and never move
  relative to the camera. SLAM latches onto them, and they tell it nothing about motion.
- **A demo is a bad SLAM video.** It's short, it starts moving right away, and it spends its time close to one
  object. SLAM wants a calm start, varied views and steady motion to get going, and the IMU needs that too before it
  can be initialised.
- **Each demo started from zero.** Nothing learned from one demo helped the next, and every demo had its own
  coordinates, so even the ones that tracked wouldn't line up with each other.
- **The lens model was a guess.** It turned out to be about 18 % off (section 5).

Masking out the gripper helped a little but didn't fix it (see `HANDOFF_SLAM_MASK.md`).

## 4. The fix: do it the UMI way

**UMI** (Universal Manipulation Interface, Chi et al.) is the original system, built around a GoPro. It never asks
SLAM to start from scratch on a demo. Instead, every recording **session** follows this order:

1. **Mapping clip (1–2 minutes).** Before any demos, film the workspace slowly and calmly, starting above a printed
   marker (the **table tag**) lying flat on the table. This is a good SLAM video by design. ORB-SLAM3 builds **one map**
   of the workspace from it and initialises the IMU. The tag fixes where the table is in that map.
2. **Gripper calibration clip (about 15 s).** Open and close the fingers, so the finger opening can be measured from
   the two small finger tags.
3. **Demos.** Each demo is **localized in the saved map** instead of starting from scratch. The demo only has to
   recognise a place the map already knows, which takes a fraction of a second, and every demo comes out in the same
   coordinates: the map's, aligned to the table.

UMI also has rules for imperfect demos, which we copied, converted from its 60 frames/s to our 30: a demo that loses
tracking for more than about 0.17 s is split at the gap or dropped, and too-short pieces are discarded. The gripper is
masked out of the SLAM images exactly the way UMI masks it.

`umi-processing/` is this recipe for our rig:

- it runs **UMI's own ORB-SLAM3 fork** (built from source, with eleven small bug fixes of ours, each documented in
  `native/patches/`);
- it is **self-contained**: whatever it reuses from dataproc is copied in, so the old pipeline is never touched;
- it adds **checks UMI doesn't have**, so a bad map or demo is caught and explained instead of silently producing bad
  training data. For example, the map must be a single map with the IMU fully initialised, the mapping clip must
  re-localize in it, scale and gravity must agree with the IMU, and the table tag must come out flat and steady.

The commands follow the session: `umiproc map`, then `umiproc calibrate`, then `umiproc run` (DIRECTIONS.md).

## 5. The added piece: the lens model

UMI doesn't ask its users to calibrate the lens. Every UMI gripper uses the **same GoPro with the same lens accessory**,
so the authors measured that lens once and ship the numbers with the code (`gopro_intrinsics_2_7k.json`).

Our camera isn't a standard product like that: an IMX219 sensor with a third-party 160° fisheye lens screwed in by
hand. Two things follow:

- **The guess was wrong.** The rig file's lens model was worked out from "160°" on a product listing. Every measurement
  since puts the focal length at about 464–472 pixels, against 572.9 in the guess: about **18 % off**. The image centre
  and the fisheye distortion vary by unit too, and those can't be worked out from a datasheet at all.
- **SLAM is very sensitive to it.** SLAM triangulates every point through the lens model and expects each one to land
  within about a pixel of where it predicts. With a wrong model, points don't line up from one frame to the next and
  maps fall apart.

The first real mapping clip (episode 128) showed this directly. Same video, same software, only the lens model
changed:

| Lens model | Result |
|---|---|
| The nominal guess | SLAM started a map 34 times and lost each one almost immediately. No usable map. |
| Measured from a board recording (an imperfect one that didn't pass its checks) | One map tracked for 31 s without a break, 89 keyframes, the IMU fully initialised. See [docs/figures/clip128_lens126.png](docs/figures/clip128_lens126.png). |

So **lens calibration comes first** (DIRECTIONS.md section 1): print a ChArUco board (a checkerboard with small markers
in it), keep it perfectly flat (taped to cardboard, or shown on a screen), and record about a minute of it from many
positions across the image. `umiproc lens-calibrate` fits a **fisheye** model, the kind ORB-SLAM3 uses (a standard
lens model can't describe 160°), and only installs it if eight checks pass. Those checks cover enough views, a low
fitting error on views it wasn't fitted to, and corners all the way to the image edges. It's done once per camera,
and redone only if the lens is refocused or remounted.

The lens model also shapes the training images: the export reshapes the fisheye image into the view the policy expects
using this model, so getting it right helps there as well.

## 6. Step by step, where things stand

| # | Step | Status |
|---|---|---|
| 1 | Lens calibration: a passing board recording, installed with `--apply` | **Next.** Six tries so far; the method works, and the remaining issues are a flat board and reaching the image's edges and corners |
| 2 | Mapping clip and `umiproc map` (checks M1–M5) | Built. On clip 128 a measured lens gave a 31 s map, but tracking was lost once, so three maps instead of one; an occasional SLAM crash when a second map starts was two bugs in ORB-SLAM3 itself, now fixed |
| 3 | Gripper calibration and `umiproc run` (localize each demo, split or drop at gaps, export the episode) | To build |
| 4 | Packaging episodes for training | Later |

The order matters: each step only makes sense once the one before it passes. A wrong lens ruins the map; without a map,
demos can't be localized; without localized demos, there's nothing to train on.
