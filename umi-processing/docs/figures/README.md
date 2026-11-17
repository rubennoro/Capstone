# Figures

`clip128_lens126.png`: the ORB-SLAM3 mapping run on clip 128 (Pi recording `episode_000128`) with the experimental lens.
That lens is the 126 fit, a calibration that did not pass its checks. The build used patches 0001-0009.

The figure has four parts:

- a timeline of every fed frame: tracked in the saved map, tracked in a later map, or not tracking;
- the saved map's camera path, seen from three sides;
- six frames from the ultrawide video;
- the numbers, computed from the CSV and the log.

Made with (`pip install -e umi-processing[plot]`):

```
umiproc plot-run --csv RUN/trajectory.csv --log RUN/log.txt --video ~/Downloads/data/incoming/episode_000128/cam0.mp4 \
  --frames 45,480,955,975,1400,2230 --out docs/figures/clip128_lens126.png \
  --title "Clip 128 (episode_000128), mapping run with the experimental lens (the 126 fit, a calibration that did NOT pass its checks), patches 0001-0009"
```

`RUN` was a scratch folder from that session: `/tmp/claude-1000/-home-ruben-Capstone-data-work/6b9540ba-9ac3-4e29-baef-c4535f45cc0e/scratchpad/crash128/run1/`.

What the figure shows:

- **Map 0 tracked well, then tracking was lost.** Map 0 tracked 923 of 2268 frames (40.7%) with no gap, from 1.4 s
  to 32.1 s (a 30.8 s stretch). It has 89 keyframes, and its IMU initialisation finished (VIBA 1 and 2 both ran).
- **Two more maps followed.** After tracking was lost (frame 964, 32.1 s), the run started two more maps: map 1 has 85
  keyframes, and map 2 has 19 keyframes without VIBA 1 or 2. Together they tracked 1231 frames (54.3%). The log
  shows 6 map resets.
- **So `map` check M1 would fail.** M1 needs exactly one map with keyframes at the end, and this run ends with 3.

Two numbers differ from earlier notes. 32.1 s here vs 32.5 s: 32.1 s is the last tracked frame; 32.5 s is frame 975,
the not-tracking frame shown in the figure. 30.8 s here vs "31 s": they agree after rounding.

The run with the nominal lens on the same clip (34 resets, a 15-keyframe fragment) is not in this figure. Those numbers
come from the main session.
