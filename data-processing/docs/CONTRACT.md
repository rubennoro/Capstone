# Raw episode contract (`dataproc.raw.v1`)

What the Pi 5 must deliver to the Jetson for every episode. It is the format of the transmission plan
(`data_transmission_plan.pdf`: two MP4s + CSVs + `meta.json` + `MANIFEST.sha256` + `COMPLETE`, sent as-is with
`rsync -a --partial`) made precise, plus the fields processing needs. The Jetson reads it with
`dataproc/raw_episode.py`; every rule below is checked there, and the problem code it reports is given in
`[brackets]`.

The Pi recorder does not write this format yet: `dataproc run` converts what it does write
(`/data/incoming`) into it. See [PI_ADAPTER.md](PI_ADAPTER.md).

## 1. Directory

```
~/data/incoming/<episode_dir>/
  cam0.mp4            main camera (IMX519)
  cam1.mp4            ultrawide camera (IMX219 fisheye)
  cam0_frames.csv     one row per frame of cam0.mp4
  cam1_frames.csv     one row per frame of cam1.mp4
  imu.csv             IMU samples (LSM6DSOX)
  meta.json           see §3
  MANIFEST.sha256     SHA-256 of every file above
  COMPLETE            empty, written last by the Pi
  .verified           written by the Jetson, never by the Pi (§6)
```

- The destination root is `~/data/incoming/` (the PDF's `/data/incoming/` moved under the dataproc root).
- `<episode_dir>`: `[A-Za-z0-9][A-Za-z0-9._-]*` (starts with a letter or digit, so never hidden, `.` or `..`)
  [`bad_episode_name`], unique for the lifetime of the rig, e.g. `<session_id>_episode_000123`. Episode numbers
  restart per session, so the session id must be part of the name.
- The episode directory is a real directory, not a symlink [`episode_is_symlink`]. It is flat, and every entry in
  it is a regular file: no subdirectories, no symlinks [`not_regular_file`].
- Every file except `MANIFEST.sha256`, `COMPLETE` and `.verified` must be listed in the manifest [`extra_file`].
  Additional listed files (e.g. a Pi log) are allowed and ignored. No file name may start with a dot except
  `.verified` (the Jetson treats other dot-files as an rsync transfer in progress, §6).
- All eight files are required [`missing_file`, `incomplete` for `COMPLETE`].
- Never re-send an episode under a new name, and never re-use a name or a (`session_id`, `episode_number`) pair:
  the Jetson quarantines duplicates (§6).

## 2. Clock and stamps

- Camera `t_ns` = the libcamera `SensorTimestamp` of the frame, **verbatim and unmodified** (no offset, no
  exposure correction). What instant it marks (start or end of exposure) is the Jetson's business: the rig yaml
  says so (`timestamp_ref`) and the Jetson derives mid-exposure from `exposure_ns`.
- **One clock.** The IMU samples, the button events and `wall_anchor.clock_ns` must be stamped on the same clock
  that `SensorTimestamp` uses, and `meta.json` `clock` names it: `CLOCK_MONOTONIC` or `CLOCK_BOOTTIME`. IMU `t_ns`
  is the sample time on that clock (FIFO timestamps mapped to it, not the read time).
- Self-check: a frame's `t_ns` should be slightly earlier than the same clock read in the request-completed
  callback for that frame (by roughly the exposure + readout + delivery time, a few to a few tens of ms). If it is
  later, or seconds away, the clocks differ.
- Integer nanoseconds everywhere (`*_ns`), never floats. `t_ns` strictly increases within each file
  [`t_ns_not_monotonic`].
- **Pre-roll** (recording before the button press) is optional but recommended: ~2 s of motion before
  `button_start_ns` lets SLAM initialize. Likewise a short post-roll after `button_stop_ns`. The training data is
  cut to `[button_start_ns, button_stop_ns]`.

## 3. `meta.json`

A JSON object. Unknown extra keys are allowed (kept, ignored). Required fields [`meta_invalid` unless noted]:

| Field | Type | Meaning |
|---|---|---|
| `schema` | string | exactly `"dataproc.raw.v1"` [`schema_version`] |
| `session_id` | string | recording session id |
| `episode_number` | int ≥ 0 | episode number within the session |
| `rig_id` | string `[A-Za-z0-9_-]+` | selects `rigs/<rig_id>.yaml` on the Jetson, e.g. `"rig_v1"` |
| `clock` | `"CLOCK_MONOTONIC"` \| `"CLOCK_BOOTTIME"` | the one clock of all `*_ns` stamps |
| `wall_anchor` | `{"clock_ns": int, "unix_ns": int}` | one simultaneous reading of `clock` and `CLOCK_REALTIME` (ns) |
| `button_start_ns` | int | button press that starts the demonstration (on `clock`) |
| `button_stop_ns` | int | button press that ends it; must be > `button_start_ns` [`button_order`] |
| `cameras` | object | exactly `cam0` and `cam1` [`camera_roles`], each as below |
| `imu` | object | as below |

Each camera (`cameras.cam0`, `cameras.cam1`):

| Field | Type | Meaning |
|---|---|---|
| `role` | string | `cam0` must be `"main"`, `cam1` must be `"ultrawide"` [`camera_roles`] |
| `resolution` | `[width, height]` ints | size of the frames in the MP4 [`video_resolution_mismatch`] |
| `fps` | number > 0 | nominal frame rate (main 60, ultrawide 30 for rig_v1) |
| `frame_count` | int ≥ 0 | rows in `camX_frames.csv` [`count_mismatch`] |
| `dropped_frames` | int ≥ 0 | frames the Pi knows it lost (should equal the `sensor_seq` gaps, §4) |
| `sensor` | string, optional | e.g. `"imx519"` |

`imu`:

| Field | Type | Meaning |
|---|---|---|
| `rate_hz` | number > 0 | nominal output data rate |
| `sample_count` | int ≥ 0 | rows in `imu.csv` [`count_mismatch`] |
| `dropped_samples` | int ≥ 0 | samples the Pi knows it lost (should equal the `sample_idx` gaps) |
| `sensor` | string, optional | e.g. `"lsm6dsox"` |

The Jetson orders waiting episodes by the wall time of the button press,
`wall_anchor.unix_ns + (button_start_ns - wall_anchor.clock_ns)`.

Example:

```json
{
  "schema": "dataproc.raw.v1",
  "session_id": "2026-11-05_s01",
  "episode_number": 123,
  "rig_id": "rig_v1",
  "clock": "CLOCK_MONOTONIC",
  "wall_anchor": {"clock_ns": 5123000000000, "unix_ns": 1793900000000000000},
  "button_start_ns": 5125000000000,
  "button_stop_ns": 5140000000000,
  "cameras": {
    "cam0": {"role": "main", "sensor": "imx519", "resolution": [1280, 720], "fps": 60,
             "frame_count": 1020, "dropped_frames": 0},
    "cam1": {"role": "ultrawide", "sensor": "imx219", "resolution": [820, 616], "fps": 30,
             "frame_count": 510, "dropped_frames": 1}
  },
  "imu": {"sensor": "lsm6dsox", "rate_hz": 416, "sample_count": 7072, "dropped_samples": 0}
}
```

## 4. CSVs

Comma-separated, UTF-8 without BOM, first line is exactly the header [`csv_header`], one row per line, no quoting,
no empty fields [`csv_parse`]. Integer columns are plain decimal integers of at most 18 digits; float columns are
finite [`csv_parse`]. At least one data row [`empty_stream`].

**`cam0_frames.csv`, `cam1_frames.csv`**

```
frame_idx,t_ns,sensor_seq,exposure_ns
0,5123000123456,48211,4000000
1,5123016790123,48212,4000000
```

- Row *i* describes decoded frame *i* of `camX.mp4`: `frame_idx` is exactly `0, 1, ..., n-1` [`frame_idx`] and the
  MP4 decodes to exactly `n` frames [`decoded_count_mismatch`]. Every captured frame is encoded; a frame that is not
  in the MP4 must not be in the CSV.
- `sensor_seq`: the sensor's frame sequence number (libcamera `SensorSequence`), strictly increasing
  [`sequence_not_increasing`]. A jump > 1 means dropped frames; it is reported with its size [`sensor_seq_gap`,
  warning] and compared with `dropped_frames` [`drop_counter_mismatch`, warning].
- `exposure_ns`: exposure time of that frame (libcamera `ExposureTime` in ns): `0 < exposure_ns < 1e9 / fps`
  with `fps` from `meta.json` [`exposure_invalid`].
  Lock libcamera `FrameDurationLimits` to the nominal fps so auto-exposure can never lengthen the frame period;
  otherwise long exposures in low light make whole episodes invalid.

**`imu.csv`**

```
t_ns,sample_idx,gx,gy,gz,ax,ay,az
5123000100000,0,0.0012,-0.0031,0.0004,0.12,-0.05,9.79
```

- `gx gy gz` in rad/s, `ax ay az` in m/s² (specific force: reads ≈ +9.81 on the axis pointing up when static),
  in the IMU's own sensor frame, unfiltered by the Pi beyond the chip's configured filters.
- `sample_idx`: counter of samples read from the chip, strictly increasing [`sequence_not_increasing`]; a jump > 1
  means dropped samples [`imu_sample_gap`, warning; `drop_counter_mismatch`].

## 5. MP4s and the button interval

- `cam0.mp4`, `cam1.mp4`: one video stream each (H.264 or any codec FFmpeg/PyAV decodes), constant resolution
  equal to `meta.json` `resolution`, frames in capture order [`video_unreadable` if PyAV cannot decode it].
  Use high quality (little compression): the frames feed SLAM and ArUco tag detection.
- Every stream (cam0, cam1, imu) should cover `[button_start_ns, button_stop_ns]`. A stream that starts after
  the start press or ends before the stop press is reported [`button_outside_stream`, warning]; one that does not
  overlap the interval at all is an error [`button_no_overlap`].

## 6. Manifest, COMPLETE, `.verified`

- `MANIFEST.sha256` is `sha256sum` output: one line per file, `<64 hex>  <name>` (two spaces, or space + `*`),
  plain file names only (a leading `./` and a UTF-8 BOM are accepted, as `sha256sum -c` accepts them). Generate it
  with `cd <dir> && sha256sum <files> > MANIFEST.sha256`. It lists the six data files [`manifest_missing_entry`] and any other file in the
  directory; it never lists itself, `COMPLETE` or `.verified` [`manifest_malformed`]. Listed files must exist
  [`manifest_listed_missing`] with that hash [`hash_mismatch`].
- `COMPLETE` is written last, after the manifest. Note that rsync may deliver it before other files, so it means
  "complete on the Pi", not "arrived".
- `.verified` is created on the Jetson **only** after `sha256sum -c MANIFEST.sha256` passes for the received
  directory. Until the Jetson ships its own verify command, the Pi's sender loop runs:

  ```bash
  ssh jetson "cd ~/data/incoming/$n && sha256sum -c --quiet MANIFEST.sha256 && touch .verified" \
    && touch "$ep/.synced"
  ```

  Intake (`dataproc/intake.py`) only takes episodes that have `.verified`. An episode that also contains any other
  dot-prefixed file (an rsync temp file: a re-send in progress) is left in `incoming/` untouched and retried on the
  next intake. Otherwise intake re-checks the manifest and the counts (`RawEpisode.validate`) and moves the
  episode into the task, or moves it to `quarantine/<episode_dir>/` (`__2`, `__3`, ... on name collisions) with
  `reason.json`. Quarantine reasons include every error above, `duplicate` (the directory name or the
  (`session_id`, `episode_number`) pair was already taken in by any task) and `rig_mismatch` (`rig_id` differs from
  the target task's rig). Nothing is ever deleted.
- The Pi must not modify an episode after `COMPLETE`, and never re-sends one that has `.synced`.

## 7. What the Jetson checks

Errors (the episode is rejected): every bracketed code above except the warnings. Warnings (`sensor_seq_gap`,
`imu_sample_gap`, `drop_counter_mismatch`, `button_outside_stream`) carry their sizes (gap counts, missing frames,
time gaps in ms) and are judged later by processing, which applies thresholds (e.g. an IMU or ultrawide gap > 20 ms
inside the button interval rejects the episode).
