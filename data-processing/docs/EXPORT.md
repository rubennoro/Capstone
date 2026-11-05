# Images and zarr export

How `dataproc` turns decoded camera frames plus grid-aligned pose/width into the training `.zarr.zip`. The schema
itself is in [DESIGN.md](DESIGN.md), under "Output zarr contract". This page covers the code and the per-episode artifact.

## Modules

| Module | Role |
|---|---|
| `dataproc/imageprep.py` | **Pure** image preparation: `MainPrep`, `UltrawidePrep`, `ImagePrep`, `letterbox`. No IO. |
| `dataproc/export/frames.py` | **Pure** frame selection on the 60 Hz grid: `nearest_frame_index`, `ultrawide_substream`, `downsample_local`. |
| `dataproc/export/images.py` | Stage `images` glue: decodes a `RawEpisode`, then runs selection and prep (`prepare_episode_images`). |
| `dataproc/export/zarr_writer.py` | `EpisodeArrays`, the per-episode `episode.zarr` (`write_episode_zarr` / `load_episode_zarr`), and `build_task_zarr_zip`. |
| `dataproc/_vendor/` | Copied from iPhUMI at 4be9982: `replay_buffer.py` (verbatim), `imagecodecs_numcodecs.py` (the `JpegXl` subset; id `imagecodecs_jpegxl`), `cv_util.py` (verbatim). |

## Images

- **Main** (`prep_main`): the native view, with no undistortion. The frame is letterboxed to a centred black square,
  then resized to 224×224 with `INTER_AREA`. The result is bit-identical to iPhUMI's
  `get_image_transform_with_border(..., mode="rgb")`.
- **Ultrawide** (`prep_ultrawide`): `to_virtual` remaps the physical fisheye to the virtual pinhole (`rig.virtual_ultrawide`:
  640×480, iPhone ultrawide K, `R_phys_virt`). It uses one precomputed `build_remap` table, bilinear interpolation, and
  paints uncovered pixels black. The same letterbox then produces 224×224. The width stage detects tags on the
  `to_virtual` output.
- **Deployment must import these exact classes** so that run-time images match training images. Build the class
  once per rig, since the remap table is the expensive part.

## Frame selection (60 Hz grid `t_grid`, mid-exposure camera times `rig.camera_time_ns`)

- **Main.** Each grid step takes the nearest frame; ties go to the earlier frame. `valid = |dt| <= max_dt_ns`.
  A main camera below 60 fps repeats frames.
  - iPhUMI uses "latest at or before" for main. Nearest is what the plan specifies. This is the only intentional
    semantic difference, and the schema is unaffected.
- **Ultrawide** (`ultrawide_substream`, default 10 Hz):
  1. **Schedule:** `t_grid[0] + k/rate` for `k = 0, 1, …` while the time is ≤ `t_grid[-1]`.
  2. **Selection:** the nearest ultrawide frame to each schedule time, with duplicates removed.
  3. **Upsample index:** for each grid step, `upsample_local` is the index of the latest selected frame at or before
     that step, clamped at 0. This is iPhUMI's `_compute_frame_indices` rule, and a test runs the original source.
  4. **Trim:** selected frames after the last one any step uses are dropped, so that `M == upsample_local[-1] + 1`.
     iPhUMI derives `n_uw` this way.
- **Image age vs state (inherited iPhUMI semantics).**
  - In training, the ultrawide image of a step is ~0–100 ms older than the state at 10 Hz, because it is the latest
    selected frame at or before the step. Selection jitter can add up to half an ultrawide frame interval.
  - The first steps of an episode may clamp to frame 0, which can lie slightly *ahead* of the state. It is the frame
    nearest `t_grid[0]`, so typically ≤ half an ultrawide frame interval (~17 ms at 30 fps), and more if early
    frames were dropped.
  - At deployment, the live ultrawide frame is ~0 ms old. This is a known train/deploy difference, not a bug.
- **Deployment timestamps.** Deployment camera timestamps must use the same mid-exposure reference as
  `rig.camera_time_ns` (same `timestamp_ref`, exposure/2 shift and `time_offset_ns`). Otherwise nearest-main-frame
  selection is biased by that offset.
- **Zarr maps** (exactly as in `gen_replay_buffer_zarr.py`):
  - `upsample_index = local + previous ultrawide total`;
  - `episode_ends_camera_right_ultrawide_rgb = cumulative M`;
  - `downsample_index = searchsorted(local, arange(M), 'left') + prev_main_end`.

## Per-episode artifact: `episode_N/processed/episode.zarr`

This is a zarr v2 directory. It is written under a temporary name and renamed into place. Replacing an existing
one is **near-atomic**:
1. the old directory is moved aside to a unique `.episode.zarr.old-*`;
2. the new one is renamed in;
3. the old one is removed.

If step 2 fails, the old directory is put back. A crash between steps 1 and 2 leaves the old episode under the
`.old-*` name, with no `episode.zarr`.

```
.zattrs              {"schema": "dataproc.episode_zarr.v1", "name": <Pi dir name>, "task_label": <task.yaml label>}
main_rgb             (N, 224, 224, 3) uint8   chunks (1, 224, 224, 3), imagecodecs_jpegxl level 99, numthreads 1
uw_rgb               (M, 224, 224, 3) uint8   same
uw_upsample_local    (N,) int64
eef_pos              (N, 3) float32           metres, world (z-up, world_offset applied)
eef_rot_axis_angle   (N, 3) float32           rotation vector
gripper_width        (N, 1) float32           metres
```

- **No re-encoding.** The images use the final zip's codec. `build_task_zarr_zip` copies their compressed chunks byte
  for byte, which avoids a second lossy JPEG XL generation and the encoding time. The test checks that the zip
  matches one built from the in-memory arrays.
- **Validation.** `EpisodeArrays` validates on construction and raises `ValueError` if any of these fail:
  - shapes and dtypes are as listed above;
  - N ≥ 2;
  - all values are finite;
  - no rotation vector is exactly 0;
  - `uw_upsample_local` is ≥ 0, non-decreasing, and ends at M − 1.

## Task zip: `build_task_zarr_zip(episodes, out_zip_path, *, tmp_dir=None, num_workers=4)`

- **Inputs:** `episodes` is a list of `EpisodeArrays` and/or `episode.zarr` paths, in output order.
- **Build** (the iPhUMI flow, using the vendored `ReplayBuffer` in a temporary `DirectoryStore`):
  1. For each episode, `add_episode` adds the lowdim arrays (no compressor) and one task `[0, N − 1)` named
     `task_label`.
  2. The ultrawide meta arrays are added.
  3. The image arrays are created (chunk = 1 frame, JPEG XL 99).
  4. Meta and lowdim arrays are rechunked to 8 KiB.
  5. The result is saved with `save_to_store` into a `zarr.ZipStore`.
- **Output:** the zip is written to `.<name>.tmp-<pid>` and renamed onto `out_zip_path`. On failure, the old zip is
  kept and no temp files remain.
- **Location:** the temp directory defaults to the output folder, so a tmpfs `/tmp` on the Jetson is never filled.
  `out_zip_path` must end in `.zarr.zip`, and its folder must already exist.
- **Errors:** duplicate episode names raise.
- **Returns** `{"path", "n_episodes", "n_steps", "n_ultrawide", "episode_names"}`.

## Verified against the golden (`tests/test_export_zarr_schema.py`)

- **Random 3-episode zip.** Compared with the golden, it has:
  - the same set of arrays, dtypes, compressor configs, filters (`vlen-utf8` on names), attrs and root attrs (`{}`),
    and the same groups;
  - image chunks `(1, 224, 224, 3)`;
  - lowdim/meta chunks that follow the iPhUMI 8 KiB rule.

  The meta semantics hold: `task_lengths = N − 1`, `task_labels_ends = cumsum`, `task_data_ends = episode_ends − 1`,
  and the ultrawide maps are consistent.
- **Golden replay.** Rebuilt from the golden's own episode lengths, per-episode ultrawide maps and lowdim values,
  every array matches the golden's shape, chunks and compressor. Every meta and lowdim array is **byte-identical**:
  same raw-chunk sha256 as `scripts/regression/iphone_pipeline_check.py`.
- **Readers.** The vendored `ReplayBuffer` and behavior_prompting's `ReplayBuffer` (loaded by file path) both open the
  zip. Decoded images are within JPEG XL level-99 loss.
