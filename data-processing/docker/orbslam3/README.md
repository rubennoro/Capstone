# ORB-SLAM3 in Docker for the `orbslam3` SLAM backend

Vendored/adapted from `sensor_rig/docker/orbslam3/` at commit 4be9982. The patches live inline in the Dockerfile.

Status (2026-11-04, laptop, x86_64): **built (image `dataproc/orbslam3:v1.0`, 3.25 GB, commit verified); not yet
run on data (deferred).** The vendored Dockerfile built unchanged on the first try.
- Build time 5 min 52 s (20 threads; no cache). Image `dataproc/orbslam3:v1.0` is 3.25 GB. Free space on `/` dropped
  from 31 GB to 25 GB (image plus build cache), so the "~20 GB free" in the Dockerfile header is conservative.
- Resolved ORB-SLAM3 commit (`/ORB_SLAM3/COMMIT`): `0df83dde1c85c7ab91a0d47de7a29685d046f637` (tag v1.0-release).
- The build only prints warnings (Eigen deprecations, unused variables); no errors.
- Future (not written yet): results on synthetic episodes will go in `docs/STATUS.md`, and an e2e test
  `tests/test_orbslam3_e2e.py` (`-m orbslam3`, skipped when the image is missing) will run the image on one.

## Build

```bash
docker build --builder default -t dataproc/orbslam3:v1.0 data-processing/docker/orbslam3
#   or: docker buildx build --load -t dataproc/orbslam3:v1.0 data-processing/docker/orbslam3
docker run --rm dataproc/orbslam3:v1.0 cat /ORB_SLAM3/COMMIT        # write the commit down
```

Pass `--builder default` (or `--load`). On this laptop the default buildx builder is a `docker-container` builder, and
without the flag its image never reaches the local image store, so `docker run dataproc/orbslam3:v1.0` can't find it.

On the Jetson (arm64), the same Dockerfile should work: `ubuntu:20.04` is multi-arch and ORB-SLAM3 builds with
`-march=native`. This is untested.

## What the backend runs per episode

The backend first writes the EuRoC folder, `times.txt` and `orbslam3_settings.yaml` into `work_dir`. Then it runs:

```
docker run --rm --user UID:GID -v <work_dir>:/data -w /data dataproc/orbslam3:v1.0 \
    /ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_euroc /ORB_SLAM3/Vocabulary/ORBvoc.txt \
    /data/orbslam3_settings.yaml /data /data/times.txt episode
```

It reads `<work_dir>/f_episode.txt`. That file is in EuRoC format: the IMU pose in the first keyframe's IMU frame,
times in ns. Gravity alignment, the IMU sanity checks, and the frame conventions are in `docs/GEOMETRY.md`. The
container's stdout goes to `<work_dir>/orbslam3_docker.log`, which is also where `n_maps` and `n_resets` are counted.

## Patches

The build greps for each patch, so a changed upstream fails loudly.

- **`find_package(OpenCV 4.4)` → 4.2.** Without it, CMake stops with `OpenCV > 4.4 not found.`, because Ubuntu 20.04
  ships OpenCV 4.2.
- **`System(..., IMU_MONOCULAR, true)` → `false`** in `mono_inertial_euroc.cc`. Without it, the binary dies right
  after loading the vocabulary with `Pangolin X11: Failed to open X display` (exit 134).
- **`-std=c++11` → `c++14`.** Without it, the build fails on C++14 constructs with current g++.

## Follow-ups

- **Map then localize** (UMI's recipe for short demos): first record one slow mapping episode, saving the atlas with
  `System.SaveAtlasToFile`. Then run each demo with `System.LoadAtlasFromFile` in localization mode. This needs a
  second settings variant.
- **Finger mask.** Upstream `mono_inertial_euroc` has no mask input, so the gripper fingers are tracked as features.
