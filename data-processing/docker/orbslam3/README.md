# ORB-SLAM3 in Docker for the `orbslam3` SLAM backend

Vendored/adapted from `sensor_rig/docker/orbslam3/` at commit 4be9982. The patches live inline in the Dockerfile.

Status: the Dockerfile and the backend (`dataproc/slam/orbslam3.py`) are written. The backend is unit-tested with a fake
`docker` runner. **The image has not been built yet, and ORB-SLAM3 has not yet run on synthetic or rig data.** Wave 2
(E6) does both and records the outcome in `docs/STATUS.md`. You need at least 20 GB free on Docker's disk.

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
