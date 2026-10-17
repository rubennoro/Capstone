# ORB-SLAM3 in Docker for `process-session --vio orbslam3-docker`

Status: the Dockerfile and the hook are written and unit-tested with a fake `docker`. **The image has never been
built, and ORB-SLAM3 has never run on rig or sim data** (criterion "real ORB-SLAM3 run + accuracy vs simulated
ground truth" is open). Build it on a machine with >= 20 GB free on Docker's disk.

```bash
docker build -t sensor-rig/orbslam3:v1.0-release-focal sensor_rig/docker/orbslam3
docker run --rm sensor-rig/orbslam3:v1.0-release-focal cat /ORB_SLAM3/COMMIT      # write the commit down
sensor-rig process-session --raw <raw root> --session rig1020 --out <export root> --vio orbslam3-docker \
    [--docker-image sensor-rig/orbslam3:v1.0-release-focal]
```

What the hook runs per demo (after `pose-prep --target orbslam3` into `<out>/.sensor_rig/vio/<session>/<NAME>/`):

```
docker run --rm --user UID:GID -v <prep>/<NAME>:/data -w /data <image> \
    /ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_euroc /ORB_SLAM3/Vocabulary/ORBvoc.txt \
    /data/orbslam3_settings.yaml /data /data/times.txt <NAME>
```

It collects `<prep>/<NAME>/f_<NAME>.txt` (EuRoC format, IMU pose, ns), then `pose-import --format orbslam3-euroc
--frame imu`, which applies gravity alignment and the motion (jump / gyro residual) checks.

Patches in the Dockerfile (the build greps for each, so a changed upstream fails loudly) and the symptoms they fix:
- `find_package(OpenCV 4.4)` → 4.2: without it CMake stops with `OpenCV > 4.4 not found.` (FATAL_ERROR), because
  Ubuntu 20.04 ships OpenCV 4.2.
- `System(..., IMU_MONOCULAR, true)` → `false` in `mono_inertial_euroc.cc`: without it the binary dies right after
  loading the vocabulary with `terminate called after throwing an instance of 'std::runtime_error'  what():
  Pangolin X11: Failed to open X display` (exit 134),
  because the viewer thread needs an X display. The alternative is `xvfb-run -a` in front of the binary.
- `-std=c++11` → `c++14`: without it the build fails on C++14 constructs in `Thirdparty` / `src` with current g++.

Mask (F3): pose-prep writes `mask_ultrawide.png` (255 = use, 0 = finger/gripper). Upstream `mono_inertial_euroc`
has no mask argument, so the mask is **unused** with this image. The hook passes `-v mask:/data/mask.png` and
`--mask /data/mask.png` only for an image labelled `sensor_rig.mask_supported=true`.

CLI contract for a mask-capable image: the entry point must accept exactly
`<voc> <settings> <seq_dir> <times_file> <name> --mask <png>` and write `f_<name>.txt` into the working directory.
Upstream-style EuRoC binaries parse their arguments by count (`argc`: 5 + 2 per extra sequence), so two extra
arguments would be read as one more `<seq_dir> <times_file>` pair. A mask-capable build must therefore parse
`--mask` before that count logic (or use a wrapper binary). Never set the label on an unpatched image.

Follow-ups for the first machine with enough disk:
1. Build the image and run one sim episode: `process-session --vio orbslam3-docker` on a `sensor-rig sim` session.
   Compare the trajectory with the sim ground truth: the relative pose error over 0.8 s should be <= 2 % + 1 mm
   and 0.5° (SPEC).
2. **Map then localize** (UMI's recipe for short demos): record one slow mapping episode per session, run ORB-SLAM3
   with atlas saving (`System.SaveAtlasToFile` in the settings), then run each demo with
   `System.LoadAtlasFromFile` in localization mode. This needs two settings variants in `pose/orbslam3.py` and a
   per-session mapping step in `process-session`.
3. UMI fork with masks: either convert the EuRoC folder to the fork's GoPro mp4 + IMU-JSON input, or add a mask
   argument to `mono_inertial_euroc` (a few lines in `Frame` feature extraction) in a patched build.
