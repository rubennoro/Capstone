# ORB-SLAM3 on the Jetson (native build) for the `orbslam3` SLAM backend

`dataproc run` (`--slam auto` or `orbslam3`) runs ORB-SLAM3's `mono_inertial_euroc` directly on the Jetson. It
looks for the build in `$ORB_SLAM3_DIR`, default `~/ORB_SLAM3`.

> **Status (2026-11-07): built on the Jetson, not yet run on data.** Built natively on the Jetson Orin
> (JetPack 7, Ubuntu 24.04, aarch64, gcc 13.3, OpenCV 4.8.0 from NVIDIA, Eigen 3.4, Boost 1.83) at commit
> `0df83dde1c85c7ab91a0d47de7a29685d046f637`. The binary links cleanly and starts. Connecting it to real recordings is
> the next step.

## Build (once)

These are the exact commands that worked. The main compile took 14 minutes at `-j2`, and the whole build about
20 minutes.

**Don't run the upstream `./build.sh`.** It uses `make -j` with no job limit, which runs out of memory on a 7.3 GB
board. The steps below do the same work by hand at `-j2`.

**Memory.** The Jetson had no swap. The build peaked at about 1.7 GB of swap, so set up the swap file first:

```bash
sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
echo 'vm.swappiness=10' | sudo tee /etc/sysctl.d/99-swappiness.conf && sudo sysctl -p /etc/sysctl.d/99-swappiness.conf
```

**1. System libraries.** build-essential, cmake, git, Eigen, OpenCV, GL and wayland were already on the Jetson.

```bash
sudo apt update
sudo apt install -y libboost-serialization-dev libboost-dev libssl-dev libglew-dev \
    libegl1-mesa-dev libepoxy-dev libxkbcommon-dev wayland-protocols
```

**2. Pangolin v0.6.** ORB-SLAM3 links against it even with the viewer off. It needs two missing-include fixes for
gcc 13.

```bash
git clone --depth 1 --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git ~/Pangolin
cd ~/Pangolin
sed -i '1i #include <cstdint>' src/image/image_io_jpg.cpp
sed -i '0,/#include/s//#include <cstdint>\n#include/' include/pangolin/log/packetstream_tags.h
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF \
      -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_PANGOLIN_FFMPEG=OFF
cmake --build build -j2
sudo cmake --install build && sudo ldconfig          # -> /usr/local/lib/libpangolin.so
```

**3. ORB-SLAM3 v1.0-release and its two patches.**

```bash
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ~/ORB_SLAM3
cd ~/ORB_SLAM3 && git checkout v1.0-release           # 0df83dde1c85c7ab91a0d47de7a29685d046f637
sed -i 's/++11/++14/g' CMakeLists.txt
sed -i 's/ORB_SLAM3::System::IMU_MONOCULAR, true)/ORB_SLAM3::System::IMU_MONOCULAR, false)/' \
    Examples/Monocular-Inertial/mono_inertial_euroc.cc
grep -q 'IMU_MONOCULAR, false)' Examples/Monocular-Inertial/mono_inertial_euroc.cc && echo "patch ok"
```

**4. Build what `build.sh` builds, at `-j2`.** Sophus is header-only; its own tests fail under gcc 13 on aarch64 (see
below), so they're skipped.

```bash
cd ~/ORB_SLAM3 && mkdir -p build_logs
for lib in DBoW2 g2o; do
  cmake -S Thirdparty/$lib -B Thirdparty/$lib/build -DCMAKE_BUILD_TYPE=Release
  cmake --build Thirdparty/$lib/build -j2
done
cmake -S Thirdparty/Sophus -B Thirdparty/Sophus/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
cmake --build Thirdparty/Sophus/build -j2
(cd Vocabulary && tar -xf ORBvoc.txt.tar.gz)          # ORBvoc.txt, 139 MB
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2 2>&1 | tee build_logs/ORB_SLAM3.build.log    # about 14 min; warnings only
```

**5. Check it.**

```bash
cd ~/ORB_SLAM3
ls -l Examples/Monocular-Inertial/mono_inertial_euroc Vocabulary/ORBvoc.txt lib/libORB_SLAM3.so
ldd Examples/Monocular-Inertial/mono_inertial_euroc lib/libORB_SLAM3.so | grep "not found"    # expect nothing
./Examples/Monocular-Inertial/mono_inertial_euroc      # prints "Usage: ..." and exits 1 (expected with no args)
```

If you built it somewhere else, point dataproc at it: `export ORB_SLAM3_DIR=/path/to/ORB_SLAM3` (add it to
`~/.bashrc` to keep it). The default, `~/ORB_SLAM3`, needs nothing.

### Patches and build changes, and why

- **ORB-SLAM3, `-std=c++11` → `c++14`.** v1.0 uses C++14 constructs, and current g++ refuses them under C++11.
- **ORB-SLAM3, `System(..., IMU_MONOCULAR, true)` → `false`** in `mono_inertial_euroc.cc`. Turns the Pangolin viewer
  off so it runs without a display. With it on, it aborts with `Pangolin X11: Failed to open X display` when
  headless.
- **Pangolin, `#include <cstdint>`** in `src/image/image_io_jpg.cpp` and `include/pangolin/log/packetstream_tags.h`.
  gcc 13 no longer pulls `<cstdint>` in indirectly, so `uint8_t` / `uint32_t` are undeclared without it.
- **Pangolin, Python / FFmpeg / tools / examples off.** Not needed, and the likeliest parts of a 2021 release to
  break on Python 3.12 and FFmpeg 6.
- **Sophus, `BUILD_TESTS=OFF`.** Its tests build with `-Werror`, and gcc 13 gives a false-positive `-Warray-bounds`
  in Eigen's ARM NEON code (`test_sim2.cpp`). ORB-SLAM3 only uses Sophus's headers.
- **No OpenCV patch.** The Docker build lowered `find_package(OpenCV 4.4)` to 4.2 for Ubuntu 20.04. The Jetson's
  OpenCV 4.8.0 (CMake config in `/usr/lib/cmake/opencv4/`) satisfies 4.4 as is.

## What the backend runs per episode

The backend first writes the EuRoC folder (`mav0/cam0`, `mav0/imu0`), `times.txt` and `orbslam3_settings.yaml` into
`processed/slam/work/` of the episode. Then, with that folder as the working directory, it runs:

```
$ORB_SLAM3_DIR/Examples/Monocular-Inertial/mono_inertial_euroc $ORB_SLAM3_DIR/Vocabulary/ORBvoc.txt \
    <work>/orbslam3_settings.yaml <work> <work>/times.txt episode
```

It reads `<work>/f_episode.txt`: EuRoC format, the IMU pose in the first keyframe's IMU frame, times in ns. Gravity
alignment, the IMU sanity checks and the frame conventions are in [GEOMETRY.md](GEOMETRY.md). stdout and stderr go
to `<work>/orbslam3.log` (its first line is the exact command), which is also where `n_maps` and `n_resets` are
counted.

## Follow-ups

- **Map then localize** (UMI's recipe for short demos): first record one slow mapping episode, saving the atlas with
  `System.SaveAtlasToFile`. Then run each demo with `System.LoadAtlasFromFile` in localization mode. This needs a
  second settings variant.
- **Finger mask.** Upstream `mono_inertial_euroc` has no mask input, so the gripper fingers are tracked as features.
