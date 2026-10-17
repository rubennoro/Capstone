# ORB-SLAM3 (upstream UZ-SLAMLab, tag v1.0-release) for `sensor-rig process-session --vio orbslam3-docker`.
#
#   docker build -t sensor-rig/orbslam3:v1.0-release-focal sensor_rig/docker/orbslam3      # ~20 GB free needed
#
# NOT BUILT IN THE DEVELOPMENT SESSION (the build machine had < 15 GB free on Docker's disk); see README.md here.
# Pinned: ORB_SLAM3_REF (tag v1.0-release, Dec 2021) and Pangolin v0.6. The resolved ORB-SLAM3 commit is written
# to /ORB_SLAM3/COMMIT and to the image label at build time; record it in your notes after the first build.
# Upstream mono_inertial_euroc has NO mask input: sensor_rig's finger mask (mask_ultrawide.png) is written but
# unused with this image (label sensor_rig.mask_supported=false). UMI's fork supports masks but reads GoPro mp4 +
# telemetry JSON, not EuRoC (follow-up, see README.md).
FROM ubuntu:20.04
ARG ORB_SLAM3_REF=v1.0-release
ARG PANGOLIN_REF=v0.6
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates pkg-config \
        libeigen3-dev libopencv-dev libboost-serialization-dev libssl-dev \
        libglew-dev libgl1-mesa-dev libegl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols \
        python3 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${PANGOLIN_REF} https://github.com/stevenlovegrove/Pangolin.git /Pangolin \
 && cmake -S /Pangolin -B /Pangolin/build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF \
 && cmake --build /Pangolin/build -j"$(nproc)" && cmake --install /Pangolin/build && ldconfig
# Patches (each verified with grep so a changed upstream fails the build loudly):
#  - v1.0 sets -std=c++11 but uses C++14 features with current compilers: c++14
#  - CMakeLists requires OpenCV 4.4 (FATAL_ERROR otherwise); Ubuntu 20.04 ships 4.2, which ORB-SLAM3 builds with
#  - mono_inertial_euroc constructs System(..., IMU_MONOCULAR, true): the Pangolin viewer needs an X display and
#    aborts in a headless container -> false (no viewer)
RUN git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git /ORB_SLAM3 \
 && cd /ORB_SLAM3 && git checkout ${ORB_SLAM3_REF} && git rev-parse HEAD > /ORB_SLAM3/COMMIT \
 && sed -i 's/++11/++14/g' CMakeLists.txt \
 && sed -i 's/find_package(OpenCV 4.4)/find_package(OpenCV 4.2)/' CMakeLists.txt \
 && sed -i 's/ORB_SLAM3::System::IMU_MONOCULAR, true)/ORB_SLAM3::System::IMU_MONOCULAR, false)/' \
        Examples/Monocular-Inertial/mono_inertial_euroc.cc \
 && grep -q 'find_package(OpenCV 4.2)' CMakeLists.txt \
 && grep -q 'IMU_MONOCULAR, false)' Examples/Monocular-Inertial/mono_inertial_euroc.cc \
 && ./build.sh
LABEL org.opencontainers.image.title="sensor-rig ORB-SLAM3" \
      sensor_rig.orbslam3_ref="${ORB_SLAM3_REF}" \
      sensor_rig.mask_supported="false"
WORKDIR /data
