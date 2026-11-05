# Vendored/adapted from sensor_rig/docker/orbslam3/Dockerfile at commit 4be9982 (image tag and labels renamed; the
# mask notes dropped: dataproc passes no mask).
# ORB-SLAM3 (upstream UZ-SLAMLab, tag v1.0-release) for the dataproc `orbslam3` SLAM backend.
#
#   docker build --builder default -t dataproc/orbslam3:v1.0 data-processing/docker/orbslam3   # ~20 GB free needed
#
# (`--builder default`, or `--load`: on the laptop the default buildx builder is a docker-container builder whose
# images do not land in the local image store otherwise.)
# Pinned: ORB_SLAM3_REF (tag v1.0-release, Dec 2021) and Pangolin v0.6. The resolved ORB-SLAM3 commit is written
# to /ORB_SLAM3/COMMIT at build time; record it in docs/STATUS.md after the first build.
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
LABEL org.opencontainers.image.title="dataproc ORB-SLAM3" \
      dataproc.orbslam3_ref="${ORB_SLAM3_REF}"
WORKDIR /data
