# argus_sof helper (Jetson only). Build: sensor_rig/native/argus_sof/build.sh   (NOT BUILT YET)
cmake_minimum_required(VERSION 3.10)
project(argus_sof CXX)
set(CMAKE_CXX_STANDARD 14)

set(MMAPI_DIR "$ENV{JETSON_MULTIMEDIA_API}")
if(NOT MMAPI_DIR)
  set(MMAPI_DIR /usr/src/jetson_multimedia_api)
endif()
if(NOT EXISTS "${MMAPI_DIR}/include/Argus/Argus.h")
  message(FATAL_ERROR
    "Jetson Multimedia API not found (looked for ${MMAPI_DIR}/include/Argus/Argus.h).\n"
    "On the Jetson: sudo apt install nvidia-l4t-jetson-multimedia-api   (or set JETSON_MULTIMEDIA_API).\n"
    "This helper cannot be built on a laptop; sensor_rig falls back to the 'argus' (GStreamer) backend.")
endif()

find_library(ARGUS_LIB nvargus_socketclient PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra)
find_library(NVBUFSURFACE_LIB nvbufsurface PATHS /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra)
find_library(EGL_LIB EGL)
if(NOT ARGUS_LIB OR NOT NVBUFSURFACE_LIB OR NOT EGL_LIB)
  message(FATAL_ERROR "libnvargus_socketclient / libnvbufsurface / libEGL not found: is this a JetPack 6 system?")
endif()

add_executable(argus_sof argus_sof.cpp
  ${MMAPI_DIR}/samples/common/classes/NvBufSurface.cpp)
target_include_directories(argus_sof PRIVATE ${MMAPI_DIR}/include ${MMAPI_DIR}/include/libjpeg-8b)
target_link_libraries(argus_sof ${ARGUS_LIB} ${NVBUFSURFACE_LIB} ${EGL_LIB} pthread rt)
