# libargus / NvBufSurface for the Argus camera backend (T2; included by CMakeLists.txt when RIG_WITH_ARGUS=ON).
# Defines the INTERFACE target rig_argus. Headers: ${RIG_ARGUS_ROOT}/include (Jetson: /usr/src/jetson_multimedia_api
# from the nvidia-l4t-jetson-multimedia-api package; Argus/ and EGLStream/ are symlinks there, nvbufsurface.h next to
# them). Libraries (JetPack 6 / L4T r36: /usr/lib/aarch64-linux-gnu/nvidia): libnvargus_socketclient (client of
# nvargus-daemon), libnvbufsurface, libEGL.
#
# RIG_ARGUS_COMPILE_ONLY=ON: headers only (laptop compile check against extracted r36.4 headers); nothing that links
# rigcommon can be linked, so build just the object: ninja common/CMakeFiles/rigcommon.dir/drivers/camera/argus/argus_camera.cpp.o
option(RIG_ARGUS_COMPILE_ONLY "Argus headers only, no libraries (compile check off-target)" OFF)

add_library(rig_argus INTERFACE)
target_include_directories(rig_argus SYSTEM INTERFACE "${RIG_ARGUS_ROOT}/include")

if(NOT RIG_ARGUS_COMPILE_ONLY)
  set(_rig_tegra_libdirs /usr/lib/aarch64-linux-gnu/nvidia /usr/lib/aarch64-linux-gnu/tegra /usr/lib/aarch64-linux-gnu)
  find_library(RIG_ARGUS_LIB nvargus_socketclient PATHS ${_rig_tegra_libdirs})
  find_library(RIG_NVBUFSURFACE_LIB nvbufsurface PATHS ${_rig_tegra_libdirs})
  find_library(RIG_EGL_LIB EGL PATHS ${_rig_tegra_libdirs})
  if(NOT RIG_ARGUS_LIB OR NOT RIG_NVBUFSURFACE_LIB OR NOT RIG_EGL_LIB)
    message(FATAL_ERROR "RIG_WITH_ARGUS: libnvargus_socketclient (${RIG_ARGUS_LIB}), libnvbufsurface "
                        "(${RIG_NVBUFSURFACE_LIB}) or libEGL (${RIG_EGL_LIB}) not found. On the Jetson: "
                        "sudo apt install nvidia-l4t-jetson-multimedia-api (JetPack 6.2). Off-target compile "
                        "check: -DRIG_ARGUS_COMPILE_ONLY=ON")
  endif()
  target_link_libraries(rig_argus INTERFACE ${RIG_ARGUS_LIB} ${RIG_NVBUFSURFACE_LIB} ${RIG_EGL_LIB})
  message(STATUS "physicalSetup: argus = ${RIG_ARGUS_LIB}; nvbufsurface = ${RIG_NVBUFSURFACE_LIB}")
else()
  message(WARNING "RIG_ARGUS_COMPILE_ONLY=ON: the Argus backend is compiled but cannot be linked")
endif()
