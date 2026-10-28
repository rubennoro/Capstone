# physicalSetup: Jetson sensor rig (collection + deployment). See README.md and docs/PIPELINE.md.
cmake_minimum_required(VERSION 3.22)
project(physicalSetup VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------------------------
option(RIG_WITH_ARGUS "Build the libargus camera backend (Jetson / extracted r36.4 headers)" OFF)
option(RIG_WITH_ORBSLAM3 "Build the ORB-SLAM3 live/offline SLAM backends (Jetson)" OFF)
option(RIG_WITH_CERES "Build Ceres-based stages (bspline traj_fit)" OFF)
option(RIG_WITH_PYTHON "Build the rigpy pybind11 bindings (common/python)" OFF)
option(RIG_BUILD_TESTS "Build the unit tests" ON)
set(RIG_SANITIZE "" CACHE STRING "Sanitizers: '' | address | undefined | thread | address,undefined")
set_property(CACHE RIG_SANITIZE PROPERTY STRINGS "" address undefined thread "address,undefined")
set(RIG_ARGUS_ROOT "/usr/src/jetson_multimedia_api" CACHE PATH "Jetson Multimedia API root (include/Argus/...)")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # rigcommon is linked into the pybind11 module
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings as errors for all first-party code. Third-party headers come in through imported targets or
# SYSTEM include dirs, which are exempt.
add_compile_options(-Wall -Wextra -Wpedantic -Werror)

# ---------------------------------------------------------------------------------------------------------------
# Sanitizers
# ---------------------------------------------------------------------------------------------------------------
if(RIG_SANITIZE)
  string(REPLACE "," ";" _rig_san_list "${RIG_SANITIZE}")
  foreach(_s IN LISTS _rig_san_list)
    if(NOT _s MATCHES "^(address|undefined|thread)$")
      message(FATAL_ERROR "RIG_SANITIZE: unknown sanitizer '${_s}' (allowed: address, undefined, thread)")
    endif()
  endforeach()
  if("thread" IN_LIST _rig_san_list AND "address" IN_LIST _rig_san_list)
    message(FATAL_ERROR "RIG_SANITIZE: thread cannot be combined with address")
  endif()
  string(REPLACE ";" "," _rig_san_flag "${_rig_san_list}")
  add_compile_options(-fsanitize=${_rig_san_flag} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=${_rig_san_flag})
  message(STATUS "physicalSetup: sanitizers = ${_rig_san_flag}")
endif()

# ---------------------------------------------------------------------------------------------------------------
# Dependencies (laptop: Ubuntu 22.04 packages; Jetson: JetPack 6.2 + apt, see scripts/jetson_setup.sh)
# ---------------------------------------------------------------------------------------------------------------
# Do not derive package prefixes from $PATH: on the laptop, anaconda3/bin on PATH would make CMake pick conda's
# GTest/yaml-cpp/nlohmann_json/fmt (ABI and header mismatch with the system OpenCV/spdlog). System prefixes (/usr)
# and CMAKE_PREFIX_PATH still work. Turn off with -DRIG_IGNORE_PATH_PREFIXES=OFF.
option(RIG_IGNORE_PATH_PREFIXES "Ignore $PATH-derived prefixes in find_package/find_library" ON)
if(RIG_IGNORE_PATH_PREFIXES)
  set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH OFF)
endif()

find_package(Threads REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs calib3d features2d highgui videoio objdetect
             OPTIONAL_COMPONENTS aruco)  # aruco: contrib module on 4.5.x; objdetect has ArUco from 4.7
find_package(yaml-cpp REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json 3.2 REQUIRED)
find_package(JPEG REQUIRED)
find_package(PkgConfig)

# GStreamer: RIG_WITH_GST defaults to ON when the development packages are found.
set(_rig_gst_found OFF)
if(PkgConfig_FOUND)
  pkg_check_modules(RIG_GST QUIET IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
  if(RIG_GST_FOUND)
    set(_rig_gst_found ON)
  endif()
endif()
option(RIG_WITH_GST "Build the GStreamer camera backend (default: ON if found)" ${_rig_gst_found})
if(RIG_WITH_GST AND NOT _rig_gst_found)
  message(FATAL_ERROR "RIG_WITH_GST=ON but gstreamer-1.0/-app/-video development packages were not found")
endif()

if(RIG_WITH_ARGUS)
  if(NOT EXISTS "${RIG_ARGUS_ROOT}/include/Argus/Argus.h")
    message(FATAL_ERROR "RIG_WITH_ARGUS=ON but ${RIG_ARGUS_ROOT}/include/Argus/Argus.h does not exist "
                        "(set RIG_ARGUS_ROOT)")
  endif()
  # T2 owns cmake/argus.cmake (libraries, EGL, NvBufSurface); it must define the target rig_argus.
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/argus.cmake OPTIONAL RESULT_VARIABLE _rig_argus_cmake)
  if(NOT TARGET rig_argus)
    add_library(rig_argus INTERFACE)
    target_include_directories(rig_argus SYSTEM INTERFACE "${RIG_ARGUS_ROOT}/include")
  endif()
endif()

if(RIG_WITH_ORBSLAM3)
  # T6 owns cmake/orbslam3.cmake; it must define the target rig_orbslam3.
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/orbslam3.cmake OPTIONAL)
  if(NOT TARGET rig_orbslam3)
    message(FATAL_ERROR "RIG_WITH_ORBSLAM3=ON but cmake/orbslam3.cmake did not define target rig_orbslam3")
  endif()
endif()

if(RIG_WITH_CERES)
  find_package(Ceres REQUIRED)
endif()

if(RIG_WITH_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  # Prefer the pybind11 installed for the chosen interpreter (e.g. the venv) over a system copy: Ubuntu 22.04's
  # /usr/lib/cmake/pybind11 is 2.9, whose numpy dtype checks break under numpy 2 ("expected a uint8 image").
  if(NOT pybind11_DIR)
    execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
                    OUTPUT_VARIABLE _rig_pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE _rig_pybind11_rc ERROR_QUIET)
    if(_rig_pybind11_rc EQUAL 0 AND EXISTS "${_rig_pybind11_dir}")
      set(pybind11_DIR "${_rig_pybind11_dir}" CACHE PATH "pybind11 CMake dir (from ${Python3_EXECUTABLE})")
    endif()
  endif()
  # >= 2.12 is the first pybind11 with numpy 2 support.
  find_package(pybind11 2.12 CONFIG REQUIRED)
endif()

# CLI11 (vendored single header, v2.4.2, BSD-3-Clause; third_party/CLI11/LICENSE).
add_library(rig_cli11 INTERFACE)
target_include_directories(rig_cli11 SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/CLI11)
add_library(rig::cli11 ALIAS rig_cli11)

# ---------------------------------------------------------------------------------------------------------------
# Version header (rig/version.hpp, generated)
# ---------------------------------------------------------------------------------------------------------------
find_package(Git QUIET)
set(RIG_GIT_DESCRIBE "unknown")
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --dirty --abbrev=10
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTPUT_VARIABLE RIG_GIT_DESCRIBE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(NOT RIG_GIT_DESCRIBE)
    set(RIG_GIT_DESCRIBE "unknown")
  endif()
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/generated/rig/version.hpp
               @ONLY)

# ---------------------------------------------------------------------------------------------------------------
# Source-tree conventions (globbed so that parallel tasks never edit the same CMakeLists; see README.md)
# ---------------------------------------------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/RigHelpers.cmake)

if(RIG_BUILD_TESTS)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  enable_testing()
endif()

add_subdirectory(common)
add_subdirectory(episode_collection)
add_subdirectory(deployment)
if(RIG_BUILD_TESTS)
  add_subdirectory(tests)
endif()

message(STATUS "physicalSetup ${PROJECT_VERSION} (${RIG_GIT_DESCRIBE}): ARGUS=${RIG_WITH_ARGUS} GST=${RIG_WITH_GST} "
               "ORBSLAM3=${RIG_WITH_ORBSLAM3} CERES=${RIG_WITH_CERES} PYTHON=${RIG_WITH_PYTHON} "
               "SANITIZE='${RIG_SANITIZE}' OpenCV=${OpenCV_VERSION}")
