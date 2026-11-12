# franka_control: the C++ process that commands the Franka Panda (STRUCTURE.md section 12).
#
#   build.sh                                              libfranka (if missing) + build/franka_control
#   cmake -S . -B build-test -DFRANKA_CONTROL_SANITIZE=ON && cmake --build build-test && ctest --test-dir build-test
#   cmake -S . -B build-tsan -DFRANKA_CONTROL_TSAN=ON && cmake --build build-tsan && ctest --test-dir build-tsan -L threads
cmake_minimum_required(VERSION 3.16)
project(franka_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g1")   # line tables: enough for sanitizer reports

set(FRANKA_CONTROL_WITH_LIBFRANKA "AUTO" CACHE STRING "Use libfranka: AUTO, ON or OFF")
set_property(CACHE FRANKA_CONTROL_WITH_LIBFRANKA PROPERTY STRINGS AUTO ON OFF)
option(FRANKA_CONTROL_SANITIZE "Build with ASan + UBSan (tests)" OFF)
option(FRANKA_CONTROL_TSAN "Build with TSan (threading tests)" OFF)
option(FRANKA_CONTROL_BUILD_TESTS "Build the GoogleTest tests" ON)
if(FRANKA_CONTROL_SANITIZE AND FRANKA_CONTROL_TSAN)
  message(FATAL_ERROR "FRANKA_CONTROL_SANITIZE and FRANKA_CONTROL_TSAN cannot be combined")
endif()
# The binary is Release (-O2). Sanitizer builds are -O2 -g too: at -O0 the FakeArm's 1 kHz loop can't keep its period.
if(NOT CMAKE_BUILD_TYPE)
  if(FRANKA_CONTROL_SANITIZE OR FRANKA_CONTROL_TSAN)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
  else()
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
endif()

# ---- the one libfranka version setting lives in config.yaml ----
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml" _version_line REGEX "^libfranka_version:")
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" LIBFRANKA_VERSION "${_version_line}")
if(NOT LIBFRANKA_VERSION)
  message(FATAL_ERROR "config.yaml: no 'libfranka_version: \"X.Y.Z\"' line")
endif()

set(CONTRACTS_EXAMPLES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../contracts/examples" CACHE PATH "contracts example files")
set(CONTRACTS_LIMITS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../contracts/contracts/limits.yaml" CACHE FILEPATH
    "contracts limits.yaml")

execute_process(COMMAND git -C "${CMAKE_CURRENT_SOURCE_DIR}" rev-parse --short HEAD
                OUTPUT_VARIABLE FRANKA_CONTROL_GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT FRANKA_CONTROL_GIT_HASH)
  set(FRANKA_CONTROL_GIT_HASH "unknown")
endif()

# ---- sanitizers ----
set(_san_flags "")
if(FRANKA_CONTROL_SANITIZE)
  set(_san_flags -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
elseif(FRANKA_CONTROL_TSAN)
  set(_san_flags -fsanitize=thread -fno-omit-frame-pointer)
endif()
if(_san_flags)
  add_compile_options(${_san_flags})
  add_link_options(${_san_flags})
endif()

# ---- dependencies ----
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)            # 0.7.0 on this machine (STRUCTURE.md 12.1 deviation)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq>=4.3)
find_path(CPPZMQ_INCLUDE_DIR zmq.hpp REQUIRED)  # cppzmq: header-only, system copy

include(FetchContent)
FetchContent_Declare(msgpack_cxx
  URL https://github.com/msgpack/msgpack-c/releases/download/cpp-6.1.1/msgpack-cxx-6.1.1.tar.gz
  URL_HASH SHA256=5fd555742e37bbd58d166199e669f01f743c7b3c6177191dd7b31fb0c37fa191)
FetchContent_GetProperties(msgpack_cxx)
if(NOT msgpack_cxx_POPULATED)
  FetchContent_Populate(msgpack_cxx)       # header-only: only its include/ is used, no build
endif()
add_library(msgpack_headers INTERFACE)
target_include_directories(msgpack_headers SYSTEM INTERFACE "${msgpack_cxx_SOURCE_DIR}/include")
target_compile_definitions(msgpack_headers INTERFACE MSGPACK_NO_BOOST)

set(FRANKA_CONTROL_HAS_LIBFRANKA OFF)
if(NOT FRANKA_CONTROL_WITH_LIBFRANKA STREQUAL "OFF")
  if(FRANKA_CONTROL_WITH_LIBFRANKA STREQUAL "ON")
    find_package(Franka ${LIBFRANKA_VERSION} EXACT REQUIRED)
  else()
    find_package(Franka ${LIBFRANKA_VERSION} EXACT QUIET)
  endif()
  if(Franka_FOUND)
    set(FRANKA_CONTROL_HAS_LIBFRANKA ON)
  endif()
endif()
if(FRANKA_CONTROL_HAS_LIBFRANKA)
  set(_libfranka_version_text "${LIBFRANKA_VERSION}")
  message(STATUS "franka_control: libfranka ${LIBFRANKA_VERSION} found")
else()
  set(_libfranka_version_text "none")
  message(STATUS "franka_control: built without libfranka (only --mock runs; run build.sh for the real arm)")
endif()

set(_warnings -Wall -Wextra -Wpedantic)

# ---- the core library: everything but main.cpp and the two libfranka classes ----
add_library(franka_control_core STATIC
  src/pose_math.cpp src/log.cpp src/command_line.cpp src/config.cpp src/config_loader.cpp src/contract_limits.cpp
  src/messages.cpp src/contract_error.cpp src/codec.cpp src/arm.cpp src/arm_status.cpp src/fake_arm.cpp
  src/homing_motion.cpp src/gripper.cpp src/fake_gripper.cpp src/gripper_worker.cpp src/safety_gate.cpp
  src/waypoint_trajectory.cpp src/impedance_controller.cpp src/watchdog.cpp src/state_machine.cpp
  src/shared_state.cpp src/control_loop.cpp src/supervisor.cpp src/command_server.cpp src/state_publisher.cpp
  src/app.cpp)
target_include_directories(franka_control_core PUBLIC include ${CPPZMQ_INCLUDE_DIR})
target_link_libraries(franka_control_core PUBLIC Eigen3::Eigen yaml-cpp PkgConfig::ZMQ msgpack_headers pthread)
target_compile_options(franka_control_core PRIVATE ${_warnings})
target_compile_definitions(franka_control_core PUBLIC
  FRANKA_CONTROL_CONFIG_FILE="${CMAKE_CURRENT_SOURCE_DIR}/config.yaml"
  CONTRACTS_LIMITS_FILE="${CONTRACTS_LIMITS_FILE}"
  CONTRACTS_EXAMPLES_DIR="${CONTRACTS_EXAMPLES_DIR}"
  FRANKA_CONTROL_LIBFRANKA_VERSION="${_libfranka_version_text}"
  FRANKA_CONTROL_GIT_HASH="${FRANKA_CONTROL_GIT_HASH}")

add_executable(franka_control src/main.cpp)
target_compile_options(franka_control PRIVATE ${_warnings})
if(FRANKA_CONTROL_HAS_LIBFRANKA)
  add_library(franka_control_libfranka STATIC src/libfranka_arm.cpp src/franka_hand_gripper.cpp)
  target_link_libraries(franka_control_libfranka PUBLIC franka_control_core Franka::Franka)
  target_compile_options(franka_control_libfranka PRIVATE ${_warnings})
  target_compile_definitions(franka_control_libfranka PUBLIC FRANKA_CONTROL_HAS_LIBFRANKA=1)
  target_link_libraries(franka_control PRIVATE franka_control_libfranka)
else()
  target_link_libraries(franka_control PRIVATE franka_control_core)
endif()

# ---- tests ----
if(FRANKA_CONTROL_BUILD_TESTS)
  enable_testing()
  # GoogleTest 1.11.0 from the system sources, compiled with this build's sanitizer flags (STRUCTURE.md 12.1 deviation)
  set(GTEST_SOURCE_DIR "/usr/src/googletest" CACHE PATH "GoogleTest sources")
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  add_subdirectory(${GTEST_SOURCE_DIR} ${CMAKE_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)
  include(GoogleTest)

  set(_test_sources
    tests/test_pose_math.cpp tests/test_safety_gate.cpp tests/test_waypoint_trajectory.cpp
    tests/test_impedance_controller.cpp tests/test_homing_motion.cpp tests/test_watchdog.cpp
    tests/test_state_machine.cpp tests/test_supervisor.cpp tests/test_control_loop.cpp tests/test_triple_buffer.cpp
    tests/test_shared_state.cpp tests/test_codec.cpp tests/test_command_server.cpp tests/test_state_publisher.cpp
    tests/test_config_loader.cpp tests/test_arm_status.cpp tests/test_gripper_worker.cpp tests/test_fake_arm_run.cpp
    tests/support/test_support.cpp)
  add_executable(franka_control_tests ${_test_sources})
  target_include_directories(franka_control_tests PRIVATE tests)
  target_link_libraries(franka_control_tests PRIVATE franka_control_core gtest gtest_main)
  target_compile_options(franka_control_tests PRIVATE ${_warnings})
  target_compile_definitions(franka_control_tests PRIVATE
    FRANKA_CONTROL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    FRANKA_CONTROL_TEST_TMP="${CMAKE_BINARY_DIR}/test_tmp")
  if(FRANKA_CONTROL_TSAN)
    # GCC 9's TSan can't start under the high mmap ASLR entropy of recent kernels ("unexpected memory mapping");
    # running the tests with ASLR off (setarch -R, no root needed) avoids it.
    set_target_properties(franka_control_tests PROPERTIES CROSSCOMPILING_EMULATOR "setarch;${CMAKE_SYSTEM_PROCESSOR};-R")
  endif()
  # Tests whose names contain "Threads" get the `threads` label (run under TSan in build-tsan/).
  set(_test_env "TSAN_OPTIONS=suppressions=${CMAKE_CURRENT_SOURCE_DIR}/tests/tsan.supp halt_on_error=1")
  gtest_discover_tests(franka_control_tests
    TEST_FILTER "*Threads*"
    PROPERTIES LABELS threads ENVIRONMENT "${_test_env}"
    DISCOVERY_TIMEOUT 60)
  gtest_discover_tests(franka_control_tests
    TEST_FILTER "-*Threads*"
    PROPERTIES ENVIRONMENT "${_test_env}"
    DISCOVERY_TIMEOUT 60)
endif()
