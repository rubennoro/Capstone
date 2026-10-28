# rigcommon: the shared library of collection and deployment (PIPELINE 8).
# Sources: every *.cpp under common/<module>/ (globbed; see cmake/RigHelpers.cmake). Public headers:
# common/include/rig/<module>/*.hpp. Private headers live next to their sources.
rig_glob_sources(RIGCOMMON_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(rigcommon STATIC ${RIGCOMMON_SOURCES})
add_library(rig::common ALIAS rigcommon)
target_include_directories(rigcommon
  PUBLIC  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/generated"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(rigcommon
  PUBLIC  Eigen3::Eigen ${OpenCV_LIBS} Threads::Threads
  PRIVATE yaml-cpp nlohmann_json::nlohmann_json spdlog::spdlog JPEG::JPEG rt)
# OpenCV's imported targets carry their include dirs; keep them SYSTEM for -Werror.
target_include_directories(rigcommon SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
rig_link_optional_deps(rigcommon)

if(RIG_WITH_PYTHON AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/python/CMakeLists.txt")
  add_subdirectory(python)  # T4: rigpy
endif()
