# episode_collection: the on-Jetson programs (ec-record, ec-calibrate, ec-slam-live, ec-viz, ec-doctor, probes)
# and the offline pipeline (ec-process). PIPELINE 4 and 8.
#
# Layout (globbed, see cmake/RigHelpers.cmake):
#   include/ec/...                 public headers of the rigec library
#   record/ calib/ viz/ offline/   sources of the rigec static library (any *.cpp)
#   apps/<name>/*.cpp              one executable per directory, named <name>
#   configs/                       example rig.yaml files
set(_ec_lib_sources "")
foreach(_d record calib viz offline)
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${_d}")
    rig_glob_sources(_s "${CMAKE_CURRENT_SOURCE_DIR}/${_d}")
    list(APPEND _ec_lib_sources ${_s})
  endif()
endforeach()

if(_ec_lib_sources)
  add_library(rigec STATIC ${_ec_lib_sources})
else()
  add_library(rigec INTERFACE)  # nothing yet (Wave 2 adds record/, calib/, viz/)
endif()
add_library(rig::ec ALIAS rigec)
if(_ec_lib_sources)
  target_include_directories(rigec PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(rigec PUBLIC rig::common PRIVATE yaml-cpp nlohmann_json::nlohmann_json spdlog::spdlog
                                                         JPEG::JPEG)
else()
  target_link_libraries(rigec INTERFACE rig::common)
endif()

rig_add_apps("${CMAKE_CURRENT_SOURCE_DIR}/apps" rig::ec rig::common spdlog::spdlog nlohmann_json::nlohmann_json
             yaml-cpp)
