# Source-tree conventions for physicalSetup (README.md "Adding code").
#
# Sources are globbed (CONFIGURE_DEPENDS) so that tasks working in parallel never edit the same CMakeLists.txt.
# Directories named after an optional dependency are compiled only when that option is ON:
#   .../argus/...     RIG_WITH_ARGUS      .../gst/...       RIG_WITH_GST
#   .../orbslam3/...  RIG_WITH_ORBSLAM3   .../ceres/...     RIG_WITH_CERES
# Directories named python/ and include/ are never globbed as library sources.

# rig_glob_sources(<out_var> <dir>): every *.cpp under <dir>, minus disabled optional-dependency directories.
function(rig_glob_sources out_var dir)
  file(GLOB_RECURSE _srcs CONFIGURE_DEPENDS "${dir}/*.cpp")
  set(_excluded "/python/" "/include/")
  if(NOT RIG_WITH_ARGUS)
    list(APPEND _excluded "/argus/")
  endif()
  if(NOT RIG_WITH_GST)
    list(APPEND _excluded "/gst/")
  endif()
  if(NOT RIG_WITH_ORBSLAM3)
    list(APPEND _excluded "/orbslam3/")
  endif()
  if(NOT RIG_WITH_CERES)
    list(APPEND _excluded "/ceres/")
  endif()
  set(_out "")
  foreach(_f IN LISTS _srcs)
    file(RELATIVE_PATH _rel "${dir}" "${_f}")
    set(_keep TRUE)
    foreach(_ex IN LISTS _excluded)
      string(FIND "/${_rel}" "${_ex}" _pos)
      if(NOT _pos EQUAL -1)
        set(_keep FALSE)
      endif()
    endforeach()
    if(_keep)
      list(APPEND _out "${_f}")
    endif()
  endforeach()
  set(${out_var} "${_out}" PARENT_SCOPE)
endfunction()

# rig_link_optional_deps(<target>): links the optional-dependency targets that are enabled and sets the
# RIG_HAVE_* compile definitions (0/1) PUBLIC so headers and factories can test them.
function(rig_link_optional_deps target)
  foreach(_opt ARGUS GST ORBSLAM3 CERES PYTHON)
    if(RIG_WITH_${_opt})
      target_compile_definitions(${target} PUBLIC RIG_HAVE_${_opt}=1)
    else()
      target_compile_definitions(${target} PUBLIC RIG_HAVE_${_opt}=0)
    endif()
  endforeach()
  if(RIG_WITH_ARGUS)
    target_link_libraries(${target} PUBLIC rig_argus)
  endif()
  if(RIG_WITH_GST)
    target_link_libraries(${target} PUBLIC PkgConfig::RIG_GST)
  endif()
  if(RIG_WITH_ORBSLAM3)
    target_link_libraries(${target} PUBLIC rig_orbslam3)
  endif()
  if(RIG_WITH_CERES)
    target_link_libraries(${target} PUBLIC Ceres::ceres)
  endif()
endfunction()

# rig_add_test_area(<dir> <libs...>): tests/<area>/test_*.cpp -> one gtest executable each, named
# test_<area>_<name>. Other *.cpp files in the area (helpers, simulators used only by tests) are compiled into a
# per-area object library linked into every test of the area. Optional-dependency subdirectories follow the same
# rule as rig_glob_sources.
function(rig_add_test_area dir)
  set(_libs ${ARGN})
  get_filename_component(_area "${dir}" NAME)
  rig_glob_sources(_all "${dir}")
  set(_tests "")
  set(_helpers "")
  foreach(_f IN LISTS _all)
    get_filename_component(_n "${_f}" NAME)
    if(_n MATCHES "^test_.*\\.cpp$")
      list(APPEND _tests "${_f}")
    else()
      list(APPEND _helpers "${_f}")
    endif()
  endforeach()
  set(_helper_target "")
  if(_helpers)
    set(_helper_target "rig_test_helpers_${_area}")
    add_library(${_helper_target} STATIC ${_helpers})
    target_link_libraries(${_helper_target} PUBLIC ${_libs} GTest::gtest)
    target_include_directories(${_helper_target} PUBLIC "${dir}")
    rig_test_definitions(${_helper_target})
  endif()
  foreach(_t IN LISTS _tests)
    get_filename_component(_name "${_t}" NAME_WE)
    string(REGEX REPLACE "^test_" "" _name "${_name}")
    set(_exe "test_${_area}_${_name}")
    add_executable(${_exe} "${_t}")
    target_link_libraries(${_exe} PRIVATE ${_helper_target} ${_libs} GTest::gtest GTest::gtest_main)
    target_include_directories(${_exe} PRIVATE "${dir}")
    rig_test_definitions(${_exe})
    gtest_discover_tests(${_exe} DISCOVERY_MODE PRE_TEST DISCOVERY_TIMEOUT 60
                         PROPERTIES LABELS "${_area}" ENVIRONMENT "RIG_TEST_AREA=${_area}")
  endforeach()
endfunction()

# Paths every test may use: RIG_SOURCE_DIR (physicalSetup/), RIG_REPO_ROOT (the git repo, for iPhUMI golden data),
# RIG_TEST_TMP_ROOT (a writable directory inside the build tree).
function(rig_test_definitions target)
  get_filename_component(_repo "${PROJECT_SOURCE_DIR}/.." ABSOLUTE)
  file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/test_tmp")
  target_compile_definitions(${target} PRIVATE RIG_SOURCE_DIR="${PROJECT_SOURCE_DIR}" RIG_REPO_ROOT="${_repo}"
                                               RIG_TEST_TMP_ROOT="${PROJECT_BINARY_DIR}/test_tmp")
endfunction()

# rig_add_apps(<apps_dir> <libs...>): every subdirectory <apps_dir>/<name>/ with *.cpp files becomes an executable
# <name> (e.g. apps/ec-imu-probe/ -> ec-imu-probe), linked with <libs> and CLI11. Optional-dependency directory
# names apply to whole apps too (an app directory named e.g. ec-slam-live/orbslam3/... is filtered per file).
function(rig_add_apps apps_dir)
  set(_libs ${ARGN})
  if(NOT IS_DIRECTORY "${apps_dir}")
    return()
  endif()
  file(GLOB _children CONFIGURE_DEPENDS LIST_DIRECTORIES true "${apps_dir}/*")
  foreach(_c IN LISTS _children)
    if(IS_DIRECTORY "${_c}")
      get_filename_component(_name "${_c}" NAME)
      rig_glob_sources(_srcs "${_c}")
      if(_srcs)
        add_executable(${_name} ${_srcs})
        target_link_libraries(${_name} PRIVATE ${_libs} rig::cli11)
        target_include_directories(${_name} PRIVATE "${_c}")
      endif()
    endif()
  endforeach()
endfunction()
