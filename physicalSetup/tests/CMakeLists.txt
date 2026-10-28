# Unit tests. Each tests/<area>/ is one area (see rig_add_test_area in cmake/RigHelpers.cmake):
#   tests/<area>/test_<name>.cpp -> executable test_<area>_<name>; other .cpp files = area helpers.
# Areas link rigcommon and rigec. The header self-sufficiency check is generated below.

file(GLOB _areas CONFIGURE_DEPENDS LIST_DIRECTORIES true "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach(_a IN LISTS _areas)
  if(IS_DIRECTORY "${_a}")
    rig_add_test_area("${_a}" rig::ec rig::common rig::cli11)
  endif()
endforeach()

# Header self-sufficiency: one TU per public header that includes only that header (twice, to check the
# include guard), compiled into an object library that is part of the default (ALL) build, so a header that is
# not self-sufficient fails the build and therefore check.sh.
file(GLOB_RECURSE _rig_public_headers CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}/common/include"
     "${PROJECT_SOURCE_DIR}/common/include/rig/*.hpp")
set(_hc_sources "")
foreach(_h IN LISTS _rig_public_headers)
  string(MAKE_C_IDENTIFIER "${_h}" _id)
  set(_tu "${CMAKE_CURRENT_BINARY_DIR}/header_check/${_id}.cpp")
  set(_content "// Generated: header self-sufficiency check for <${_h}>.\n#include \"${_h}\"\n#include \"${_h}\"\n")
  if(EXISTS "${_tu}")
    file(READ "${_tu}" _old)
  else()
    set(_old "")
  endif()
  if(NOT _old STREQUAL _content)
    file(WRITE "${_tu}" "${_content}")
  endif()
  list(APPEND _hc_sources "${_tu}")
endforeach()
add_library(rig_header_check OBJECT ${_hc_sources})
target_link_libraries(rig_header_check PRIVATE rig::common)
