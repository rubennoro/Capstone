# rigpy (T4): pybind11 bindings of rig::camera_model and rig::imageprep. Built only with RIG_WITH_PYTHON=ON
# (root CMakeLists finds Python3 + pybind11). Configure with the interpreter the module is for, e.g.
#   -DRIG_WITH_PYTHON=ON -DPython3_EXECUTABLE=<venv>/bin/python -Dpybind11_DIR=$(<venv>/bin/python -m pybind11 --cmakedir)
pybind11_add_module(rigpy MODULE rigpy.cpp)
target_link_libraries(rigpy PRIVATE rig::common)

if(RIG_BUILD_TESTS)
  # Python-side test: rigpy reproduces the committed golden bytes and Python cv2 / camera_model.py (needs numpy +
  # cv2 in that interpreter, i.e. the T4 venv).
  add_test(NAME rigpy.bindings
           COMMAND ${Python3_EXECUTABLE} -I ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rigpy.py
                   --module-dir $<TARGET_FILE_DIR:rigpy> --source-dir ${PROJECT_SOURCE_DIR})
  set_tests_properties(rigpy.bindings PROPERTIES LABELS "python")
endif()
