# umiproc_slam against a built UMI ORB-SLAM3 fork (github.com/cheng-chi/ORB_SLAM3 + native/patches). native/build.sh
# runs this; by hand:
#   cmake -S umi-processing/native/umi_slam -B $HOME/ORB_SLAM3_umi/umiproc_build -DORB_SLAM3_DIR=$HOME/ORB_SLAM3_umi \
#         -DPangolin_DIR=$HOME/ORB_SLAM3_umi/Thirdparty/Pangolin/build [-DOpenCV_DIR=...]
#   cmake --build $HOME/ORB_SLAM3_umi/umiproc_build -j2
cmake_minimum_required(VERSION 3.10)
project(umiproc_slam CXX)

set(ORB_SLAM3_DIR "$ENV{HOME}/ORB_SLAM3_umi" CACHE PATH "built UMI ORB-SLAM3 fork")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 14)                    # as the fork
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")    # as the fork (Eigen alignment)

find_package(OpenCV 4.0 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
# the fork's in-tree Pangolin, never another installed one (a Jetson may have Pangolin v0.6 in /usr/local)
if(NOT Pangolin_DIR)
  set(Pangolin_DIR "${ORB_SLAM3_DIR}/Thirdparty/Pangolin/build")
endif()
find_package(Pangolin REQUIRED NO_DEFAULT_PATH PATHS "${Pangolin_DIR}")

add_executable(umiproc_slam umiproc_slam.cc)
target_include_directories(umiproc_slam PRIVATE
  ${ORB_SLAM3_DIR} ${ORB_SLAM3_DIR}/include ${ORB_SLAM3_DIR}/include/CameraModels ${ORB_SLAM3_DIR}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS})
target_link_libraries(umiproc_slam
  ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so ${OpenCV_LIBS} ${Pangolin_LIBRARIES}
  ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib/libg2o.so
  -lboost_serialization -lcrypto)
