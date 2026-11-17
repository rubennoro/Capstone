# dataproc_umi_slam against a built UMI ORB-SLAM3 fork (github.com/cheng-chi/ORB_SLAM3).
#   cmake -S data-processing/native/umi_slam -B <build> -DORB_SLAM3_DIR=$HOME/ORB_SLAM3_umi
#   cmake --build <build> -j8
cmake_minimum_required(VERSION 3.10)
project(dataproc_umi_slam CXX)

set(ORB_SLAM3_DIR "$ENV{HOME}/ORB_SLAM3_umi" CACHE PATH "built UMI ORB-SLAM3 fork")
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 14)                    # as the fork
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")

find_package(OpenCV 4.0 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED HINTS "${ORB_SLAM3_DIR}/Thirdparty/Pangolin/build")

add_executable(dataproc_umi_slam dataproc_umi_slam.cc)
target_include_directories(dataproc_umi_slam PRIVATE
  ${ORB_SLAM3_DIR} ${ORB_SLAM3_DIR}/include ${ORB_SLAM3_DIR}/include/CameraModels ${ORB_SLAM3_DIR}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR} ${Pangolin_INCLUDE_DIRS})
target_link_libraries(dataproc_umi_slam
  ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so ${OpenCV_LIBS} ${Pangolin_LIBRARIES}
  ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib/libg2o.so
  -lboost_serialization -lcrypto)
