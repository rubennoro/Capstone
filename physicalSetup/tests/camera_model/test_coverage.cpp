// SPDX-License-Identifier: MIT
// T4: the nominal 160 deg uw model covers the virtual iPhone ultrawide at 100 % (PIPELINE 2); coverage report.
#include <gtest/gtest.h>

#include "rig/camera_model/coverage.hpp"

namespace rig {
namespace {

CameraModelParams uw_nominal() {
  CameraModelParams p;
  p.kind = CameraModelKind::kKannalaBrandt4;
  p.width = 1640;
  p.height = 1232;
  p.fx = p.fy = 735.0;
  p.cx = 819.5;
  p.cy = 615.5;
  p.distortion = {0, 0, 0, 0};
  p.fov_max_deg = 160.0;
  return p;
}
CameraModelParams iphone_uw() {  // firstreal session JSON ultrawideCameraIntrinsics
  CameraModelParams p;
  p.kind = CameraModelKind::kVirtualPinhole;
  p.width = 640;
  p.height = 480;
  p.fx = p.fy = 246.64919;
  p.cx = 320.16495;
  p.cy = 239.53679;
  return p;
}

TEST(Coverage, Nominal160CoversIphoneUltrawide) {
  auto phys = make_camera_model(uw_nominal());
  auto virt = VirtualPinhole::create(iphone_uw());
  ASSERT_TRUE(phys.ok() && virt.ok());
  auto rep = coverage_report(**phys, **virt);
  ASSERT_TRUE(rep.ok());
  EXPECT_EQ(rep->n_total, 640 * 480);
  EXPECT_EQ(rep->n_valid, rep->n_total);
  EXPECT_EQ(rep->coverage, 1.0);
  EXPECT_GT(rep->min_margin_px, 40.0);  // ~49 px vertical slack with the nominal model
  EXPECT_LT(rep->max_ray_angle_deg, 80.0);
  std::printf("nominal 160 deg -> iPhone uw: coverage %.6f, min margin %.1f px, max ray angle %.2f deg, "
              "src u [%.1f, %.1f] v [%.1f, %.1f]\n",
              rep->coverage, rep->min_margin_px, rep->max_ray_angle_deg, rep->src_u_min, rep->src_u_max,
              rep->src_v_min, rep->src_v_max);
  auto t = build_remap(**phys, **virt);
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t->coverage, 1.0);
  EXPECT_EQ(cv::countNonZero(t->valid_mask), 640 * 480);
}

TEST(Coverage, ReportMatchesTableForPartialCoverage) {
  CameraModelParams main;
  main.kind = CameraModelKind::kPinholeRadtan;
  main.width = 1920;
  main.height = 1080;
  main.fx = main.fy = 1478.0;
  main.cx = 959.5;
  main.cy = 539.5;
  main.distortion = {0, 0, 0, 0};
  auto phys = make_camera_model(main);
  auto virt = VirtualPinhole::create(iphone_uw());
  auto rep = coverage_report(**phys, **virt);
  auto t = build_remap(**phys, **virt, Eigen::Matrix3d::Identity(), false);
  ASSERT_TRUE(rep.ok() && t.ok());
  EXPECT_LT(rep->coverage, 0.5);
  EXPECT_EQ(rep->coverage, t->coverage);
  EXPECT_EQ(rep->n_valid + rep->n_outside + rep->n_model_invalid, rep->n_total);
  EXPECT_EQ(t->map1.type(), CV_32FC1);
  EXPECT_EQ(t->map1.at<float>(0, 0), -16.0f);  // camera_model.py:36
}

TEST(Coverage, RejectsNonRotation) {
  auto phys = make_camera_model(uw_nominal());
  auto virt = VirtualPinhole::create(iphone_uw());
  Eigen::Matrix3d S = 2.0 * Eigen::Matrix3d::Identity();
  EXPECT_EQ(build_remap(**phys, **virt, S).status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(coverage_report(**phys, **virt, S).status().code(), StatusCode::kInvalidArgument);
  cv::Mat dst;
  RemapTable empty;
  EXPECT_FALSE(empty.apply(cv::Mat(4, 4, CV_8UC3), dst).ok());
}

}  // namespace
}  // namespace rig
