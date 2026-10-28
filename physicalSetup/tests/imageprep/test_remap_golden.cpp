// SPDX-License-Identifier: MIT
// T4 golden: build_remap + RemapTable::apply + letterbox vs sensor_rig camera_model.py build_remap and cv2 4.12
// (gen_golden.py). Maps, valid mask, remapped image and 224 output must be bit-identical.
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "golden_util.hpp"
#include "rig/camera_model/camera_model.hpp"
#include "rig/imageprep/imageprep.hpp"

namespace rig {
namespace {

Eigen::Matrix3d parse_R(const std::string& s) {
  std::istringstream is(s);
  Eigen::Matrix3d R;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) is >> R(i, j);
  return R;
}

std::string trim(const std::string& s) {
  const auto a = s.find_first_not_of(' '), b = s.find_last_not_of(' ');
  return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

TEST(RemapGolden, CommittedSynthBitExact) {
  std::ifstream f(test::imageprep_data_dir() + "/remap_golden.txt");
  ASSERT_TRUE(f.good());
  auto virt = VirtualPinhole::create(test::iphone_uw_virtual());
  ASSERT_TRUE(virt.ok());
  std::string line;
  int n = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto parts = test::split(line, " | ");
    ASSERT_EQ(parts.size(), 8u);
    SCOPED_TRACE(parts[0]);
    auto phys = make_camera_model(test::parse_model_line(parts[1]));
    ASSERT_TRUE(phys.ok()) << phys.status().to_string();
    const Eigen::Matrix3d R = parse_R(parts[2]);
    auto t = build_remap(**phys, **virt, R, true);
    ASSERT_TRUE(t.ok()) << t.status().to_string();
    std::istringstream cov(parts[3]);
    double coverage = 0;
    long n_valid = 0;
    cov >> coverage >> n_valid;
    EXPECT_EQ(cv::countNonZero(t->valid_mask), n_valid);
    EXPECT_NEAR(t->coverage, coverage, 1e-15);
    std::istringstream hs(parts[4]);
    std::string h1, h2, hv;
    hs >> h1 >> h2 >> hv;
    EXPECT_EQ(t->map1.type(), CV_16SC2);
    EXPECT_EQ(t->map2.type(), CV_16UC1);
    EXPECT_EQ(test::fnv1a64_hex(t->map1), h1);
    EXPECT_EQ(test::fnv1a64_hex(t->map2), h2);
    EXPECT_EQ(test::fnv1a64_hex(t->valid_mask), hv);
    const cv::Mat src = test::synth_image((*phys)->width(), (*phys)->height());
    cv::Mat out;
    ASSERT_TRUE(t->apply(src, out).ok());
    EXPECT_EQ(test::fnv1a64_hex(out), trim(parts[5]));
    auto lb = letterbox_square_resize(out, 224, false);
    auto lbs = letterbox_square_resize(out, 224, true);
    ASSERT_TRUE(lb.ok() && lbs.ok());
    EXPECT_EQ(test::fnv1a64_hex(*lb), trim(parts[6]));
    EXPECT_EQ(test::fnv1a64_hex(*lbs), trim(parts[7]));
    ++n;
  }
  EXPECT_EQ(n, 3);
}

// Real-texture sources and a float-map comparison; needs RIG_T4_EXTENDED_DIR.
TEST(RemapGolden, ExtendedRealTextureAndFloatMaps) {
  const std::string dir = test::extended_dir();
  if (dir.empty()) GTEST_SKIP() << "RIG_T4_EXTENDED_DIR not set (gen_golden.py extended)";
  auto virt = VirtualPinhole::create(test::iphone_uw_virtual());
  ASSERT_TRUE(virt.ok());
  std::ifstream f(dir + "/remap_ext.txt");
  ASSERT_TRUE(f.good());
  std::string line;
  int n = 0;
  std::set<std::string> map_checked;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    auto parts = test::split(line, " | ");
    ASSERT_EQ(parts.size(), 7u);
    SCOPED_TRACE(parts[0] + " " + parts[3]);
    auto phys = make_camera_model(test::parse_model_line(parts[1]));
    ASSERT_TRUE(phys.ok());
    const Eigen::Matrix3d R = parse_R(parts[2]);
    if (!map_checked.count(parts[0])) {
      map_checked.insert(parts[0]);
      auto tf = build_remap(**phys, **virt, R, false);
      ASSERT_TRUE(tf.ok());
      const auto read_f32 = [&](const std::string& p) {
        cv::Mat m(tf->dst_size, CV_32FC1);
        std::ifstream b(p, std::ios::binary);
        b.read(reinterpret_cast<char*>(m.data), static_cast<std::streamsize>(m.total() * 4));
        EXPECT_TRUE(b.good()) << p;
        return m;
      };
      const cv::Mat px = read_f32(dir + "/map_" + parts[0] + "_x.f32");
      const cv::Mat py = read_f32(dir + "/map_" + parts[0] + "_y.f32");
      cv::Mat dx, dy;
      cv::compare(px, tf->map1, dx, cv::CMP_NE);
      cv::compare(py, tf->map2, dy, cv::CMP_NE);
      EXPECT_EQ(cv::countNonZero(dx), 0) << "float32 x map entries differing from Python";
      EXPECT_EQ(cv::countNonZero(dy), 0) << "float32 y map entries differing from Python";
    }
    auto t = build_remap(**phys, **virt, R, true);
    ASSERT_TRUE(t.ok());
    const cv::Mat src = cv::imread(dir + "/" + parts[3], cv::IMREAD_COLOR);
    ASSERT_FALSE(src.empty());
    cv::Mat out;
    ASSERT_TRUE(t->apply(src, out).ok());
    EXPECT_EQ(test::fnv1a64_hex(out), parts[4]);
    auto lb = letterbox_square_resize(out, 224, true);
    ASSERT_TRUE(lb.ok());
    EXPECT_EQ(test::fnv1a64_hex(*lb), parts[5]);
    const cv::Mat exp = cv::imread(dir + "/" + trim(parts[6]), cv::IMREAD_COLOR);
    EXPECT_EQ(cv::norm(*lb, exp, cv::NORM_INF), 0.0);
    ++n;
  }
  EXPECT_GT(n, 0);
  std::printf("extended remap cases: %d\n", n);
}

}  // namespace
}  // namespace rig
