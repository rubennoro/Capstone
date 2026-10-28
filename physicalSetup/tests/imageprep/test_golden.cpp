// SPDX-License-Identifier: MIT
// T4 golden: letterbox_square_resize vs iPhUMI cv_util.py:5-34 run in Python cv2 4.12 (gen_golden.py) on crops of
// real firstreal iPhone frames. Requirement: bit-exact (hash of the output bytes).
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include "golden_util.hpp"
#include "rig/imageprep/imageprep.hpp"

namespace rig {
namespace {

struct Case {
  std::string name;
  int out = 0, swap = 0, rows = 0, cols = 0, ch = 0;
  std::string hash, expected_png;
};

std::vector<Case> read_cases(const std::string& path, bool extended) {
  std::ifstream f(path);
  std::vector<Case> cases;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    Case c;
    is >> c.name >> c.out >> c.swap >> c.rows >> c.cols >> c.ch >> c.hash;
    if (extended) is >> c.expected_png;
    cases.push_back(c);
  }
  return cases;
}

TEST(ImagePrepGolden, CommittedFixturesBitExact) {
  const std::string dir = test::imageprep_data_dir();
  const auto cases = read_cases(dir + "/golden.txt", false);
  ASSERT_GE(cases.size(), 20u);
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name + " out=" + std::to_string(c.out) + " swap=" + std::to_string(c.swap));
    const cv::Mat src = cv::imread(dir + "/" + c.name + ".png", cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(src.empty());
    auto out = letterbox_square_resize(src, c.out, c.swap != 0);
    ASSERT_TRUE(out.ok()) << out.status().to_string();
    EXPECT_EQ(out->rows, c.rows);
    EXPECT_EQ(out->cols, c.cols);
    EXPECT_EQ(out->channels(), c.ch);
    EXPECT_EQ(test::fnv1a64_hex(*out), c.hash);
  }
}

// Full-size frames (1920x1440 main, 640x480 uw) from 3 firstreal episodes; needs RIG_T4_EXTENDED_DIR.
TEST(ImagePrepGolden, ExtendedRealFramesBitExact) {
  const std::string dir = test::extended_dir();
  if (dir.empty()) GTEST_SKIP() << "RIG_T4_EXTENDED_DIR not set (gen_golden.py extended)";
  const auto cases = read_cases(dir + "/golden_ext.txt", true);
  ASSERT_FALSE(cases.empty());
  int n = 0;
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name + " swap=" + std::to_string(c.swap));
    const cv::Mat src = cv::imread(dir + "/" + c.name, cv::IMREAD_COLOR);
    ASSERT_FALSE(src.empty());
    auto out = letterbox_square_resize(src, c.out, c.swap != 0);
    ASSERT_TRUE(out.ok());
    const cv::Mat exp = cv::imread(dir + "/" + c.expected_png, cv::IMREAD_COLOR);
    ASSERT_FALSE(exp.empty());
    // The expected PNG holds the Python bytes as-is (PNG is lossless; imread returns them unchanged).
    double maxdiff = cv::norm(*out, exp, cv::NORM_INF);
    EXPECT_EQ(maxdiff, 0.0);
    EXPECT_EQ(test::fnv1a64_hex(*out), c.hash);
    ++n;
  }
  std::printf("extended letterbox cases: %d\n", n);
}

TEST(ImagePrep, GeometryMatchesCvUtil) {
  // cv_util.py:13-17 examples.
  constexpr auto g = letterbox_geometry(1920, 1440);
  static_assert(g.size == 1920 && g.top == 240 && g.bottom == 240 && g.left == 0 && g.right == 0);
  constexpr auto h = letterbox_geometry(211, 158);
  static_assert(h.size == 211 && h.top == 26 && h.bottom == 27 && h.left == 0 && h.right == 0);
  constexpr auto v = letterbox_geometry(158, 211);
  static_assert(v.left == 26 && v.right == 27 && v.top == 0);
  SUCCEED();
}

}  // namespace
}  // namespace rig
