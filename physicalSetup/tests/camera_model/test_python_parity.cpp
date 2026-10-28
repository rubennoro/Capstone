// SPDX-License-Identifier: MIT
// T4: C++ models vs sensor_rig/sensor_rig/camera_model.py on random points (gen_golden.py write_parity), 1e-9.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "rig/camera_model/camera_model.hpp"

namespace rig {
namespace {

constexpr double kTol = 1e-9;

CameraModelParams parse_line(const std::string& line) {
  std::istringstream is(line);
  CameraModelParams p;
  int kind = 0, n = 0;
  is >> kind >> p.width >> p.height >> p.fx >> p.fy >> p.cx >> p.cy >> p.fov_max_deg >> n;
  p.kind = static_cast<CameraModelKind>(kind);
  p.distortion.resize(static_cast<size_t>(n));
  for (auto& d : p.distortion) is >> d;
  return p;
}

// libstdc++ operator>> rejects "nan"; read tokens and convert with strtod (exact round trip of repr(float)).
double num(std::istream& is) {
  std::string t;
  is >> t;
  EXPECT_TRUE(static_cast<bool>(is)) << "parity file truncated";
  char* end = nullptr;
  const double v = std::strtod(t.c_str(), &end);
  EXPECT_TRUE(end != t.c_str() && *end == '\0') << "bad number '" << t << "'";
  return v;
}

struct Stats {
  long n_proj = 0, n_proj_valid = 0, n_unproj = 0, n_unproj_valid = 0;
  double max_uv_err = 0, max_ray_err = 0;
};

void run_parity(const std::string& path, long min_points) {
  std::ifstream f(path);
  ASSERT_TRUE(f.good()) << path;
  std::string line;
  std::getline(f, line);  // comment
  int n_models = 0;
  f >> n_models;
  ASSERT_GE(n_models, 5);
  Stats total;
  for (int m = 0; m < n_models; ++m) {
    std::string name, mline;
    f >> name;
    std::getline(f, mline);  // rest of the name line
    std::getline(f, mline);
    SCOPED_TRACE(name);
    auto model = make_camera_model(parse_line(mline));
    ASSERT_TRUE(model.ok()) << model.status().to_string();
    const bool virt = (*model)->kind() == CameraModelKind::kVirtualPinhole;
    Stats s;
    long n = 0;
    f >> n;
    for (long i = 0; i < n; ++i) {
      const double X = num(f), Y = num(f), Z = num(f);
      const int ok = static_cast<int>(num(f));
      const double u = num(f), v = num(f);
      Eigen::Vector2d uv;
      bool cok = (*model)->project(Eigen::Vector3d(X, Y, Z), &uv);
      // camera_model.py:372: VirtualPinhole.project's flag includes the [0, W-1] x [0, H-1] image check.
      if (virt) cok = cok && uv.x() >= 0 && uv.y() >= 0 && uv.x() <= (*model)->width() - 1 &&
                      uv.y() <= (*model)->height() - 1;
      ASSERT_EQ(cok, ok != 0) << "project flag at point " << i;
      ++s.n_proj;
      if (ok) {
        ++s.n_proj_valid;
        s.max_uv_err = std::max({s.max_uv_err, std::abs(uv.x() - u), std::abs(uv.y() - v)});
      }
    }
    f >> n;
    for (long i = 0; i < n; ++i) {
      const double u = num(f), v = num(f);
      const int ok = static_cast<int>(num(f));
      const double rx = num(f), ry = num(f), rz = num(f);
      Eigen::Vector3d ray;
      const bool cok = (*model)->unproject(Eigen::Vector2d(u, v), &ray);
      ASSERT_EQ(cok, ok != 0) << "unproject flag at pixel " << i;
      ++s.n_unproj;
      if (ok) {
        ++s.n_unproj_valid;
        s.max_ray_err = std::max({s.max_ray_err, std::abs(ray.x() - rx), std::abs(ray.y() - ry), std::abs(ray.z() - rz)});
      }
    }
    EXPECT_LE(s.max_uv_err, kTol);
    EXPECT_LE(s.max_ray_err, kTol);
    EXPECT_GT(s.n_proj_valid, 0);
    EXPECT_GT(s.n_unproj_valid, 0);
    std::printf("%-20s project %ld/%ld valid, max |duv| %.3g px; unproject %ld/%ld valid, max |dray| %.3g\n",
                name.c_str(), s.n_proj_valid, s.n_proj, s.max_uv_err, s.n_unproj_valid, s.n_unproj, s.max_ray_err);
    total.n_proj += s.n_proj;
  }
  EXPECT_GE(total.n_proj, min_points);
}

TEST(CameraModelParity, CommittedRandomPoints) {
  run_parity(std::string(RIG_SOURCE_DIR) + "/tests/camera_model/data/parity.txt", 5 * 120);
}

TEST(CameraModelParity, ExtendedDense) {
  const char* d = std::getenv("RIG_T4_EXTENDED_DIR");
  if (!d) GTEST_SKIP() << "RIG_T4_EXTENDED_DIR not set";
  run_parity(std::string(d) + "/parity_dense.txt", 5 * 20000);
}

}  // namespace
}  // namespace rig
