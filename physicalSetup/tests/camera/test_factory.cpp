// SPDX-License-Identifier: MIT
// make_camera() validation and compiled-in backends.
#include <gtest/gtest.h>

#include "rig/drivers/camera_backend.hpp"

namespace rig {
namespace {

CameraConfig base() {
  CameraConfig c;
  c.id = "x";
  c.width = 64;
  c.height = 48;
  c.fps = 30;
  return c;
}

TEST(CameraFactory, Validation) {
  CameraConfig c = base();
  c.id = "";
  EXPECT_EQ(make_camera(c).status().code(), StatusCode::kInvalidArgument);
  c = base();
  c.sensor = "imx477";
  EXPECT_NE(make_camera(c).status().message().find("unknown 'imx477'"), std::string::npos);
  c = base();
  c.fps = 0;
  EXPECT_EQ(make_camera(c).status().code(), StatusCode::kInvalidArgument);
  c = base();
  c.drop_gap_factor = 1.0;
  EXPECT_EQ(make_camera(c).status().code(), StatusCode::kInvalidArgument);
  c = base();
  c.focus.mode = FocusMode::kFixed;
  c.focus.position = 5000;
  auto r = make_camera(c);
  EXPECT_EQ(r.status().code(), StatusCode::kInvalidArgument);
  EXPECT_NE(r.status().message().find("cameras.x.driver.focus: position"), std::string::npos) << r.status().message();
  c = base();
  c.backend = CameraBackendKind::kReplay;
  EXPECT_NE(make_camera(c).status().message().find("replay.path"), std::string::npos);
  EXPECT_TRUE(make_camera(base()).ok());
}

TEST(CameraFactory, OptionalBackends) {
  CameraConfig c = base();
  c.backend = CameraBackendKind::kArgus;
  auto a = make_camera(c);
#if RIG_HAVE_ARGUS
  EXPECT_TRUE(a.ok());
#else
  EXPECT_EQ(a.status().code(), StatusCode::kUnimplemented);
#endif
  c.backend = CameraBackendKind::kGst;
  auto g = make_camera(c);
#if RIG_HAVE_GST
  EXPECT_TRUE(g.ok());
#else
  EXPECT_EQ(g.status().code(), StatusCode::kUnimplemented);
#endif
}

}  // namespace
}  // namespace rig
