// SPDX-License-Identifier: MIT
// Table-driven mode selection (PLAN T2 "mode selection logic, table driven, Arducam/NVIDIA mode tables").
#include <gtest/gtest.h>

#include "rig/drivers/camera/sensor_modes.hpp"

namespace rig {
namespace {

CameraConfig cfg(const std::string& sensor, int w, int h, double fps, int force = -1) {
  CameraConfig c;
  c.id = "cam";
  c.sensor = sensor;
  c.width = w;
  c.height = h;
  c.fps = fps;
  c.sensor_mode = force;
  return c;
}

TEST(SensorModes, TablesMatchPublishedDumps) {
  const auto& a = known_sensor_modes("imx219");
  ASSERT_EQ(a.size(), 5u);
  EXPECT_EQ(a[3].width, 1640);
  EXPECT_EQ(a[3].height, 1232);
  EXPECT_EQ(a[3].index, 3);
  EXPECT_TRUE(a[3].full_fov);
  EXPECT_DOUBLE_EQ(a[3].max_analog_gain, 10.625);
  EXPECT_EQ(a[3].min_exposure_ns, 13000);
  EXPECT_EQ(a[3].max_exposure_ns, 683709000);
  const auto& b = known_sensor_modes("imx519");
  ASSERT_EQ(b.size(), 4u);
  EXPECT_EQ(b[2].width, 1920);
  EXPECT_DOUBLE_EQ(b[2].max_analog_gain, 16.0);
  EXPECT_TRUE(known_sensor_modes("ov5647").empty());
  EXPECT_TRUE(is_known_sensor("imx519"));
  EXPECT_FALSE(is_known_sensor(""));
}

struct SelectCase {
  const char* sensor;
  int w, h;
  double fps;
  int force;
  bool ok;
  int index;             // expected mode index when ok
  const char* fragment;  // expected substring of the error otherwise
};

class SelectTest : public ::testing::TestWithParam<SelectCase> {};

TEST_P(SelectTest, Select) {
  const SelectCase& t = GetParam();
  auto r = select_sensor_mode(known_sensor_modes(t.sensor), cfg(t.sensor, t.w, t.h, t.fps, t.force));
  if (t.ok) {
    ASSERT_TRUE(r.ok()) << r.status().to_string();
    EXPECT_EQ(r->index, t.index);
    EXPECT_EQ(r->width, t.w);
  } else {
    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.status().message().find(t.fragment), std::string::npos) << r.status().message();
    EXPECT_NE(r.status().message().find("cameras.cam.driver"), std::string::npos);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Table, SelectTest,
    ::testing::Values(
        // The rig's configured modes (rig_jetson_imx519_imx219w160_lsm6dsox.yaml).
        SelectCase{"imx519", 1920, 1080, 60.0, -1, true, 2, ""},   // 59.999999 accepted with the tolerance
        SelectCase{"imx219", 1640, 1232, 30.0, -1, true, 3, ""},   // 29.999999 accepted
        SelectCase{"imx219", 1640, 1232, 15.0, -1, true, 3, ""},   // lower rate within the range
        SelectCase{"imx219", 3280, 2464, 21.0, -1, true, 0, ""},
        SelectCase{"imx519", 3840, 2160, 17.0, -1, true, 1, ""},
        // Crops fail loudly unless forced.
        SelectCase{"imx219", 1280, 720, 60.0, -1, false, -1, "cropped mode on imx219"},
        SelectCase{"imx219", 1920, 1080, 30.0, -1, false, -1, "cropped mode on imx219"},
        SelectCase{"imx519", 1280, 720, 60.0, -1, false, -1, "cropped mode on imx519"},
        SelectCase{"imx219", 1280, 720, 60.0, 4, true, 4, ""},     // forced crop is allowed
        // Rate above the mode maximum, unknown size, bad forced index.
        SelectCase{"imx219", 1640, 1232, 60.0, -1, false, -1, "no sensor mode 1640x1232@60"},
        SelectCase{"imx519", 1920, 1080, 61.0, -1, false, -1, "no sensor mode"},
        SelectCase{"imx519", 1000, 1000, 30.0, -1, false, -1, "available modes: 0: 4656x3496"},
        SelectCase{"imx519", 1920, 1080, 60.0, 7, false, -1, "sensor_mode 7 does not exist"},
        SelectCase{"imx519", 1920, 1080, 60.0, 3, false, -1, "sensor_mode 3 is 1280x720"},
        SelectCase{"imx519", 1280, 720, 121.0, 3, false, -1, "cannot run at"}));

TEST(SensorModes, FpsTolerance) {
  SensorMode m;
  m.min_fps = 1.0;
  m.max_fps = 59.999999;
  EXPECT_TRUE(mode_supports_fps(m, 60.0));
  EXPECT_FALSE(mode_supports_fps(m, 60.1));
  EXPECT_FALSE(mode_supports_fps(m, 0.0));
  EXPECT_FALSE(mode_supports_fps(m, 0.5));
}

TEST(SensorModes, FrameDuration) {
  EXPECT_EQ(frame_duration_ns(60.0), 16666667);
  EXPECT_EQ(frame_duration_ns(30.0), 33333333);
  EXPECT_EQ(frame_duration_ns(0.0), 0);
}

TEST(SensorModes, AnnotateDriverModes) {
  std::vector<SensorMode> driver(3);
  driver[0].width = 1640, driver[0].height = 1232, driver[0].index = 3;
  driver[1].width = 1280, driver[1].height = 720, driver[1].index = 4;
  driver[2].width = 800, driver[2].height = 600, driver[2].index = 5, driver[2].full_fov = true;
  annotate_sensor_modes("imx219", driver);
  EXPECT_TRUE(driver[0].full_fov);
  EXPECT_FALSE(driver[1].full_fov);
  EXPECT_FALSE(driver[2].full_fov);
  EXPECT_NE(driver[2].description.find("not in the T2 table"), std::string::npos);
  std::vector<SensorMode> other(1);
  other[0].full_fov = true;
  annotate_sensor_modes("unknown", other);
  EXPECT_TRUE(other[0].full_fov);
}

TEST(SensorModes, Format) {
  const std::string s = format_sensor_mode(known_sensor_modes("imx219")[3]);
  EXPECT_NE(s.find("3: 1640x1232"), std::string::npos) << s;
  EXPECT_NE(s.find("full FOV"), std::string::npos);
  EXPECT_EQ(format_sensor_modes({}), "(no modes)");
}

}  // namespace
}  // namespace rig
