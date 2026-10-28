// SPDX-License-Identifier: MIT
// Control plan (locks, ranges) and GStreamer launch strings.
#include <gtest/gtest.h>

#include "rig/drivers/camera/capture_plan.hpp"
#include "rig/drivers/camera/sensor_modes.hpp"

namespace rig {
namespace {

// The example rig.yaml cameras (episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml).
CameraConfig main_cfg() {
  CameraConfig c;
  c.id = "main";
  c.sensor = "imx519";
  c.width = 1920;
  c.height = 1080;
  c.fps = 60.0;
  c.exposure_ns = 8000000;
  c.analog_gain = 2.0;
  return c;
}
CameraConfig uw_cfg() {
  CameraConfig c;
  c.id = "uw";
  c.sensor = "imx219";
  c.sensor_id = 1;
  c.width = 1640;
  c.height = 1232;
  c.fps = 30.0;
  c.exposure_ns = 4000000;
  c.analog_gain = 4.0;
  return c;
}
SensorMode mode_of(const CameraConfig& c) { return select_sensor_mode(known_sensor_modes(c.sensor), c).value(); }

TEST(CapturePlan, MainLocked) {
  const CameraConfig c = main_cfg();
  auto p = build_capture_plan(c, mode_of(c));
  ASSERT_TRUE(p.ok()) << p.status().to_string();
  EXPECT_EQ(p->frame_duration_ns, 16666667);
  EXPECT_EQ(p->exposure_min_ns, 8000000);
  EXPECT_EQ(p->exposure_max_ns, 8000000);
  EXPECT_DOUBLE_EQ(p->gain_min, 2.0);
  EXPECT_DOUBLE_EQ(p->gain_max, 2.0);
  EXPECT_DOUBLE_EQ(p->isp_gain_min, 1.0);
  EXPECT_DOUBLE_EQ(p->isp_gain_max, 1.0);
  EXPECT_TRUE(p->exposure_locked && p->gain_locked);
  EXPECT_FALSE(p->ae_lock_deferred);  // nothing for AE to converge
  EXPECT_TRUE(p->awb_lock_deferred);  // awb auto converges first, then locks
  EXPECT_TRUE(p->needs_settle());
  EXPECT_FALSE(p->denoise);
  EXPECT_FALSE(p->edge_enhance);
  EXPECT_EQ(p->wbmode, 1);
  EXPECT_EQ(p->fifo_length, 4);
}

TEST(CapturePlan, AeFreeDefersLockAndUsesModeRange) {
  CameraConfig c = uw_cfg();
  c.exposure_ns = 0;
  c.awb_mode = "off";
  auto p = build_capture_plan(c, mode_of(c));
  ASSERT_TRUE(p.ok());
  EXPECT_TRUE(p->ae_lock_deferred);
  EXPECT_FALSE(p->awb_lock_deferred);
  EXPECT_EQ(p->exposure_min_ns, 13000);
  // 30 fps on a mode whose maximum is 29.999999 fps (duration 33333334): clamped to the mode's minimum duration,
  // as Argus would (Settings.h:131-133); the exposure range is capped at that frame duration.
  EXPECT_EQ(p->frame_duration_ns, 33333334);
  EXPECT_EQ(p->exposure_max_ns, 33333334);
  EXPECT_EQ(p->wbmode, 0);
}

TEST(CapturePlan, Errors) {
  struct Case {
    void (*mutate)(CameraConfig&);
    const char* fragment;
  };
  const Case cases[] = {
      {[](CameraConfig& c) { c.exposure_ns = 20000000; }, "exceeds the frame duration"},
      {[](CameraConfig& c) { c.exposure_ns = 1000; }, "outside the mode range"},
      {[](CameraConfig& c) { c.analog_gain = 17.0; }, "analog_gain: 17 outside"},
      {[](CameraConfig& c) { c.isp_digital_gain = 0.5; }, "isp_digital_gain"},
      {[](CameraConfig& c) { c.awb_mode = "sunny"; }, "awb_mode: unknown 'sunny'"},
      {[](CameraConfig& c) { c.buffer_count = 0; }, "buffer_count"},
      {[](CameraConfig& c) { c.exposure_ns = -1; }, "exposure_ns: must be >= 0"},
  };
  for (const Case& t : cases) {
    CameraConfig c = main_cfg();
    const SensorMode m = mode_of(c);
    t.mutate(c);
    auto p = build_capture_plan(c, m);
    ASSERT_FALSE(p.ok()) << t.fragment;
    EXPECT_EQ(p.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(p.status().message().find(t.fragment), std::string::npos) << p.status().message();
    EXPECT_EQ(p.status().message().rfind("cameras.main.driver", 0), 0u) << p.status().message();
  }
}

TEST(CapturePlan, AwbNames) {
  EXPECT_EQ(awb_mode_index("off"), 0);
  EXPECT_EQ(awb_mode_index("daylight"), 5);
  EXPECT_EQ(awb_mode_index("cloudy_daylight"), 6);
  EXPECT_EQ(awb_mode_index("shade"), 8);
  EXPECT_EQ(awb_mode_index("x"), -1);
}

TEST(CapturePlan, GstNvargusPipeline) {
  const CameraConfig c = uw_cfg();
  auto p = build_capture_plan(c, mode_of(c)).value();
  EXPECT_EQ(build_gst_pipeline(c, p),
            "nvarguscamerasrc name=rig_src sensor-id=1 sensor-mode=3 exposuretimerange=\"4000000 4000000\" "
            "gainrange=\"4 4\" ispdigitalgainrange=\"1 1\" aelock=true awblock=false wbmode=1 tnr-mode=0 ee-mode=0 "
            "aeantibanding=0 ! video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1,format=NV12 ! "
            "nvvidconv ! video/x-raw,format=NV12 ! appsink name=rig_sink sync=false max-buffers=4 drop=false "
            "emit-signals=false");
  CameraConfig g = c;
  g.pixel_format = PixelFormat::kBGR8;
  g.denoise = true;
  const std::string s = build_gst_pipeline(g, build_capture_plan(g, mode_of(g)).value());
  EXPECT_NE(s.find("tnr-mode=1"), std::string::npos);
  EXPECT_NE(s.find("format=NV12 ! videoconvert ! video/x-raw,format=BGR ! appsink"), std::string::npos) << s;
}

TEST(CapturePlan, GstOverridePipeline) {
  CameraConfig c;
  c.id = "t";
  c.width = 320;
  c.height = 240;
  c.fps = 30.0;
  c.gst.source_override = "videotestsrc is-live=true";
  SensorMode m;
  m.min_fps = 1;
  m.max_fps = 30;
  auto p = build_capture_plan(c, m).value();
  EXPECT_EQ(build_gst_pipeline(c, p, "s"),
            "videotestsrc is-live=true ! videoconvert ! videoscale ! "
            "video/x-raw,format=NV12,width=320,height=240,framerate=30/1 ! appsink name=s sync=false max-buffers=4 "
            "drop=false emit-signals=false");
}

TEST(CapturePlan, Framerates) {
  EXPECT_EQ(gst_framerate(60.0), "60/1");
  EXPECT_EQ(gst_framerate(29.97002997), "30000/1001");
  EXPECT_EQ(gst_framerate(12.5), "12500/1000");
  EXPECT_EQ(gst_format_name(PixelFormat::kYUYV), "YUY2");
  EXPECT_EQ(gst_format_name(PixelFormat::kJPEG), "");
}

}  // namespace
}  // namespace rig
