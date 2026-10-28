// SPDX-License-Identifier: MIT
// ec-live (L1): pure model + renderer (live_view.hpp).
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>

#include <opencv2/imgproc.hpp>

#include "../../episode_collection/apps/ec-live/live_view.hpp"
#include "rig/config/rig_config.hpp"
#include "rig/imageprep/imageprep.hpp"

using namespace eclive;
using rig::TimeNs;

namespace {

constexpr TimeNs kMs = 1000000;
const char* kRigYaml = RIG_SOURCE_DIR "/episode_collection/configs/rig_jetson_imx519_imx219w160_lsm6dsox.yaml";

bool any_line_contains(const std::vector<TextLine>& v, const std::string& s, Severity* sev = nullptr) {
  for (const auto& l : v)
    if (l.text.find(s) != std::string::npos) {
      if (sev) *sev = l.severity;
      return true;
    }
  return false;
}

CameraView running_camera(TimeNs now) {
  CameraView c;
  c.id = "main";
  c.role = "main";
  c.backend = "argus";
  c.width = 1920;
  c.height = 1080;
  c.fps = 60;
  c.started = true;
  c.started_ns = now - 5000 * kMs;
  c.has_frame = true;
  c.frames_seen = 300;
  c.t_mid_ns = now - 20 * kMs;
  c.last_rx_ns = now - 10 * kMs;
  c.meta.exposure_ns = 8000000;
  c.meta.analog_gain = 2.0;
  c.meta.digital_gain = 1.0;
  c.meta.focus_status = rig::FocusStatus::kLockedVcmI2c;
  c.meta.focus_position = 512;
  c.meta.color_range = rig::ColorRange::kFull;
  c.measured_fps = 59.9;
  c.has_stats = true;
  return c;
}

/// NV12 frame with constant Y and CbCr.
rig::ImageFrame nv12_frame(int w, int h, uint8_t y, uint8_t cb, uint8_t cr, rig::ColorRange range) {
  rig::ImageFrame f;
  f.camera_id = "t";
  f.width = w;
  f.height = h;
  f.fmt = rig::PixelFormat::kNV12;
  const std::size_t ny = static_cast<std::size_t>(w) * h, nuv = static_cast<std::size_t>(w) * (h / 2);
  std::shared_ptr<uint8_t[]> buf(new uint8_t[ny + nuv]);
  std::memset(buf.get(), y, ny);
  for (std::size_t i = 0; i < nuv; i += 2) buf[ny + i] = cb, buf[ny + i + 1] = cr;
  f.planes[0] = {0, static_cast<std::size_t>(w), w, h};
  f.planes[1] = {ny, static_cast<std::size_t>(w), w / 2, h / 2};
  f.data = std::shared_ptr<const uint8_t[]>(buf);
  f.size = ny + nuv;
  f.t_sof_ns = f.t_mid_ns = f.t_arrival_ns = 1;
  f.meta.color_range = range;
  return f;
}

}  // namespace

TEST(RateMeter, OneSecondWindow) {
  RateMeter m;
  EXPECT_EQ(m.rate_hz(), 0.0);
  for (int i = 0; i <= 40; ++i) m.add(i * 50 * kMs, static_cast<uint64_t>(i) * 3);  // 60 Hz in 50 ms steps
  EXPECT_NEAR(m.rate_hz(), 60.0, 1e-9);
  // The window follows a rate change within ~1 s.
  for (int i = 41; i <= 70; ++i) m.add(i * 50 * kMs, 120 + static_cast<uint64_t>(i - 40) * 1);  // 20 Hz
  EXPECT_NEAR(m.rate_hz(), 20.0, 1e-9);
  // A counter that goes backwards (new service) resets the meter.
  m.add(71 * 50 * kMs, 0);
  EXPECT_EQ(m.rate_hz(), 0.0);
}

TEST(ImuHistory, KeepsFiveSecondsAndLatest) {
  ImuHistory h;
  EXPECT_FALSE(h.has_sample());
  EXPECT_TRUE(std::isnan(h.temperature_c()));
  for (int i = 0; i < 833 * 7; ++i) {
    rig::ImuSample s;
    s.t_ns = static_cast<TimeNs>(i) * 1200480;  // ~833 Hz
    s.acc_mps2 = {0, 0, 9.81};
    s.gyro_rps = {1.0, 0, 0};
    if (i % 66 == 0) s.temp_c = 30.0f + static_cast<float>(i) * 1e-4f;
    h.add(s);
  }
  const TimeNs now = static_cast<TimeNs>(833 * 7 - 1) * 1200480;
  h.prune(now);
  ASSERT_FALSE(h.points().empty());
  EXPECT_GE(h.points().front().t_ns, now - 5 * kSecondNs);
  EXPECT_NEAR(static_cast<double>(h.points().size()), 5 * 833.0, 3);
  EXPECT_NEAR(h.points().back().gyro_dps[0], 57.2958, 1e-3);  // rad/s -> deg/s
  EXPECT_FALSE(std::isnan(h.temperature_c()));                // last non-NaN kept
  // A sample without a time updates latest() but is not plotted.
  rig::ImuSample s;
  s.acc_mps2 = {1, 2, 3};
  const std::size_t n = h.points().size();
  h.add(s);
  EXPECT_EQ(h.points().size(), n);
  EXPECT_DOUBLE_EQ(h.latest().acc_mps2.y(), 2.0);
}

TEST(Layout, FitRectKeepsAspectAndCentres) {
  const cv::Rect r = fit_rect({1920, 1080}, {0, 0, 800, 600});
  EXPECT_EQ(r.width, 800);
  EXPECT_EQ(r.height, 450);
  EXPECT_EQ(r.y, 75);
  EXPECT_EQ(fit_rect({0, 0}, {5, 5, 10, 10}).area(), 0);
}

TEST(Layout, TwoCamerasAndImuStrip) {
  RenderOptions o;
  const Layout L = compute_layout(o, {{1920, 1080}, {1640, 1232}}, true);
  EXPECT_EQ(L.size.width, 1600);
  ASSERT_EQ(L.tiles.size(), 2u);
  EXPECT_FALSE((L.tiles[0] & L.tiles[1]).area() > 0);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ((L.images[i] & L.tiles[i]), L.images[i]);  // inside its tile
  }
  EXPECT_NEAR(static_cast<double>(L.images[0].width) / L.images[0].height, 1920.0 / 1080, 0.01);
  EXPECT_NEAR(static_cast<double>(L.images[1].width) / L.images[1].height, 1640.0 / 1232, 0.01);
  EXPECT_GE(L.imu.y, L.tiles[0].y + L.tiles[0].height);
  EXPECT_EQ(L.imu.height, o.imu_panel_height);
  EXPECT_EQ(L.size.height, L.imu.y + L.imu.height);
  // --width scales everything.
  o.width = 1000;
  const Layout S = compute_layout(o, {{1920, 1080}, {1640, 1232}}, false);
  EXPECT_EQ(S.size.width, 1000);
  EXPECT_EQ(S.imu.area(), 0);
  EXPECT_EQ(S.size.height, S.tiles[0].height);
}

TEST(Layout, SingleCameraCappedAndImuOnly) {
  RenderOptions o;
  const Layout one = compute_layout(o, {{1640, 1232}}, false);
  EXPECT_EQ(one.tiles[0].height, o.max_camera_height);
  EXPECT_EQ(one.images[0].height, o.max_camera_height);
  const Layout imu = compute_layout(o, {}, true);
  EXPECT_TRUE(imu.tiles.empty());
  EXPECT_EQ(imu.imu.y, 0);
  EXPECT_EQ(imu.size.height, o.imu_panel_height);
}

TEST(Overlay, CameraLinesShowEverything) {
  const TimeNs now = 100 * kSecondNs;
  CameraView c = running_camera(now);
  auto v = camera_overlay_lines(c, now);
  EXPECT_TRUE(any_line_contains(v, "main (main)  1920x1080@60  [argus]"));
  Severity s;
  ASSERT_TRUE(any_line_contains(v, "fps 59.9", &s));
  EXPECT_EQ(s, Severity::kOk);
  EXPECT_TRUE(any_line_contains(v, "drops 0"));
  EXPECT_TRUE(any_line_contains(v, "exp 8.00 ms"));
  EXPECT_TRUE(any_line_contains(v, "gain 2.00"));
  EXPECT_TRUE(any_line_contains(v, "focus vcm_i2c @512"));
  EXPECT_TRUE(any_line_contains(v, "range full"));
  EXPECT_TRUE(any_line_contains(v, "age 20 ms"));
  // Drops and a low rate turn the fps line yellow and show the breakdown.
  c.stats.drops_seq_gap = 2;
  c.stats.drops_time_gap = 1;
  c.measured_fps = 40;
  v = camera_overlay_lines(c, now);
  ASSERT_TRUE(any_line_contains(v, "drops 3", &s));
  EXPECT_EQ(s, Severity::kWarn);
  EXPECT_TRUE(any_line_contains(v, "seq 2  time 1"));
  // Unknown range is flagged; the ultrawide has no focus line.
  c.meta.color_range = rig::ColorRange::kUnknown;
  ASSERT_TRUE(any_line_contains(camera_overlay_lines(c, now), "range unknown", &s));
  EXPECT_EQ(s, Severity::kWarn);
  c.role = "ultrawide";
  c.meta.focus_status = rig::FocusStatus::kNotApplicable;
  EXPECT_FALSE(any_line_contains(camera_overlay_lines(c, now), "focus"));
  // Not started: only the header.
  c.started = false;
  EXPECT_EQ(camera_overlay_lines(c, now).size(), 1u);
}

TEST(Overlay, StallText) {
  const TimeNs now = 100 * kSecondNs;
  CameraView c = running_camera(now);
  EXPECT_EQ(stall_text(c, now, 1.0), "");
  c.last_rx_ns = now - 2500 * kMs;
  EXPECT_EQ(stall_text(c, now, 1.0), "NO FRAMES for 2.5 s");
  // Never a frame: counted from the start.
  c.has_frame = false;
  c.last_rx_ns = rig::kInvalidTime;
  c.started_ns = now - 3 * kSecondNs;
  EXPECT_EQ(stall_text(c, now, 1.0), "NO FRAMES for 3.0 s");
  c.started_ns = now - 500 * kMs;
  EXPECT_EQ(stall_text(c, now, 1.0), "");
  c.started = false;  // failed cameras show FAILED instead
  c.started_ns = now - 10 * kSecondNs;
  EXPECT_EQ(stall_text(c, now, 1.0), "");
}

TEST(Overlay, ImuPanelLines) {
  ImuView u;
  u.enabled = u.started = true;
  u.id = "imu0";
  u.backend = "lsm6dsox_i2c";
  u.odr_hz = 833;
  u.measured_rate_hz = 832.5;
  u.has_stats = true;
  u.stats.fifo_level_max = 40;
  u.stats.fifo_watermark_words = 256;
  u.stats.clock.residual_ns.count = 10;
  u.stats.clock.residual_ns.min = -20000;
  u.stats.clock.residual_ns.max = 35000;
  u.stats.clock.residual_ns.p99 = 30000;
  ImuHistory h;
  rig::ImuSample s;
  s.t_ns = 1;
  s.acc_mps2 = {0.0, 3.0, 4.0};
  s.gyro_rps = {0.5, 0, -1.0};
  s.temp_c = 31.25f;
  h.add(s);
  auto v = imu_panel_lines(u, h);
  EXPECT_TRUE(any_line_contains(v, "|a|  5.000 m/s2"));
  EXPECT_TRUE(any_line_contains(v, "gyro x   +28.6"));
  EXPECT_TRUE(any_line_contains(v, "z   -57.3   deg/s"));
  EXPECT_TRUE(any_line_contains(v, "temp 31.2 C"));  // %.1f of 31.25 (round-half-even)
  EXPECT_TRUE(any_line_contains(v, "rate 832.5 Hz"));
  Severity sev;
  ASSERT_TRUE(any_line_contains(v, "FIFO max 40/256", &sev));
  EXPECT_EQ(sev, Severity::kOk);
  EXPECT_TRUE(any_line_contains(v, "resid p99 30 us  max 35 us"));
  u.stats.fifo_overruns = 1;
  ASSERT_TRUE(any_line_contains(imu_panel_lines(u, h), "overruns fifo 1", &sev));
  EXPECT_EQ(sev, Severity::kBad);
  u.started = false;
  u.error = "io_error: /dev/i2c-7: Permission denied";
  v = imu_panel_lines(u, h);
  EXPECT_TRUE(any_line_contains(v, "FAILED TO START"));
  EXPECT_TRUE(any_line_contains(v, "Permission denied"));
}

TEST(Overlay, StatusLine) {
  const TimeNs now = 100 * kSecondNs;
  std::vector<CameraView> cams{running_camera(now), running_camera(now)};
  cams[1].id = "uw";
  cams[1].last_rx_ns = now - 4 * kSecondNs;
  CameraView failed;
  failed.id = "x";
  cams.push_back(failed);
  ImuView u;
  u.enabled = u.started = true;
  u.id = "imu0";
  u.measured_rate_hz = 833.2;
  ImuHistory h;
  rig::ImuSample s;
  s.t_ns = 1;
  s.acc_mps2 = {0, 0, 9.81};
  h.add(s);
  const std::string line = status_line(cams, u, h, now, now - 12 * kSecondNs, 1.0);
  EXPECT_NE(line.find("[   12s]"), std::string::npos) << line;
  EXPECT_NE(line.find("main 59.9fps drops 0 age 20ms"), std::string::npos) << line;
  EXPECT_NE(line.find("uw NO FRAMES for 4.0 s"), std::string::npos) << line;
  EXPECT_NE(line.find("x FAILED"), std::string::npos) << line;
  EXPECT_NE(line.find("imu0 833.2Hz |a| 9.81"), std::string::npos) << line;
  u.enabled = false;
  EXPECT_EQ(status_line(cams, u, h, now, now, 1.0).find("imu0"), std::string::npos);
}

TEST(Plot, NiceRange) {
  EXPECT_DOUBLE_EQ(plot_range(9.81, 12), 20);
  EXPECT_DOUBLE_EQ(plot_range(0, 12), 20);
  EXPECT_DOUBLE_EQ(plot_range(150, 20), 200);
  EXPECT_DOUBLE_EQ(plot_range(0.3, 0), 0.5);
  EXPECT_DOUBLE_EQ(plot_range(0, 0), 1);
  EXPECT_DOUBLE_EQ(plot_range(std::nan(""), 20), 20);
  EXPECT_DOUBLE_EQ(plot_range(1000, 20), 2000);
}

TEST(Text, WrapAndHints) {
  EXPECT_EQ(wrap_text("short", 10), std::vector<std::string>{"short"});
  const auto w = wrap_text("aaaa bbbb cccc dddd", 9);
  ASSERT_EQ(w.size(), 2u);
  EXPECT_EQ(w[0], "aaaa bbbb");
  EXPECT_EQ(w[1], "cccc dddd");
  EXPECT_EQ(wrap_text("abcdefghij", 4).size(), 3u);
  EXPECT_EQ(wrap_text("", 4), std::vector<std::string>{""});
  EXPECT_NE(failure_hint(true, "argus", rig::unimplemented("x")).find("RIG_WITH_ARGUS"), std::string::npos);
  EXPECT_NE(failure_hint(false, "lsm6dsox_i2c", rig::io_error("/dev/i2c-7: Permission denied")).find("i2c group"),
            std::string::npos);
  EXPECT_NE(failure_hint(false, "lsm6dsox_i2c", rig::device_error("WHO_AM_I (0x0f) = 0x00")).find("i2cdetect"),
            std::string::npos);
  EXPECT_NE(failure_hint(false, "lsm6dsox_i2c", rig::not_found("/dev/i2c-99: no such device")).find("ls /dev/i2c-*"),
            std::string::npos);
  EXPECT_EQ(failure_hint(true, "mock", rig::invalid_argument("x")), "");
}

TEST(Config, MockOverridesKeepSizes) {
  auto rc = rig::load_rig_config(kRigYaml);
  ASSERT_TRUE(rc.ok()) << rc.status().to_string();
  rig::RigConfig c = *rc;
  apply_mock_overrides(c);
  for (const auto& kv : c.cameras) {
    const rig::CameraConfig& before = rc->cameras.at(kv.first).driver;
    EXPECT_EQ(kv.second.driver.backend, rig::CameraBackendKind::kMock);
    EXPECT_EQ(kv.second.driver.width, before.width);
    EXPECT_EQ(kv.second.driver.height, before.height);
    EXPECT_EQ(kv.second.driver.fps, before.fps);
    EXPECT_EQ(kv.second.driver.exposure_ns, before.exposure_ns);
  }
  EXPECT_EQ(c.imu.driver.backend, rig::ImuBackendKind::kSim);
  EXPECT_EQ(c.imu.driver.odr_hz, rc->imu.driver.odr_hz);
  EXPECT_TRUE(c.imu.driver.sim.motion);
  // Still a valid config.
  EXPECT_TRUE(rig::parse_rig_config(rig::to_yaml(c)).ok());
  EXPECT_EQ(display_order(c), (std::vector<std::string>{"main", "uw"}));
}

TEST(Convert, FrameToDisplayHonoursColorRange) {
  // Y = 16, neutral chroma: limited range -> black (0), full range -> 16.
  for (const auto& [range, expect] :
       std::vector<std::pair<rig::ColorRange, int>>{{rig::ColorRange::kFull, 16}, {rig::ColorRange::kLimited, 0}}) {
    for (cv::Size target : {cv::Size(), cv::Size(480, 270), cv::Size(100, 56)}) {
      auto f = nv12_frame(960, 540, 16, 128, 128, range);
      auto r = frame_to_display(f, target);
      ASSERT_TRUE(r.ok()) << r.status().to_string();
      EXPECT_EQ(r->type(), CV_8UC3);
      EXPECT_EQ(r->size(), target.area() > 0 ? target : cv::Size(960, 540));
      const cv::Scalar m = cv::mean(*r);
      for (int c = 0; c < 3; ++c) EXPECT_NEAR(m[c], expect, 1.0) << rig::to_string(range) << " " << target;
    }
  }
}

TEST(Convert, HalvedPathMatchesDirect) {
  // A coloured gradient: the 2x YCbCr pre-downscale must look like converting first and downscaling after.
  auto f = nv12_frame(640, 480, 0, 0, 0, rig::ColorRange::kFull);
  auto* d = const_cast<uint8_t*>(f.data.get());
  for (int y = 0; y < 480; ++y)
    for (int x = 0; x < 640; ++x) d[y * 640 + x] = static_cast<uint8_t>((x + y) / 5);
  for (int y = 0; y < 240; ++y)
    for (int x = 0; x < 320; ++x) {
      d[640 * 480 + y * 640 + 2 * x] = static_cast<uint8_t>(64 + x / 3);
      d[640 * 480 + y * 640 + 2 * x + 1] = static_cast<uint8_t>(200 - y / 2);
    }
  auto direct = rig::frame_to_bgr(f);
  ASSERT_TRUE(direct.ok());
  cv::Mat expect;
  cv::resize(*direct, expect, {160, 120}, 0, 0, cv::INTER_AREA);
  auto got = frame_to_display(f, {160, 120});
  ASSERT_TRUE(got.ok());
  cv::Mat diff;
  cv::absdiff(expect, *got, diff);
  EXPECT_LT(cv::mean(diff)[0] + cv::mean(diff)[1] + cv::mean(diff)[2], 3 * 2.0);
  EXPECT_FALSE(frame_to_display(rig::ImageFrame{}, {}).ok());
}

TEST(Render, MosaicShowsTilesFailuresAndImu) {
  const TimeNs now = 100 * kSecondNs;
  RenderOptions o;
  std::vector<CameraView> cams{running_camera(now), running_camera(now), CameraView{}};
  cams[0].image = cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(0, 200, 0));
  cams[1].id = "uw";
  cams[1].width = 1640;
  cams[1].height = 1232;
  cams[1].has_frame = false;  // started, never delivered -> NO FRAMES
  cams[1].last_rx_ns = rig::kInvalidTime;
  cams[2].id = "x";
  cams[2].width = 640;
  cams[2].height = 480;
  cams[2].error = "unimplemented: argus not compiled in";
  const Layout L = compute_layout(o, {{1920, 1080}, {1640, 1232}, {640, 480}}, true);
  ImuView u;
  u.enabled = u.started = true;
  u.id = "imu0";
  ImuHistory h;
  for (int i = 0; i < 4000; ++i) {
    rig::ImuSample s;
    s.t_ns = now - 4 * kSecondNs + i * kMs;
    s.acc_mps2 = {std::sin(i * 0.01) * 5, 0, 9.81};
    s.gyro_rps = {0, std::cos(i * 0.01), 0};
    h.add(s);
  }
  const cv::Mat img = render_mosaic(L, cams, u, h, o, now);
  ASSERT_EQ(img.size(), L.size);
  ASSERT_EQ(img.type(), CV_8UC3);
  // Camera image drawn (green), away from the overlay box.
  const cv::Rect im = L.images[0];
  const cv::Vec3b px = img.at<cv::Vec3b>(im.y + im.height - 10, im.x + im.width / 2);
  EXPECT_EQ(px, cv::Vec3b(0, 200, 0));
  // NO FRAMES / FAILED banners are red text (B, G low, R high) somewhere in those tiles.
  auto has_red = [&](const cv::Rect& r) {
    cv::Mat m;
    cv::inRange(img(r), cv::Scalar(0, 0, 200), cv::Scalar(120, 120, 255), m);
    return cv::countNonZero(m) > 50;
  };
  EXPECT_TRUE(has_red(L.tiles[1]));
  EXPECT_TRUE(has_red(L.tiles[2]));
  EXPECT_FALSE(has_red(L.images[0]));
  // IMU strip: text and plots, i.e. not a flat colour.
  cv::Scalar mean, sd;
  cv::meanStdDev(img(L.imu), mean, sd);
  EXPECT_GT(sd[0] + sd[1] + sd[2], 10.0);
}
